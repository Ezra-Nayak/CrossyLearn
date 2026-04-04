import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np

from train_vision import setup_device

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 50
MAX_LR = 1e-3
NOISE = 0.05
DATA_DIR = r"D:\python\crossy_learn\expert_run\pass"
DEVICE = setup_device()


# ==========================================
# 1. SOTA ARCHITECTURE: RESNET + LAYERNORM
# ==========================================
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class SOTA_ActorCritic(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()

        # Spatial ResNet Backbone (Preserves complex car/log relationships)
        # Input: [128, 20, 20] -> Output:[256, 5, 5] -> GlobalAvgPool -> 256
        self.resnet = nn.Sequential(
            ResNetBlock(128, 128, stride=1),
            ResNetBlock(128, 256, stride=2),  # -> 10x10
            ResNetBlock(256, 256, stride=2),  # -> 5x5
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 1x1
            nn.Flatten()  # -> 256 flat features
        )

        cnn_out_dim = 256
        scalar_dim = 1

        # SOTA MLP Head: LayerNorm + GELU prevents dead neurons
        self.actor = nn.Sequential(
            nn.Linear(cnn_out_dim + scalar_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(cnn_out_dim + scalar_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def _get_features(self, latents, scalars):
        if latents.dim() == 3: latents = latents.unsqueeze(0)
        cnn_out = self.resnet(latents)
        if scalars.dim() == 1: scalars = scalars.unsqueeze(1)
        return torch.cat([cnn_out, scalars], dim=1)


# ==========================================
# 2. SOTA LOSS: SOFT FOCAL LOSS
# ==========================================
class SoftFocalLoss(nn.Module):
    """
    Dynamically scales loss based on confidence.
    Eliminates the need for manual class weights while solving the Idle vs Up imbalance.
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Standard Cross Entropy for Soft Targets
        ce_loss = -targets * torch.log(probs)

        # Focal Weighting: lowers loss for confident predictions
        focal_weight = torch.pow(1.0 - probs, self.gamma)

        loss = ce_loss * focal_weight
        return loss.sum(dim=-1).mean()


# ==========================================
# 3. SOTA REGULARIZATION: EMA WEIGHTS
# ==========================================
class EMA:
    """Keeps a moving average of weights for ultra-smooth, stable validation."""

    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ==========================================
# DATASET PROCESSING
# ==========================================
class ExpertDataset(Dataset):
    def __init__(self, files, snip_start_frames=3, snip_end_frames=10):
        self.clean_data = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        total_snipped = 0

        for f in files:
            trajectory = torch.load(f, weights_only=False)
            if len(trajectory) <= (snip_start_frames + snip_end_frames) + 5:
                total_snipped += len(trajectory)
                continue

            trimmed = trajectory[snip_start_frames: -snip_end_frames]
            total_snipped += (len(trajectory) - len(trimmed))

            for item in trimmed:
                lat = torch.FloatTensor(item['latents'])
                sca = torch.FloatTensor(item['scalars'])
                mask = torch.FloatTensor(item['mask'])
                act = int(item['action'])

                if mask[act] < -1:
                    act = 3 if mask[3] == 0 else 0

                safety = item.get('safety', None)
                target_prob = torch.zeros(4)

                if safety is not None:
                    valid_mask = (mask >= -1.0).float()
                    safety_t = torch.FloatTensor(safety) * valid_mask
                    safety_t[act] = 1.0

                    safe_count = safety_t.sum().item()
                    if safe_count > 1:
                        target_prob = safety_t * (0.3 / (safe_count - 1))
                        target_prob[act] = 0.7
                    else:
                        target_prob[act] = 1.0
                else:
                    valid_mask = (mask >= -1.0).float()
                    valid_count = valid_mask.sum().item()
                    if valid_count > 1:
                        target_prob = valid_mask * (0.15 / (valid_count - 1))
                        target_prob[act] = 0.85
                    else:
                        target_prob[act] = 1.0

                self.clean_data.append((lat, sca, mask, target_prob, act))
                action_counts[act] += 1

        print(f"[DATA] Loaded {len(files)} runs. Snipped {total_snipped} edge frames.")
        print(f"[DATA] Clean samples: {len(self.clean_data)}")
        print(
            f"[DATA] Actions -> Up: {action_counts[0]}, Left: {action_counts[1]}, Right: {action_counts[2]}, Idle: {action_counts[3]}")

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        return self.clean_data[idx]


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def train():
    os.makedirs("checkpoints", exist_ok=True)

    all_files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    np.random.shuffle(all_files)
    split_idx = int(0.85 * len(all_files))

    train_dataset = ExpertDataset(all_files[:split_idx])
    val_dataset = ExpertDataset(all_files[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SOTA_ActorCritic().to(DEVICE)
    ema = EMA(model)

    # AdamW incorporates true Weight Decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-2)

    # OneCycleLR dynamically pushes the LR up and down to escape local minima
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader),
                                              epochs=EPOCHS)

    criterion = SoftFocalLoss(gamma=2.0)

    print(f"\n--- STARTING SOTA IMITATION LEARNING ({len(train_dataset)} Train | {len(val_dataset)} Val) ---")

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss, correct_train = 0, 0

        for lat, sca, mask, target_prob, act in train_loader:
            lat, sca, mask, target_prob, act = lat.to(DEVICE), sca.to(DEVICE), mask.to(DEVICE), target_prob.to(
                DEVICE), act.to(DEVICE)

            # Gaussian Latent Noise Regularization
            lat = lat + (torch.randn_like(lat) * NOISE)

            # Horizontal Flip Augmentation
            if torch.rand(1).item() > 0.5:
                lat = torch.flip(lat, dims=[3])
                sca = -sca
                mask = mask[:, [0, 2, 1, 3]]
                target_prob = target_prob[:, [0, 2, 1, 3]]
                act_flipped = act.clone()
                act_flipped[act == 1] = 2
                act_flipped[act == 2] = 1
                act = act_flipped

            features = model._get_features(lat, sca)
            logits = model.actor(features)

            # Apply Action Masks before Focal Loss
            logits = logits + mask

            loss = criterion(logits, target_prob)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update()

            total_train_loss += loss.item()
            correct_train += (torch.argmax(logits, dim=1) == act).sum().item()

        # Validation (Using the EMA Shadow Weights)
        ema.apply_shadow()
        model.eval()
        total_val_loss, correct_val = 0, 0

        with torch.no_grad():
            for lat, sca, mask, target_prob, act in val_loader:
                lat, sca, mask, target_prob, act = lat.to(DEVICE), sca.to(DEVICE), mask.to(DEVICE), target_prob.to(
                    DEVICE), act.to(DEVICE)

                features = model._get_features(lat, sca)
                logits = model.actor(features) + mask

                loss = criterion(logits, target_prob)
                total_val_loss += loss.item()
                correct_val += (torch.argmax(logits, dim=1) == act).sum().item()

        ema.restore()  # Restore true weights for next training epoch

        train_acc = (correct_train / len(train_dataset)) * 100
        val_acc = (correct_val / len(val_dataset)) * 100

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {total_train_loss / len(train_loader):.4f} ({train_acc:.1f}%) | Val Loss: {total_val_loss / len(val_loader):.4f} ({val_acc:.1f}%) | LR: {scheduler.get_last_lr()[0]:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the EMA shadow weights as they are the most stable representation
            ema.apply_shadow()
            torch.save(model.state_dict(), "checkpoints/sota_bc_best.pth")
            ema.restore()

    print(f"\n[SUCCESS] Saved BEST model (Val Acc: {best_val_acc:.1f}%) to checkpoints/sota_bc_best.pth")


if __name__ == "__main__":
    train()