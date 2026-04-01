import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np

from train_ppo import ActorCritic, PPO_DEVICE

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4

NOISE = 0.05

DATA_DIR = r"D:\python\crossy_learn\expert_run" # !!!!!!!~~~~~~~~~~~~


class ExpertDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, snip_start_frames=3, snip_end_frames=10):
        files = glob.glob(os.path.join(data_dir, "*.pt"))

        self.clean_data =[]
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        total_snipped = 0

        for f in files:
            trajectory = torch.load(f, weights_only=False)

            # --- TRAJECTORY TRIMMING ---
            # The RAM Bot environment already waits for UI to clear during reset.
            # We only snip the first 5 frames (camera settling) and last 3 frames (imminent collision state).
            if len(trajectory) <= (snip_start_frames + snip_end_frames) + 5:
                total_snipped += len(trajectory)
                continue  # Skip runs that are too short to be meaningful

            trimmed_trajectory = trajectory[snip_start_frames: -snip_end_frames]
            total_snipped += (len(trajectory) - len(trimmed_trajectory))

            for item in trimmed_trajectory:
                lat = torch.FloatTensor(item['latents'])
                sca = torch.FloatTensor(item['scalars'])
                mask = torch.FloatTensor(item['mask'])
                act = int(item['action'])

                # --- DATA CLEANING ---
                # Force label to IDLE if a blocked key was somehow recorded
                if mask[act] < -1:
                    act = 3 if mask[3] == 0 else 0

                self.clean_data.append((lat, sca, mask, act))
                action_counts[act] += 1

        print(f"[DATA] Loaded {len(files)} runs. Snipped {total_snipped} edge frames.")
        print(f"[DATA] Clean samples: {len(self.clean_data)}")
        print(
            f"[DATA] Action Distribution -> Up: {action_counts[0]}, Left: {action_counts[1]}, Right: {action_counts[2]}, Idle: {action_counts[3]}")

        # --- CLASS WEIGHTING ---
        # Prevent the AI from just learning to "Stand Still" since most frames are IDLE.
        total_samples = len(self.clean_data)
        self.class_weights = torch.zeros(4, device=PPO_DEVICE)
        for i in range(4):
            # Inverse frequency weighting
            if action_counts[i] > 0:
                self.class_weights[i] = total_samples / (4.0 * action_counts[i])
            else:
                self.class_weights[i] = 1.0

        print(f"[DATA] Class Weights applied to penalize Idle-spamming: {self.class_weights.cpu().numpy().round(2)}")

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        return self.clean_data[idx]


def get_next_checkpoint_name():
    checkpoints = glob.glob("checkpoints/ppo_crossy_*.pth")
    if not checkpoints:
        return "checkpoints/ppo_crossy_1.pth"
    latest_num = max([int(x.split('_')[-1].split('.')[0]) for x in checkpoints])
    return f"checkpoints/ppo_crossy_{latest_num + 1}.pth"


def train():
    os.makedirs("checkpoints", exist_ok=True)
    dataset = ExpertDataset()

    if len(dataset) == 0:
        print("[ERROR] No expert data found. Run record_expert.py first!")
        return

    # --- TRAIN / VALIDATION SPLIT ---
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ActorCritic(action_dim=4).to(PPO_DEVICE)

    checkpoints = glob.glob("checkpoints/ppo_crossy_*.pth")
    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model.load_state_dict(torch.load(latest_cp, map_location=PPO_DEVICE, weights_only=False))
        print(f"[LOAD] Fine-tuning existing brain: {latest_cp}")
    else:
        print("[LOAD] Starting with a fresh brain.")

    # Added Weight Decay (L2 Regularization) to heavily penalize memorization
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=dataset.class_weights)

    print(f"\n--- STARTING BEHAVIORAL CLONING ({train_size} Train | {val_size} Val) ---")

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        correct_train = 0

        for lat, sca, mask, act in train_loader:
            lat, sca, mask, act = lat.to(PPO_DEVICE), sca.to(PPO_DEVICE), mask.to(PPO_DEVICE), act.to(PPO_DEVICE)

            # --- DATA AUGMENTATION (Latent Noise) ---
            # Inject 5% Gaussian noise into the latent space.
            # This acts as a heavy regularizer to completely obliterate memorization.
            noise = torch.randn_like(lat) * NOISE
            lat = lat + noise

            # --- DATA AUGMENTATION (Horizontal Flip) ---
            # 50% chance to flip the screen and swap Left/Right actions
            if torch.rand(1).item() > 0.5:
                # Latents shape is [B, 128, 20, 20]. Flip the width dimension (dim=3).
                lat = torch.flip(lat, dims=[3])

                # CRITICAL FIX: Invert the scalar X coordinate!
                # If the screen flips, a chicken on the left (-X) is now on the right (+X).
                sca = -sca

                # Swap mask limits for Left (1) and Right (2)
                mask = mask[:, [0, 2, 1, 3]]

                # Swap human actions
                act_flipped = act.clone()
                act_flipped[act == 1] = 2
                act_flipped[act == 2] = 1
                act = act_flipped

            features = model._get_features(lat, sca)
            logits = model.actor(features)

            logits = logits + mask
            loss = criterion(logits, act)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_train += (predictions == act).sum().item()

        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for lat, sca, mask, act in val_loader:
                lat, sca, mask, act = lat.to(PPO_DEVICE), sca.to(PPO_DEVICE), mask.to(PPO_DEVICE), act.to(PPO_DEVICE)

                # No augmentation during validation
                features = model._get_features(lat, sca)
                logits = model.actor(features)
                logits = logits + mask

                loss = criterion(logits, act)
                total_val_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                correct_val += (predictions == act).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = (correct_train / train_size) * 100

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (correct_val / val_size) * 100

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%)")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    save_path = get_next_checkpoint_name()
    # Save the BEST validation model, not the final overfit one
    torch.save(best_model_state, save_path)

    print(f"\n[SUCCESS] Saved BEST model (Val Loss: {best_val_loss:.4f}) to {save_path}")
    print("[INFO] train_ppo.py will automatically resume from here and generate the Critic values.")


if __name__ == "__main__":
    train()