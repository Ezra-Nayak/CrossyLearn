# --- train_vision.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# --- CONFIG ---
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
LATENT_DIM = 32  # 16 Context + 16 Trend
IMG_SIZE = 128
STACK_SIZE = 4
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


# --- DEVICE SETUP (AMD SUPPORT) ---
def setup_device():
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[SYSTEM] Using DirectML Device: {device}")
        return device
    except ImportError:
        print("[SYSTEM] DirectML not found. Using CPU.")
        return torch.device("cpu")


DEVICE = setup_device()


# --- DATASET ---
class CrossyVisionDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.data = []
        print(f"[LOADER] Found {len(self.files)} chunk files.")

        for f in self.files:
            try:
                # Allow pickle for object arrays (tuples)
                chunk = np.load(f, allow_pickle=True)
                self.data.extend(chunk)
            except Exception as e:
                print(f"[ERROR] Failed to load {f}: {e}")

        print(f"[LOADER] Total sequences loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # input_stack: (4, 128, 128), target: (128, 128)
        input_stack, target_frame = self.data[idx]

        # Add channel dim to target for loss calculation: (1, 128, 128)
        target_frame = np.expand_dims(target_frame, axis=0)

        return torch.FloatTensor(input_stack), torch.FloatTensor(target_frame)


# --- SPLIT-BRAIN VAE MODEL ---
class SplitBrainVAE(nn.Module):
    def __init__(self):
        super(SplitBrainVAE, self).__init__()

        # --- ENCODER (Takes 4 stacked frames) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.flatten_size = 256 * 8 * 8
        self.split_dim = LATENT_DIM // 2

        # Heads for Context (Static)
        self.fc_mu_c = nn.Linear(self.flatten_size, self.split_dim)
        self.fc_log_c = nn.Linear(self.flatten_size, self.split_dim)

        # Heads for Trend (Motion/Next Frame)
        self.fc_mu_t = nn.Linear(self.flatten_size, self.split_dim)
        self.fc_log_t = nn.Linear(self.flatten_size, self.split_dim)

        # --- DECODER (Takes Context + Trend to predict ONE frame) ---
        # Note: In a pure oracle, Trend predicts Next, Context reconstructs Current.
        # Here we use a shared decoder structure but will call it twice.

        self.decoder_input = nn.Linear(self.split_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid()  # Normalize 0-1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        # x shape: (B, 4, 128, 128)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        # Split Brain
        mu_c, log_c = self.fc_mu_c(h), self.fc_log_c(h)
        mu_t, log_t = self.fc_mu_t(h), self.fc_log_t(h)

        z_c = self.reparameterize(mu_c, log_c)
        z_t = self.reparameterize(mu_t, log_t)

        # Reconstruction (Using Context) -> Should look like Current Frame (Index 3 of input)
        recon_static = self.decode(z_c)

        # Prediction (Using Trend) -> Should look like Next Frame (Target)
        pred_next = self.decode(z_t)

        return recon_static, pred_next, mu_c, log_c, mu_t, log_t


# --- VERIFICATION FUNCTION ---
def verify_vae(model, val_loader, epoch):
    """
    Saves a grid:
    [Input t-3] [Input t-2] [Input t-1] [Input t]
    [Recon t]   [Pred t+1]  [Actual t+1]
    """
    model.eval()
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(DEVICE)

        # Forward pass
        recon, pred, _, _, _, _ = model(inputs)

        # Take first item in batch
        img_stack = inputs[0].cpu().numpy()  # (4, 128, 128)
        img_recon = recon[0].cpu().numpy().squeeze()  # (128, 128)
        img_pred = pred[0].cpu().numpy().squeeze()
        img_target = targets[0].cpu().numpy().squeeze()

        fig, axs = plt.subplots(2, 4, figsize=(12, 6))

        # Row 1: The Input Stack
        for i in range(4):
            axs[0, i].imshow(img_stack[i], cmap='gray')
            axs[0, i].set_title(f"Input t-{3 - i}")
            axs[0, i].axis('off')

        # Row 2: Results
        axs[1, 0].imshow(img_stack[3], cmap='gray')
        axs[1, 0].set_title("Input t (Ref)")
        axs[1, 0].axis('off')

        axs[1, 1].imshow(img_recon, cmap='gray')
        axs[1, 1].set_title("Context Recon (t)")
        axs[1, 1].axis('off')

        axs[1, 2].imshow(img_pred, cmap='gray')
        axs[1, 2].set_title("Trend Pred (t+1)")
        axs[1, 2].axis('off')

        axs[1, 3].imshow(img_target, cmap='gray')
        axs[1, 3].set_title("Actual (t+1)")
        axs[1, 3].axis('off')

        plt.suptitle(f"Epoch {epoch} - Split-Brain Vision", color='blue')
        plt.tight_layout()
        plt.savefig(f"{LOG_DIR}/verify_epoch_{epoch}.png")
        plt.close()


# --- TRAIN LOOP ---
def train():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    dataset = CrossyVisionDataset("data")
    if len(dataset) == 0:
        print("[ERROR] No data found. Run collect_data.py first!")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = SplitBrainVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"[TRAIN] Starting on {DEVICE} with {len(dataset)} samples.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Input Stack: [t-3, t-2, t-1, t]
            current_frame = inputs[:, 3:4, :, :]  # Extract t for Context Loss

            recon, pred, mu_c, log_c, mu_t, log_t = model(inputs)

            # 1. Context Loss: Does Head A understand what the screen looks like right now?
            loss_context = F.mse_loss(recon, current_frame)

            # 2. Trend Loss: Does Head B understand where cars will be in t+1?
            loss_trend = F.mse_loss(pred, targets)

            # 3. KLD (Regularization)
            kld_loss = -0.5 * torch.sum(1 + log_c - mu_c.pow(2) - log_c.exp())
            kld_loss += -0.5 * torch.sum(1 + log_t - mu_t.pow(2) - log_t.exp())
            kld_loss /= BATCH_SIZE * 2  # Normalize

            loss = loss_context + (2.0 * loss_trend) + (0.0001 * kld_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"\rEpoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} (C:{loss_context:.4f} T:{loss_trend:.4f})",
                    end="")

        print(f"\n[INFO] Epoch {epoch} Complete. Avg Loss: {total_loss / len(dataloader):.4f}")

        # Verify and Save
        verify_vae(model, dataloader, epoch)
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/crossy_vae_latest.pth")


if __name__ == "__main__":
    train()