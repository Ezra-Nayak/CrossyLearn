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
BATCH_SIZE = 128        # Crank this up for stability
LR = 1e-4               # Slightly lower for fine-tuning
EPOCHS = 1000            # Let it run for a while
LATENT_DIM = 64
IMG_SIZE = 160
STACK_SIZE = 4
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


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


class CrossyVisionDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.data = []
        print(f"[LOADER] Found {len(self.files)} chunk files.")

        for f in self.files:
            try:
                chunk = np.load(f, allow_pickle=True)
                self.data.extend(chunk)
            except Exception as e:
                print(f"[ERROR] Failed to load {f}: {e}")

        print(f"[LOADER] Total sequences loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_stack, target_frame = self.data[idx]
        target_frame = np.expand_dims(target_frame, axis=0)
        return torch.FloatTensor(input_stack), torch.FloatTensor(target_frame)


class SplitBrainVAE(nn.Module):
    def __init__(self):
        super(SplitBrainVAE, self).__init__()

        # --- ENCODER (160x160 Input) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 10x10
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # 256 * 10 * 10 = 25600
        self.flatten_size = 25600
        self.split_dim = LATENT_DIM // 2

        self.fc_mu_c = nn.Linear(self.flatten_size, self.split_dim)
        self.fc_log_c = nn.Linear(self.flatten_size, self.split_dim)
        self.fc_mu_t = nn.Linear(self.flatten_size, self.split_dim)
        self.fc_log_t = nn.Linear(self.flatten_size, self.split_dim)

        # --- DECODER ---
        self.decoder_input = nn.Linear(self.split_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 160x160
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 10, 10)  # Reshape to 10x10 feature map
        return self.decoder(x)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu_c, log_c = self.fc_mu_c(h), self.fc_log_c(h)
        mu_t, log_t = self.fc_mu_t(h), self.fc_log_t(h)

        z_c = self.reparameterize(mu_c, log_c)
        z_t = self.reparameterize(mu_t, log_t)

        recon_static = self.decode(z_c)
        pred_next = self.decode(z_t)

        return recon_static, pred_next, mu_c, log_c, mu_t, log_t


def verify_vae(model, val_loader, epoch):
    model.eval()
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(DEVICE)
        recon, pred, _, _, _, _ = model(inputs)

        # Grab first sample
        img_stack = inputs[0].cpu().numpy()
        img_recon = recon[0].cpu().numpy().squeeze()
        img_pred = pred[0].cpu().numpy().squeeze()
        img_target = targets[0].cpu().numpy().squeeze()

        fig, axs = plt.subplots(2, 4, figsize=(14, 7))

        # Row 1: Inputs
        for i in range(4):
            axs[0, i].imshow(img_stack[i], cmap='gray')
            axs[0, i].set_title(f"Input t-{3 - i}")
            axs[0, i].axis('off')

        # Row 2: Outputs
        axs[1, 0].imshow(img_stack[3], cmap='gray')
        axs[1, 0].set_title("Current (Reference)")
        axs[1, 0].axis('off')

        axs[1, 1].imshow(img_recon, cmap='gray')
        axs[1, 1].set_title("Context (Recon)")
        axs[1, 1].axis('off')

        axs[1, 2].imshow(img_pred, cmap='gray')
        axs[1, 2].set_title("Trend (Prediction)")
        axs[1, 2].axis('off')

        axs[1, 3].imshow(img_target, cmap='gray')
        axs[1, 3].set_title("Target (Next Frame)")
        axs[1, 3].axis('off')

        plt.suptitle(f"Epoch {epoch} | 160px Split-Brain", color='blue')
        plt.tight_layout()
        plt.savefig(f"{LOG_DIR}/verify_epoch_{epoch}.png")
        plt.close()


def train():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    dataset = CrossyVisionDataset("data")
    if len(dataset) < 500:
        print(f"[WARNING] Only {len(dataset)} samples found. VAE requires 2000+ for good results.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = SplitBrainVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"[TRAIN] Starting on {DEVICE} with {len(dataset)} samples.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            current_frame = inputs[:, 3:4, :, :]

            recon, pred, mu_c, log_c, mu_t, log_t = model(inputs)

            loss_context = F.mse_loss(recon, current_frame)
            loss_trend = F.mse_loss(pred, targets)

            kld = -0.5 * torch.sum(1 + log_c - mu_c.pow(2) - log_c.exp())
            kld += -0.5 * torch.sum(1 + log_t - mu_t.pow(2) - log_t.exp())
            kld /= (BATCH_SIZE * 2)

            # EXPERIMENTAL: Increased KLD to structure the latent space better.
            # If images turn gray/checkerboard, lower this back to 0.00001
            KLD_WEIGHT = 0.00001
            loss = loss_context + (2.0 * loss_trend) + (KLD_WEIGHT * kld)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f}")

        if epoch % 50 == 0 or epoch == 1:
            verify_vae(model, dataloader, epoch)

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/crossy_vae_ep{epoch}.pth")
            print(f"[SAVE] Permanent checkpoint saved at epoch {epoch}")
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/crossy_vae_latest_KLD.pth")


if __name__ == "__main__":
    train()