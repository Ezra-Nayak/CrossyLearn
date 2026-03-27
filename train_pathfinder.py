# --- train_pathfinder.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from train_vision import SplitBrainVAE, setup_device

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-3
VAE_CHECKPOINT = "checkpoints/crossy_vae_latest.pth"
DEVICE = setup_device()


class PathDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.data = []
        for f in self.files:
            try:
                # (InputStack, CurrentXY, FutureFlatVector)
                chunk = np.load(f, allow_pickle=True)
                self.data.extend(chunk)
            except:
                pass
        print(f"Loaded {len(self.data)} path samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stack, curr_pos, future_pts = self.data[idx]
        return (torch.FloatTensor(stack),
                torch.FloatTensor(curr_pos),
                torch.FloatTensor(future_pts))


class Pathfinder(nn.Module):
    def __init__(self):
        super(Pathfinder, self).__init__()
        # Input: 64 (VAE Latents) + 2 (Current XY) = 66
        self.net = nn.Sequential(
            nn.Linear(66, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # Output: 3 points * (x, y) = 6 coords
        )

    def forward(self, latents, current_pos):
        x = torch.cat([latents, current_pos], dim=1)
        return self.net(x)


def train():
    # 1. Load VAE (Frozen)
    vae = SplitBrainVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE))
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False

    # 2. Setup Pathfinder
    dataset = PathDataset("data_pathfinder")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Pathfinder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("--- TRAINING PATHFINDER ---")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for stack, curr_pos, target_path in loader:
            stack = stack.to(DEVICE)
            curr_pos = curr_pos.to(DEVICE)
            target_path = target_path.to(DEVICE)

            # Get VAE Latents
            with torch.no_grad():
                _, _, mu_c, _, mu_t, _ = vae(stack)
                latents = torch.cat([mu_c, mu_t], dim=1)

            # Predict Path
            pred_path = model(latents, curr_pos)

            loss = criterion(pred_path, target_path)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.5f}")

        if epoch % 20 == 0:
            torch.save(model.state_dict(), "imp_backups/pathfinder_latest.pth")


if __name__ == "__main__":
    train()