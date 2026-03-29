# --- train_vision.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.optim import Optimizer


class DMLAdam(Optimizer):
    """
    DirectML-Compatible Adam Optimizer.
    Replaces the 'lerp' operator (unsupported on DML GPU) with basic add/mul.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DMLAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad

                # Prevent Overfitting: L2 Regularization
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # SOTA Fix: Replace lerp_ with mul_ and add_ for DML support
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

# --- CONFIG ---
BATCH_SIZE = 128        # Crank this up for stability
LR = 1e-4               # Slightly lower for fine-tuning
EPOCHS = 500            # Let it run for a while
LATENT_DIM = 256
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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: [Batch, Channels, Height, Width]
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate distances to find nearest codebook vectors
        distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # --- SOTA FIX: DEAD CODE REVIVAL (DML Optimized) ---
        if self.training:
            # Create a temporary one-hot matrix to track usage across the batch
            # We use the 2D scatter pattern which is confirmed working on your DML setup
            usage_map = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            usage_map.scatter_(1, encoding_indices, 1)

            # Sum across all pixels in the batch to get usage per codebook entry
            usage_count = torch.sum(usage_map, dim=0)
            dead_codes = torch.nonzero(usage_count == 0).squeeze(-1)

            if dead_codes.numel() > 0:
                # Teleport dead codes directly onto random latent patches from this batch
                rand_indices = torch.randint(0, flat_inputs.shape[0], (dead_codes.numel(),), device=inputs.device)

                # Update codebook weights with actual latent data (Revival)
                self.embedding.weight.data[dead_codes] = flat_inputs[rand_indices].detach()

                # Recompute distances so the final quantization uses the new codes immediately
                distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                             + torch.sum(self.embedding.weight ** 2, dim=1)
                             - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))
                encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # -----------------------------------

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the latents
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2],
                                                                        inputs.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Loss calculation (Codebook loss + Commitment loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator (passes gradients through the argmin)
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity (Metric to ensure the whole codebook is being used, not just 1 or 2 codes)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class SpatialVQVAE(nn.Module):
    def __init__(self):
        super(SpatialVQVAE, self).__init__()

        self.num_embeddings = 512
        self.embedding_dim = 64

        # --- ENCODER (160x160 Input -> 10x10 Spatial Grid) ---
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

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 10x10, 128 Channels (64 Context + 64 Trend)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Discrete Codebooks (One for Context Brain, One for Trend Brain)
        self.vq_c = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.vq_t = VectorQuantizer(self.num_embeddings, self.embedding_dim)

        # --- DECODER (10x10 Grid -> 160x160 Image) ---
        # Note: We NO LONGER FLATTEN. The decoder reconstructs directly from the spatial 10x10 map!
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, 128, kernel_size=4, stride=2, padding=1),  # 20x20
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

    def forward(self, x):
        h = self.encoder(x)  # Shape:[B, 128, 10, 10]

        # Split the 128 channels physically down the middle: 64 for context, 64 for trend
        z_c, z_t = torch.split(h, self.embedding_dim, dim=1)

        # Snap spatial embeddings to nearest Codebook Tokens
        quantized_c, vq_loss_c, perplexity_c = self.vq_c(z_c)
        quantized_t, vq_loss_t, perplexity_t = self.vq_t(z_t)

        recon_static = self.decoder(quantized_c)
        pred_next = self.decoder(quantized_t)

        return recon_static, pred_next, vq_loss_c, vq_loss_t, perplexity_c, perplexity_t


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
    dataset_size = len(dataset)
    if dataset_size < 500:
        print(f"[WARNING] Only {dataset_size} samples found. VAE requires 2000+ for good results.")

    # Prevent Overfitting: Train/Validation Split
    val_size = max(1, int(dataset_size * 0.1))  # 10% for validation
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_batch = min(BATCH_SIZE, val_size)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False, drop_last=False)

    model = SpatialVQVAE().to(DEVICE)

    # Resume Training Logic
    start_epoch = 1
    latest_ckpt = f"{CHECKPOINT_DIR}/crossy_vae_latest.pth"
    if os.path.exists(latest_ckpt):
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
            print(f"[TRAIN] Resumed weights from {latest_ckpt}.")
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {e}")

    # SOTA: Use custom DMLAdam with weight_decay for L2 Regularization (Overfitting Prevention)
    optimizer = DMLAdam(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"[TRAIN] Starting on {DEVICE} with {train_size} train and {val_size} val samples.")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Prevent Overfitting: Random Horizontal Flip (Doubles effective dataset size)
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, dims=[3])
                targets = torch.flip(targets, dims=[3])

            current_frame = inputs[:, 3:4, :, :]

            # Prevent Overfitting: Stronger Denoising & Spatial Dropout
            noise = torch.randn_like(inputs) * 0.1
            noisy_inputs = torch.clamp(inputs + noise, 0.0, 1.0)

            mask = (torch.rand_like(inputs[:, 0:1, :, :]) > 0.05).float()
            noisy_inputs = noisy_inputs * mask

            recon, pred, vq_loss_c, vq_loss_t, perp_c, perp_t = model(noisy_inputs)

            # Perceptual Spatial loss
            loss_context = F.l1_loss(recon, current_frame)
            loss_trend = F.l1_loss(pred, targets)

            # KLD is replaced natively by Vector Quantization Commitment Loss
            loss = loss_context + (2.0 * loss_trend) + vq_loss_c + vq_loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                val_current = val_inputs[:, 3:4, :, :]

                v_recon, v_pred, v_vq_loss_c, v_vq_loss_t, v_perp_c, v_perp_t = model(val_inputs)

                v_loss_context = F.l1_loss(v_recon, val_current)
                v_loss_trend = F.l1_loss(v_pred, val_targets)

                v_loss = v_loss_context + (2.0 * v_loss_trend) + v_vq_loss_c + v_vq_loss_t
                val_loss += v_loss.item()

        val_loss /= len(val_loader)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            verify_vae(model, val_loader, epoch)

        # Save Best Validation Model (Early Stopping / Prevent Overfitting)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/crossy_vae_best.pth")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/crossy_vae_ep{epoch}.pth")
            print(f"[SAVE] Permanent checkpoint saved at epoch {epoch}")

        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/crossy_vae_latest.pth")


if __name__ == "__main__":
    train()