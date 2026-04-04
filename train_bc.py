# --- train_bc.py ---
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Import the ActorCritic architecture exactly as the PPO agent sees it
from train_ppo import ActorCritic

# --- CONFIG ---
EXPERT_DATA_DIR = r"D:\python\crossy_learn\expert_run\pass"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-4  # Keep it low to prevent overfitting/memorization
WEIGHT_DECAY = 1e-5

console = Console()


class CrossyExpertDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        files = glob.glob(os.path.join(data_dir, "*.pt"))

        console.print(f"[cyan]Found {len(files)} expert trajectories. Loading...[/cyan]")

        for file in files:
            try:
                trajectory = torch.load(file, weights_only=False)
                for step in trajectory:
                    # SOTA DATA FILTERING:
                    # In record_runs_ram_expert.py, we set safety to [0,0,0,0] for river hazards.
                    # We skip these so the BC agent ONLY learns Road/Traffic mastery!
                    if sum(step['safety']) == 0.0:
                        continue

                    self.samples.append(step)
            except Exception as e:
                console.print(f"[red]Error loading {file}: {e}[/red]")

        console.print(f"[bold green]Successfully loaded {len(self.samples)} perfect road/vehicle frames![/bold green]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        latents = torch.FloatTensor(s['latents'])
        scalars = torch.FloatTensor(s['scalars'])
        mask = torch.FloatTensor(s['mask'])
        action = torch.tensor(s['action'], dtype=torch.long)

        return latents, scalars, mask, action


def train_behavioral_cloning():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold magenta]Starting Behavioral Cloning (Imitation Learning) on {device}[/bold magenta]")

    # 1. Prepare Data
    full_dataset = CrossyExpertDataset(EXPERT_DATA_DIR)

    if len(full_dataset) == 0:
        console.print("[bold red]No valid data found! Run the expert recorder first.[/bold red]")
        return

    # 90/10 Split to monitor for overfitting
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model (Action Dim = 4)
    model = ActorCritic(action_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        # We don't need the whole loop tracked, just a standard progress indicator
        for latents, scalars, masks, actions in train_loader:
            latents, scalars, masks, actions = latents.to(device), scalars.to(device), masks.to(device), actions.to(
                device)

            optimizer.zero_grad()

            # Forward pass exactly as it occurs in PPO evaluation
            features = model._get_features(latents, scalars)
            action_logits = model.actor(features)

            # Apply Action Masking so it doesn't get penalized for ignoring masked directions
            masked_logits = action_logits + masks

            loss = criterion(masked_logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(masked_logits, 1)
            total_train += actions.size(0)
            correct_train += (predicted == actions).sum().item()

        train_acc = 100 * correct_train / total_train

        # 4. Validation Loop
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for latents, scalars, masks, actions in val_loader:
                latents, scalars, masks, actions = latents.to(device), scalars.to(device), masks.to(device), actions.to(
                    device)

                features = model._get_features(latents, scalars)
                masked_logits = model.actor(features) + masks

                loss = criterion(masked_logits, actions)
                val_loss += loss.item()

                _, predicted = torch.max(masked_logits, 1)
                total_val += actions.size(0)
                correct_val += (predicted == actions).sum().item()

        val_acc = 100 * correct_val / total_val
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        console.print(f"Epoch [bold cyan]{epoch:02d}/{EPOCHS}[/bold cyan] | "
                      f"Train Loss: {avg_train_loss:.4f} | Train Acc: [green]{train_acc:.1f}%[/green] | "
                      f"Val Loss: {avg_val_loss:.4f} | Val Acc: [bold green]{val_acc:.1f}%[/bold green]")

        # 5. Save Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # We save this specifically as 'ppo_crossy_0.pth'.
            # In your train_ppo.py, any checkpoint index < 10 triggers the `is_bc_resume` logic!
            save_path = os.path.join(CHECKPOINT_DIR, "ppo_crossy_0.pth")
            torch.save(model.state_dict(), save_path)
            console.print(f"  [yellow]-> New best model saved to {save_path}[/yellow]")

    console.print("[bold magenta]Behavioral Cloning Complete![/bold magenta]")
    console.print(
        "You can now run [bold]train_ppo.py[/bold]. The PPO agent will load 'ppo_crossy_0.pth', freeze the Actor, and warm up the Critic!")


if __name__ == "__main__":
    train_behavioral_cloning()