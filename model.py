# --- model.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def setup_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"[SYSTEM] Using DirectML device: {device}")
            return device
    except ImportError:
        pass
    print("[SYSTEM] Using CPU (DirectML not found or failed)")
    return torch.device("cpu")


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Value Stream (How good is the state?)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage Stream (How good is this action compared to others?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine (Q = V + (A - mean(A)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals