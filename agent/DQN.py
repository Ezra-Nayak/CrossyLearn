import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_device():
    """Checks for DirectML availability and sets the device accordingly."""
    try:
        import torch_directml
        if torch_directml.is_available():
            dml_device = torch_directml.device()
            print(f"Using DirectML device: {dml_device}")
            return dml_device
        else:
            print("DirectML is not available. Falling back to CPU.")
            return torch.device("cpu")
    except (ImportError, Exception) as e:
        print(f"DirectML not found or failed ({e}). Falling back to CPU.")
        return torch.device("cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        'ReLU' (Rectified Linear Unit) is a standard activation function
        that introduces non-linearity, allowing the network to learn
        more complex patterns.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)