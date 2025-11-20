import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Basic Multilayer Perceptron (MLP) template.
    - Supports variable hidden layers
    - Optional dropout
    - Optional activation selection
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if activation not in self.activations:
            raise ValueError(f"Unknown activation: {activation}")

        act = self.activations[activation]

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (optional customization)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
