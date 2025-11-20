import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import get_activation

class SimpleCNN(nn.Module):
    def __init__(self, num_filters=16, kernel_size=5, dropout=0.2, activation='gelu', output_activation='none', **kwargs):
        super().__init__()
        
        padding = kernel_size // 2  # keep length the same
        print(num_filters, kernel_size, activation, output_activation)

        self.conv = nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=padding)
        self.activation = get_activation(activation)
        self.batchnorm = nn.BatchNorm1d(num_filters)

        # global *average* pooling over sequence length (not sum)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters, num_filters)
        self.fc2 = nn.Linear(num_filters, 1)
        self.output_activation = get_activation(output_activation)

    def forward(self, x_seq, x_struct):
        x = x_seq
        x = self.activation(self.conv(x))      # (batch, C, L)
        x = self.batchnorm(x)
        x = x.mean(dim=2)           # global average pooling → (batch, C)
        x = self.dropout(x)
        x = self.fc2(F.gelu(self.fc1(x)))              # (batch, 1)
        x = x.squeeze(-1)
        return self.output_activation(x)
    
    def get_regularization_penalty(self):
        return torch.zeros(1)[0]