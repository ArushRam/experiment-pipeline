import torch
import torch.nn as nn

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "none": nn.Identity(),
}

def get_activation(key):
    if key in ACTIVATION_MAP:
        return ACTIVATION_MAP[key]
    else:
        raise ValueError("Activation not found.")