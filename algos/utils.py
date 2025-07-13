import torch
import matplotlib.pyplot as plt
import numpy as np

def rotation_matrix_2d(theta, device=None):
    if isinstance(theta, float) or isinstance(theta, int):
        theta = torch.tensor(theta, device=device)
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0,                 0,              1.0],
    ], device=device)

def load_link(path):
    """Load a link image from the specified path."""
    try:
        img = plt.imread(path)
        return torch.tensor(img[:,:,:3].sum(-1) > 1)
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None