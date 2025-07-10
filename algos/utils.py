import torch
import matplotlib.pyplot as plt
import numpy as np

def rotation_matrix_2d(theta):
    if isinstance(theta, float) or isinstance(theta, int):
        theta = torch.tensor(theta)
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0,0,1.0]
    ])

def load_link(path):
    """Load a link image from the specified path."""
    try:
        img = plt.imread(path)
        colors = [np.sum(img[:,:, 0]), np.sum(img[:,:, 1]), np.sum(img[:,:, 2])]
        color = np.argmax(colors)

        return torch.tensor(img > 0).sum(dim=-1), color
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None, None