import torch
import matplotlib.pyplot as plt
import numpy as np

def pixel_to_cartesian(x,y,w,h, scale=1.0):
    y = -(y - h // 2) * scale
    x = (x - w // 2) * scale
    return x,y

def cartesian_to_pixel(x,y,w,h, scale=1.0):
    y = -(1./scale) * y + h // 2 
    x  = (1./scale) * x + w // 2
    return x, y

def rot_2d(theta, device=None):
    if isinstance(theta, float) or isinstance(theta, int):
        theta = torch.tensor(theta, device=device)
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0,                 0,              1.0],
    ], device=device)


def drot_2d_dtheta(theta, device=None):
    """Derivative of 2D rotation matrix with respect to theta"""
    omega = torch.tensor([
        [0, -1., 0],
        [1., 0,  0],
        [0,  0,  0],
    ], device=device)
    return rot_2d(theta, device=device) @ omega

def load_link(path):
    """Load a link image from the specified path."""
    try:
        img = plt.imread(path)
        return torch.tensor(img[:,:,:3].sum(-1) > 1)
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None