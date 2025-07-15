import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import (pixel_to_cartesian, cartesian_to_pixel)

conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
conv.weight = torch.nn.Parameter(torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32), requires_grad=False)

def transform_segmentation(seg_map, T, scale=1.0, smoothe=False):
    """transform segmentation map with provided transformation matrix
    Arguments:
    segmentation_map -- a 2D, bitmap of object(s) to be segmented
    transformation_matrix -- a 3x3 tensor representing a homogenous 2D transformation matrix
    scale -- scaling factor of coordinates for numerical stability
    """
    # create the grid
    height, width = seg_map.shape[:2]
    y_indices, x_indices = torch.meshgrid(torch.arange(height, device=seg_map.device), torch.arange(width, device=seg_map.device), indexing='ij')
    x_indices, y_indices = pixel_to_cartesian(x_indices.float(), y_indices.float(), width, height, scale=scale)
    grid = torch.stack((x_indices, y_indices, torch.ones_like(x_indices)), dim=-1)

    # apply the transformation 
    t_grid = grid @ T.T

    # seg_coords 
    seg_coords = t_grid[seg_map.bool()] # seg_coords now (num_pixels, 3)
    #seg_coords = seg_coords.clone()

    seg_coords[:,0], seg_coords[:,1] = cartesian_to_pixel(seg_coords[:,0], seg_coords[:,1], width, height, scale=scale)
    seg_coords = seg_coords[:, :2].long()

    # collect only valid coordinates
    seg_coords = seg_coords[(seg_coords[:, 0] >= 0) & (seg_coords[:, 0] < width) & (seg_coords[:, 1] >= 0) & (seg_coords[:, 1] < height)]

    t_seg_map = torch.zeros(height, width, device=seg_map.device, dtype=torch.bool)
    # seg_coords are transposed here due to indexing order
    # coords are [[x,y], [x,y], ...], indexes are [[x,x], [y,y], ...]
    t_seg_map[[seg_coords[:,1], seg_coords[:,0]]] = True

    if smoothe:
        # TODO xhin - there is likely a better approach
        with torch.no_grad():
            conv_out = conv(t_seg_map.unsqueeze(0).unsqueeze(0))
        t_seg_map[conv_out.squeeze(0).squeeze(0) > 2] = 1
    
    return t_seg_map