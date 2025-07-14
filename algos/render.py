import torch
import matplotlib.pyplot as plt
import numpy as np
#from utils import rot_2d

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

    y_indices = -(y_indices.float() - height // 2) * scale
    x_indices = (x_indices.float() - width // 2) * scale

    grid = torch.stack((x_indices, y_indices, torch.ones_like(x_indices)), dim=-1)

    # apply the transformation 
    t_grid = grid @ T.T
    #t_grid = grid

    # seg_coords 
    seg_coords = t_grid[seg_map.bool()] # seg_coords now (num_pixels, 3)
    #seg_coords = seg_coords.clone()

    seg_coords[:, 1] = (-(1/scale)*seg_coords[:, 1] + height // 2)
    seg_coords[:, 0] = (1/scale)*seg_coords[:,0] + width // 2
    seg_coords = seg_coords[:, :2].long()

    # collect only valid coordinates
    seg_coords = seg_coords[(seg_coords[:, 0] >= 0) & (seg_coords[:, 0] < width) & (seg_coords[:, 1] >= 0) & (seg_coords[:, 1] < height)]

    # h = y, w = x - have to swap back
    #seg_coords[:, [0,1]] = seg_coords[:, [1,0]]

    t_seg_map = torch.zeros(height, width, device=seg_map.device, dtype=torch.bool)
    # seg_coords are transposed here due to indexing order
    # coords are [[x,y], [x,y], ...], indexes are [[x,x], [y,y], ...]
    #t_seg_map[*seg_coords.long().T] = 1
    t_seg_map[[seg_coords[:,1], seg_coords[:,0]]] = True

    if smoothe:
        # TODO xhin - there is likely a better approach
        # heuristic-byhand approach
        with torch.no_grad():
            conv_out = conv(t_seg_map.unsqueeze(0).unsqueeze(0))
        t_seg_map[conv_out.squeeze(0).squeeze(0) > 2] = 1
    
    return t_seg_map

# seg_map = torch.zeros(7,7)
# seg_map[1,1] = 1
# seg_map[1,2] = 1
# seg_map[1,3] = 1


# t_seg = transform_segmentation(seg_map, rot_2d(torch.pi/2))
# plt.imshow(seg_map);plt.show()
# plt.imshow(t_seg);plt.show()