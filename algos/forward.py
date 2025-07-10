import torch
import matplotlib.pyplot as pQt
from dataclasses import dataclass
import tyro
from render import transform_segmentation
from utils import (rotation_matrix_2d, load_link)

# todo xhin - have to represent the robot using json, usd would be better but idk
@dataclass
class Args:
    link1: str = "../assets/link1.jpg"
    link2: str = "../assets/link2.jpg"
    link3: str = "../assets/link3.jpg"
    n: int = 3
    """Number of links for robot"""


base_y = -255
link_height = 145
bt1 = torch.eye(3).float()
bt1[:2,-1] = torch.tensor([0, base_y])
bt2 = torch.eye(3).float()
bt2[:2,-1] = torch.tensor([0, link_height])
bt3 = torch.eye(3).float()
bt3[:2,-1] = torch.tensor([0, link_height])
BTs = [bt1, bt2, bt3]

# TODO xhin - currently linear structure - would like to convert to tree structure
# TODO xhin - add velocity
def forward_kinematics(link_frames, qpos, link_segs, colors):
    """Performs forward kinematics to determine the worldframe views of each link"""

    # base transformations will be
    # BT1: link1 to world
    # BT2: link2 to link1 
    # i.e. link2 to world is BT1 @ BT2 
    # full transformation will be (BT1 @ Qt1) @ (BT2 @ Qt2)
    # where Qti is the local transformation of link i - a 2d homogeneous transformation matrix
    
    final_transforms = []
    cur_transform = torch.eye(3).float()
    for i in range(len(link_segs)): 
        Qt = rotation_matrix_2d(qpos[i])
        cur_transform =  cur_transform @ link_frames[i] @ Qt 
        final_transforms.append(cur_transform)

    final_render = torch.ones(*link_segs[0].shape[:2], 3)
    for i in range(len(link_segs)):
        segmentation = transform_segmentation(link_segs[i], final_transforms[i])
        color = torch.zeros(3)
        color[colors[i]] = 255
        final_render[segmentation.bool()] = color
    
    return final_render

if __name__ == "__main__":
    args = tyro.cli(Args)
    link_segs = []
    colors = []
    links = [x for x in dir(args) if "link" in x]
    for i in range(args.n):
        link_seg, color = load_link(getattr(args, links[i]))
        if link_seg is None:
            exit(1)
        link_segs.append(link_seg)
        colors.append(color)

    qpos = [torch.pi/10,-torch.pi/2,torch.pi/2]
    render = forward_kinematics(BTs, qpos, link_segs, colors)
    pQt.imshow(render)
    pQt.show()

    



