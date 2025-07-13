import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
import tyro
from render import transform_segmentation
from utils import (rotation_matrix_2d, load_link)
import time

# todo xhin - have to represent the robot using json, usd would be better but idk
@dataclass
class Args:
    link1: str = "../assets/link1.jpg"
    link2: str = "../assets/link2.jpg"
    link3: str = "../assets/link3.jpg"
    n: int = 3
    fps: int = 60
    """Number of links for robot"""

# TODO xhin - robot description file + parser 
# or just a separate robot.py file
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
# TODO xhin - this function should only return the list/tree of transformations + end-effector poses
# delegate the rest to the render file
def forward_kinematics(link_frames, qpos, link_segs, colors):
    """Performs forward kinematics to determine the worldframe views of each link"""

    # base transformations will be
    # BT1: link1 to world
    # BT2: link2 to link1 
    # i.e. link2 to world is BT1 @ BT2 
    # full transformation will be (BT1 @ Qt1) @ (BT2 @ Qt2) @ ... @ (BTN @ QtN)
    # where Qti is the local transformation of link i - a 2d homogeneous transformation matrix
    
    final_transforms = []
    cur_transform = torch.eye(3).float()
    for i in range(len(link_segs)): 
        Qt = rotation_matrix_2d(qpos[i])
        cur_transform =  cur_transform @ link_frames[i] @ Qt 
        final_transforms.append(cur_transform)

    final_render = torch.ones(*link_segs[0].shape[:2], 3, dtype=torch.uint8) * 255 
    for i in range(len(link_segs)):
        segmentation = transform_segmentation(link_segs[i], final_transforms[i])
        color = torch.zeros(3, dtype=torch.uint8)
        color[colors[i]] = 255
        final_render[segmentation.bool()] = color
    
    return final_render


# TODO xhin - move this functionality to a playground/testing file
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

    qpos = [-torch.pi/4, torch.pi/4, torch.pi/2]  # Initial positions
    fig, ax = plt.subplots()
    img = ax.imshow(torch.ones(*link_segs[0].shape[:2], 3, dtype=torch.uint8) * 255)

    for key in list(mpl.rcParams):
        if key.startswith("keymap."):
            mpl.rcParams[key] = []

    end_loop = False
    ctrl_keys = ['q', 'w', 'e', 'r', 't', 'y']
    def on_press(event):
        global end_loop
        if event.key == 'escape':
            end_loop = True
        if not event.key in ctrl_keys:
            return
        index = ctrl_keys.index(event.key)
        qpos[index // 2] -= (2 * (index % 2) - 1) * 0.1  # Increment or decrement the joint angle

    fig.canvas.mpl_connect('key_press_event', on_press)

    while not end_loop: 
        render = forward_kinematics(BTs, qpos, link_segs, colors)
        img.set_data(render.numpy())
        plt.pause(1/args.fps)  # Pause to simulate animation




# [[1,0,x],# [[1,2,0], 
# [0,1,y], # [4,5,0],
# [0,0,1]] # [0,0,1]]


# [[1,0,x],# [[1,2,0],  1 2 x
# [0,1,y], # [4,5,0],
# [0,0,1]] # [0,0,1]]


# ,# [[1,2,0], [[1,0,x]   1 2 (x + 2y)
# , #[4,5,0],   [0,1,y]
#  # [0,0,1]]   [0,0,1]]
