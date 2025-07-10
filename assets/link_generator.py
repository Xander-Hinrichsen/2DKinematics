from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Args:
    # Example arguments, customize as needed
    output_file: str = "link.jpg"  # --output-file
    resolution: int = 512  # --resolution
    link_height: int = 150
    link_width: int = 10
    color: str = "red"  # --color

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    img = np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8)
    match args.color.lower():
        case 'red':
            color = 0
        case 'green':
            color = 1
        case _:
            color = 2 

    half_res = args.resolution // 2
    img[max(0,half_res-args.link_height):half_res, half_res-args.link_width:half_res+args.link_width, color] = 255
    plt.imsave(args.output_file, img)

