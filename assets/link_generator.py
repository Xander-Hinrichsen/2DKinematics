from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Args:
    # Example arguments, customize as needed
    output_file: str = "link.png"  # --output-file
    resolution: int = 512  # --resolution
    link_height: int = 150
    link_width: int = 16

if __name__ == "__main__":
    args = tyro.cli(Args)
    img = np.zeros((args.resolution, args.resolution), dtype=np.uint8)
    half_res = args.resolution // 2
    img[max(0,half_res-args.link_height):half_res, half_res-args.link_width:half_res+args.link_width] = 255
    plt.imsave(args.output_file, img)

