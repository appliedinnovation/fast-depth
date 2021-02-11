import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.append(os.getcwd())
import utils
'''
Assumptions:
    - Data can be stored recursively in source folder.
    - Images are saved as '<id>_<number>.raw'
    - Depths are saved as '<id>_depth_motion_<number>.raw'

'''

parser = argparse.ArgumentParser(
    description='View raw images and depths stored in a folder')
parser.add_argument('-d',
                    type=str,
                    required=True,
                    help='Folder to copy images from.')
parser.add_argument('--height', type=int, default=384, help='Image height.')
parser.add_argument('--width', type=int, default=672, help='Image width.')
parser.add_argument('--channels', type=int, default=4, help='Image channels.')
parser.add_argument('--max-depth',
                    type=int,
                    default=30,
                    help="Maximum depth for visualization.")
args = parser.parse_args()

# Recursively list all data files
src_files = glob.glob(os.path.join(args.d, "**/*.raw"), recursive=True)

# Pair images and depths
paired_files = {}
for f in src_files:

    # Remove top-level source directory
    # relative_file_path = f.split(args.d)[-1]
    name = f.split("/")[-1]
    parent = f.split(args.d)[-1].split(name)[0]

    # Extract camera id and image number
    idx = name.split("_")[-1].split(".raw")[0]
    camera = name.split(idx)[0].split("depth_motion")[0].split(
        "depthFlow")[0][:-1]
    parent_camera = os.path.join(parent, camera + "_" + idx)

    if parent_camera not in paired_files:
        paired_files[parent_camera] = {}

    # Add image and depth to pair
    if "depth_motion" in name:
        paired_files[parent_camera]["depth"] = f
    elif "depthFlow" not in name:
        paired_files[parent_camera]["image"] = f

for data in paired_files.values():
    image = utils.raw_to_numpy(data["image"], args.height, args.width,
                               args.channels)[:, :, 0:3]
    depth = utils.raw_to_numpy(data["depth"], args.height, args.width,
                               args.channels)[:, :, 0]

    image = cv2.normalize(image,
                          None,
                          alpha=0,
                          beta=255,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)

    depth = utils.visualize_depth(depth, args.max_depth)

    stacked = np.hstack([image, depth])
    cv2.imshow(data["image"], stacked)
    k = cv2.waitKey(0)
    if k == ord('r'):
        os.remove(data["image"])
        os.remove(data["depth"])
        print("Removing {}\{}".format(data["image"], data["depth"]))
    cv2.destroyAllWindows()