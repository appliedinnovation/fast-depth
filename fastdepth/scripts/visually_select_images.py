import argparse
import glob
import os
import sys
from shutil import move

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
parser.add_argument('-s', '--src',
                    type=str,
                    required=True,
                    help='Folder to move images from.')
parser.add_argument('-d', '--dst', type=str, required=True, help="Folder to move images to.")
parser.add_argument('--height', type=int, default=384, help='Image height.')
parser.add_argument('--width', type=int, default=672, help='Image width.')
parser.add_argument('--channels', type=int, default=4, help='Image channels.')
parser.add_argument('--max-depth',
                    type=int,
                    default=30,
                    help="Maximum depth for visualization.")
args = parser.parse_args()

if not os.path.exists(args.dst):
    os.makedirs(args.dst)

# Recursively list all data files
src_files = glob.glob(os.path.join(args.src, "**/*.raw"), recursive=True)

# Pair images and depths
paired_files = {}
for f in src_files:

    # Remove top-level source directory
    relative_file_path = f.split(args.src)[-1]
    name = f.split("/")[-1]
    parent = f.split(args.src)[-1].split(name)[0]

    # Extract camera id and image number
    idx = name.split("_")[-1].split(".raw")[0]
    camera = name.split(idx)[0].split("depth_motion")[0].split(
        "depthFlow")[0][:-1]
    parent_camera = os.path.join(parent, camera + "_" + idx)

    if parent_camera not in paired_files:
        paired_files[parent_camera] = {}

    # Add image and depth to pair
    if "depth_motion" in name:
        paired_files[parent_camera]["depth"] = relative_file_path
    elif "depthFlow" not in name:
        paired_files[parent_camera]["image"] = relative_file_path

for data in paired_files.values():
    isrc = os.path.join(args.src, data["image"])
    idst = os.path.join(args.dst, data["image"])
    dsrc = os.path.join(args.src, data["depth"])
    ddst = os.path.join(args.dst, data["depth"])

    image = utils.raw_to_numpy(isrc, args.height, args.width,
                               args.channels)[:, :, 0:3]
    depth = utils.raw_to_numpy(dsrc, args.height, args.width,
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
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('m'):
        dst_file_parent = idst.rpartition('/')[0]
        if not os.path.exists(dst_file_parent):
            os.makedirs(dst_file_parent, exist_ok=True)
        move(isrc, idst)
        move(dsrc, ddst)

    cv2.destroyAllWindows()
