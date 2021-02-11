import argparse
import glob
import os
import random
import sys
from shutil import copy

import numpy as np

sys.path.append(os.getcwd())

'''
Assumptions:
    - Data can be stored recursively in source folder.
    - Images are saved as '<id>_<number>.raw'
    - Depths are saved as '<id>_depth_motion_<number>.raw'

'''

parser = argparse.ArgumentParser(
    description='Randomly copy files from a dataset folder into another folder'
)
parser.add_argument('-d',
                    type=str,
                    required=True,
                    help='Folder to copy images from.')
parser.add_argument('-s',
                    type=str,
                    required=True,
                    help='Folder to copy images to.')
parser.add_argument('-n',
                    type=int,
                    required=True,
                    help='Number of images to copy.')
args = parser.parse_args()

if not os.path.exists(args.s):
    os.makedirs(args.s)

# Recursively list all data files
src_files = glob.glob(os.path.join(args.d, "**/*.raw"), recursive=True)

# Pair images and depths
paired_files = {}
for f in src_files:

    # Remove top-level source directory
    relative_file_path = f.split(args.d)[-1]
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
        paired_files[parent_camera]["depth"] = relative_file_path
    elif "depthFlow" not in name:
        paired_files[parent_camera]["image"] = relative_file_path

# Random sample of images
random_sample = random.sample(paired_files.items(), args.n)

# Copy random sample to destination
for sample in random_sample:
    if "image" in sample[1] and "depth" in sample[1]:
        isrc = os.path.join(args.d, sample[1]["image"])
        idst = os.path.join(args.s, sample[1]["image"])
        dsrc = os.path.join(args.d, sample[1]["depth"])
        ddst = os.path.join(args.s, sample[1]["depth"])

        # Make directory if needed
        relative_file_path = idst.rpartition('/')[0]
        if not os.path.exists(relative_file_path):
            os.makedirs(relative_file_path, exist_ok=True)

        copy(isrc, idst)
        copy(dsrc, ddst)
