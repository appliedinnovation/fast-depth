import os
import sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import cv2
import utils
import torch

# Import custom dataset
try:
    dataset_path = os.environ["DATASETS_ABS_PATH"]
except KeyError:
    print("Datasets.py absolute path not found in PATH")
sys.path.append(dataset_path)
import Datasets

parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-s',
                    type=str,
                    required=True,
                    help='Folder to save selected images.')
parser.add_argument('-d',
                    type=str,
                    required=True,
                    help='Folder to select images from.')
args = parser.parse_args()

if not os.path.exists(args.s):
    os.makedirs(args.s)

# Find existing images so to not overwrite them
latest_idx = 0
image_paths = sorted(os.listdir(args.s))
if image_paths:
    latest_idx = len(image_paths)

dataset = Datasets.FastDepthDataset([args.d],
                                    split='train',
                                    depth_min=0.1,
                                    depth_max=80,
                                    input_shape_model=(224, 224))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=1,
                                          pin_memory=True)

# Find a way to copy raw files over

for i, (input, target) in enumerate(data_loader):

    # Convert to np array
    input = utils.tensor_to_rgb(input)
    input = cv2.resize(input, (input.shape[0] * 5, input.shape[1] * 5))
    cv2.imshow("Image", input)
    cv2.waitKey(2000)

    filename = os.path.join(args.s, "image_000{}.png".format(latest_idx + i))
    cv2.imwrite(filename, input)
