import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

sys.path.append(os.getcwd())
import models
import utils

parser = argparse.ArgumentParser(
    description='Depth prediction from single image.')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help="Path to model.")
parser.add_argument('-i',
                    '--image',
                    type=str,
                    required=True,
                    help="Path to image JPG or PNG.")
parser.add_argument('-e',
                    '--encoder',
                    type=str,
                    default='resnet18',
                    help="Type of model encoder (resnet18 or mobilenet).")
parser.add_argument('--cpu', action='store_true', help="Flag to run inference on cpu.")
args = parser.parse_args()

if args.cpu:
    device_str = "cpu"
else:
    device_str = "cuda:0"
device = torch.device(device_str)

params = {"encoder": args.encoder}
model, _ = utils.load_model(params, args.model, device=device)
model.to(device)

to_tensor = transforms.ToTensor()

original = cv2.imread(args.image)
original = cv2.resize(original, (224, 224))
image = cv2.normalize(original, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC3)

input = to_tensor(image).to(device).unsqueeze(0)

depth = model(input)
depth = np.squeeze(depth.detach().cpu().numpy())

depth = cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
depth = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)

display = np.hstack([original, depth])
display = cv2.resize(display, (original.shape[0] * 4, original.shape[1] * 2))

save_path = os.path.dirname(args.image)
save_path = args.image.split('.png')[0].split('.jpg')[0]
cv2.imwrite(save_path + "_prediction.png", display)