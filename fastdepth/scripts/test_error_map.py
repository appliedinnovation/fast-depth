import os
import sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import cv2
import models
import utils
import torch
from torchvision import transforms


parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-m', '--model', type=str,
                    required=True, help="Path to model.")
parser.add_argument('--resnet18', action='store_true')
parser.add_argument('-i', default=0, type=int)
args = parser.parse_args()


h = 384
w = 672
base_path = "/data/datasets/unreal_ldr_sample/"
img = np.fromfile(base_path + "front_{}.raw".format(args.i),
                  np.dtype(('f4', 4)), h * w).reshape(h, w, 4)
img = img[:, :, 0:3]

depth = np.fromfile(base_path + "front_depth_motion_{}.raw".format(args.i),
                    np.dtype(('f4', 4)), h * w).reshape(h, w, 4)
depth = depth[:, :, 0]
depth[depth > 80] = 80

device = torch.device("cuda:0")
model_state_dict, _, _ = utils.load_checkpoint(args.model)
model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)
if args.resnet18:
    model = models.ResNetSkipAdd(
        layers=18, output_size=(224, 224), pretrained=True)
else:
    model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
if model_state_dict:
    model.load_state_dict(model_state_dict)
model.to(device)

# Preprocess
to_tensor = transforms.ToTensor()
h = 224
w = 224
img = cv2.resize(img, (h, w))
img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
depth = cv2.resize(depth, (h, w))
input = to_tensor(img).to(device).unsqueeze(0)
target = torch.unsqueeze(torch.from_numpy(depth), 0)

# Prediction
prediction = model(input).cpu()

# Error Map
prediction[prediction > 25] = 25
target[target > 25] = 25
abs_diff = (prediction - target).abs()
error_map = np.squeeze(abs_diff.detach().cpu().numpy())

error_map = utils.normalize_new_range(error_map)
error_map = cv2.normalize(error_map, error_map, 0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U)
error_map = cv2.cvtColor(error_map, cv2.COLOR_GRAY2BGR)
error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_HOT)

nframe = np.array([])
nframe = cv2.normalize(img, nframe, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Target
depth = cv2.normalize(depth, depth, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
depth = np.array(depth * 255, dtype=np.uint8)
depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)

# Prediction
prediction = np.squeeze(prediction.detach().cpu().numpy())
prediction = cv2.normalize(prediction, prediction, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
prediction= np.array(prediction * 255, dtype=np.uint8)
prediction = cv2.applyColorMap(prediction, cv2.COLORMAP_TURBO)

display = np.hstack([nframe, depth, prediction, error_map])
display = cv2.resize(display, (2000, 500))

cv2.imshow("Display", display)
cv2.waitKey(10000)
cv2.imwrite("results/error_map_example.jpg", display)
