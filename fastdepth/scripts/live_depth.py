import os
import sys
sys.path.append(os.getcwd())
import torch
import torchvision
import argparse
import utils
import models
import cv2
from torch.utils import data
from torchvision import transforms
import numpy as np

# global display
def draw_circle(event,x,y,flags,param):
    global unscaled_depth
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Depth: {} m".format(unscaled_depth[y][x]))

parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-m', '--model', type=str, required=True, help="Path to model.")
parser.add_argument('-s', type=str, help="Video file name to save.")
args = parser.parse_args()

model_path = args.model

model_state_dict, _, _, _ = utils.load_checkpoint(args.model)
model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)
model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
if model_state_dict:
    model.load_state_dict(model_state_dict)
device = torch.device("cuda:0")
model.to(device)

cap = cv2.VideoCapture(0)
to_tensor = transforms.ToTensor()

if args.s:
    out = cv2.VideoWriter("live_depth.avi", cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (224*2, 224))

display = np.array([])
cv2.namedWindow('Live Depth')
cv2.setMouseCallback('Live Depth',draw_circle, display)

while True:
    _, frame = cap.read()
    frame = cv2.normalize(frame, frame, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    frame = cv2.resize(frame, (224, 224))
    
    input = to_tensor(frame).to(device).unsqueeze(0)

    depth_pred = model(input)

    nframe = np.array([])
    nframe = cv2.normalize(frame, nframe, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    unscaled_depth = np.squeeze(depth_pred.data.cpu().numpy())
    unscaled_depth[unscaled_depth > 25.0] = 25.0
    
    depth = unscaled_depth.copy()
    depth = cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    depth = np.array(depth * 255, dtype=np.uint8)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    
    display = np.hstack([nframe, depth])
    
    cv2.imshow("Live Depth", depth)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    
    if args.s:
        out.write(display)
