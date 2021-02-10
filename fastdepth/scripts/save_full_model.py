import os
import sys
sys.path.append(os.getcwd())
import torch
import torchvision
import argparse
import utils
import models

parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-m', '--model', type=str, required=True, help="Path to model.")
parser.add_argument('--resnet18', action='store_true')
parser.add_argument('--save-gpu', action='store_true')
parser.add_argument('--nyu', action='store_true')
args = parser.parse_args()

model_path = args.model

if args.nyu:
    checkpoint = torch.load(args.model)
    model = checkpoint['model']
else:
    model_state_dict, _, _, _ = utils.load_checkpoint(args.model)
    model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)
    if args.resnet18:
        model = models.ResNetSkipAdd(layers=18, output_size=(224, 224), pretrained=True)
    else:
        model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
    if model_state_dict:
        model.load_state_dict(model_state_dict)

if args.save_gpu:
    print("Saving model on GPU")
    model.to(torch.device("cuda:0"))
else:
    print("Saving model on CPU")
    model.to(torch.device("cpu"))

model_dir = os.path.join(*model_path.split('/')[:-1])
model_name = model_path.split('/')[-1]
device_ext = "gpu" if args.save_gpu else "cpu"

save_path = os.path.join(model_dir, 'full_model_' + model_name[:-4] + "_" + device_ext + ".pth")
torch.save(model, save_path)
print("Saved to ", save_path)
