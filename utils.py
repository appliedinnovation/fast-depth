import os
import cv2
import torch
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import datetime

cmap = plt.cm.viridis


def parse_command():
    data_names = ['nyudepthv2',
                  'unreal']

    from dataloaders.dataloader import MyDataloader
    modality_names = MyDataloader.modality_names

    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH',)
    parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
    parser.add_argument('--resume', type=str, default=None, help="Path to model checkpoint to resume training.")
    parser.add_argument('-n', '--num_photos_saved', type=int, default=1, help="Number of comparison photos to save during evaluation.")
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def visualize_depth(depth):
    # so the image isn't all white, convert it to range [0, 1.0]
    _mean, _std = (np.mean(depth), np.std(depth))
    _min, _max = (np.min(depth), np.max(depth))
    
    # print('Depth (min, max):', (_min, _max))
    # print('Depth (mean, std):', (_mean, _std))

    newMax = _mean + 2 * _std
    newMin = _mean - 2 * _std
    if newMax < _max:
        _max = newMax
    if newMin > _min:
        _min = newMin
    _range = _max-_min
    if _range:
        depth -= _min
        depth /= _range
    
    # Convert to bgr
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    # Color mapping for better visibility / contrast
    depth = np.array(depth * 255, dtype=np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    return depth

def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    # d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    # d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    # depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    # depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    depth_target_col = visualize_depth(depth_target_cpu)
    depth_pred_col = visualize_depth(depth_pred_cpu)

    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def load_training_parameters(file):
    if not os.path.isfile(file):
        raise ValueError("Parameters file does not exist")

    params = json.load(open(file))
    return params["dataset_paths"], \
            params["train_val_split"], \
            params["depth_min"], \
            params["depth_max"], \
            params["batch_size"], \
            params["num_workers"], \
            params["gpu"], \
            params["loss"], \
            params["optimizer"], \
            params["num_epochs"], \
            params["stats_frequency"], \
            params["save_frequency"], \
            params["save_dir"], \
            params["max_checkpoints"]


def format_dataset_path(dataset_paths):
    if isinstance(dataset_paths, str):
        data_paths = {
            dataset_paths
        }
    elif isinstance(dataset_paths, list):
        data_paths = set()
        for path in dataset_paths:
            data_paths.add(path)
    
    return data_paths


def make_dir_with_date(root_dir, prefix):
    time = datetime.datetime.now()
    date_dir = os.path.join(root_dir, prefix + "_" + time.strftime("%m_%d_%H_%M"))
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    return date_dir


def get_train_val_split_lengths(train_val_split, dataset_length):
    return [int(np.around(train_val_split[0] * 0.01 * dataset_length)), \
            int(np.around(train_val_split[1] * 0.01 * dataset_length))]


def load_checkpoint(model_path):
    if model_path and os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))

        checkpoint = torch.load(model_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_result']

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_path, checkpoint['epoch']))
        
    else:
        model_state_dict = None
        optimizer_state_dict = None
        start_epoch = 0
        best_loss = 100000 # Very high number

        if model_path:
            print("=> no checkpoint found at '{}'".format(model_path))

    return model_state_dict,\
            optimizer_state_dict,\
            start_epoch,\
            best_loss,\


def get_save_path(epoch, save_dir="./results"):
    time = datetime.datetime.now()
    save_path = os.path.join(save_dir, "fastdepth_{time}_epoch_{epoch}.pth".format(time=time.strftime("%m_%d_%H_%M"),
                                                                                   epoch=str(epoch).zfill(4)))

    return save_path


def save_model(model, optimizer, save_path, epoch, loss, max_checkpoints=None):
    if max_checkpoints:
        checkpoint_dir = os.path.split(save_path)[0]
        checkpoints = sorted(os.listdir(checkpoint_dir))
        while len(checkpoints) >= max_checkpoints:
            # Remove oldest checkpoints
            os.remove(os.path.join(checkpoint_dir, checkpoints[0]))
            checkpoints.pop(0)

    torch.save({
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "epoch" : epoch,
            "best_result" : loss
            }, save_path)


def optimizer_to(device, optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.device == "cpu":
                state[k] = v.cuda()


def save_losses_plot(path, num_epochs, losses, title):
    x = np.arange(1, num_epochs + 1, 1)
    plt.plot(x, losses)
    plt.xticks(x, x)
    plt.xlabel("Epochs")
    plt.ylabel("{} Loss (m)".format(title))
    plt.title("{} Loss".format(title))
    plt.savefig(path, bbox_inches="tight")
