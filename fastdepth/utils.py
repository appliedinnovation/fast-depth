import os
import sys
import cv2
import torch
from torchvision import transforms
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import datetime
from collections import OrderedDict
import models

sys.path.append(os.getcwd())

cmap = plt.cm.viridis

def raw_to_numpy(path, height, width, channels):
    return np.fromfile(path, np.dtype(('f4', channels)), height * width).reshape(height, width, channels)    

def parse_command():
    data_names = ['nyudepthv2',
                  'unreal']

    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--gpu', default='0', type=str,
                        metavar='N', help="gpu id")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to model checkpoint to resume training.")
    parser.add_argument('-n', '--num_photos_saved', type=int, default=1,
                        help="Number of comparison photos to save during evaluation.")
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


# Returns normalized image between 0 and 1
def normalize_new_range(input):
    _mean, _std = (np.mean(input), np.std(input))
    _min = np.min(input)
    _max = np.max(input)
    
    newMax = _mean + 2 * _std
    newMin = _mean - 2 * _std
    if newMax < _max:
        _max = newMax
    if newMin > _min:
        _min = newMin
    input[input > _max] = _max
    input[input < _min] = _min
    _range = _max-_min
    input -= _min
    input /= _range

    return input


def visualize_depth(depth, far_clip=None):
    if far_clip:
        depth[depth > far_clip] = far_clip
    
    depth = normalize_new_range(depth)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Convert to bgr
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    # Color mapping for better visibility / contrast
    # depth = np.array(depth * 255, dtype=np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    return depth


def visualize_depth_compare(depth, target):
    _min = min(np.min(target), np.min(depth))
    _max = max(np.max(target), np.max(depth))

    _range = _max-_min
    if _range:
        depth -= _min
        depth /= _range
        target -= _min
        target /= _range

    # Convert to bgr
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    # Color mapping for better visibility / contrast
    depth = np.array(depth * 255, dtype=np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    target = np.array(target * 255, dtype=np.uint8)
    target = cv2.applyColorMap(target, cv2.COLORMAP_TURBO)
    return depth, target


def tensor_to_rgb(input):
    return np.transpose(np.squeeze(input.detach().cpu().numpy()), (1, 2, 0))


def tensor_to_depth(input):
    return np.squeeze(input.detach().cpu().numpy())


def merge_into_row(input, depth_target, depth_pred, error_map=None):
    if error_map is not None:
        error_map = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)
        img_merge = np.hstack([input, depth_target, depth_pred, error_map])
    else:
        img_merge = np.hstack([input, depth_target, depth_pred])

    return img_merge


# Inputs are np arrays
# Am I clipping prediction somewhere already? If not, do so here
def calculate_error_map(target, prediction):

    # Clamp to relatively small depth values
    target[target > 25] = 25
    prediction[prediction > 25] = 25
    error_map = np.abs(prediction - target)
    error_map = normalize_new_range(error_map)
    error_map = cv2.normalize(error_map, error_map, 0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U)
    error_map = cv2.cvtColor(error_map, cv2.COLOR_GRAY2BGR)
    error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_HOT)
    return error_map


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def write_results(img, results):
    font = cv2.FONT_HERSHEY_SIMPLEX

    blank = np.zeros(shape=(224, 224, 3), dtype=np.float32)
    out = cv2.hconcat([img, blank])

    rmse = "RMSE: {:.2f}m".format(results.rmse)
    mae = "MAE: {:.2f}m".format(results.mae)
    delta1 = "Delta1: {:.2f}m".format(results.delta1)
    cv2.putText(out, rmse, (out.shape[1] - blank.shape[1], 30),
                font, 1, (255, 255, 255), 1)
    cv2.putText(out, mae, (out.shape[1] - blank.shape[1], 60),
                font, 1, (255, 255, 255), 1)
    cv2.putText(out, delta1, (out.shape[1] - blank.shape[1], 90),
                font, 1, (255, 255, 255), 1)

    return out


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def load_config_file(file):
    if not os.path.isfile(file):
        raise ValueError("Parameters file does not exist")

    return json.load(open(file))


def format_dataset_path(dataset_paths):
    if isinstance(dataset_paths, str):
        dataset_paths = {
            dataset_paths
        }
    elif isinstance(dataset_paths, list):
        data_paths = set()
        for path in dataset_paths:
            data_paths.add(path)
        dataset_paths = data_paths

    return dataset_paths


def make_dir_with_date(root_dir, prefix):
    time = datetime.datetime.now()

    pid = None
    try:
        pid = os.environ["COMET_OPTIMIZER_PROCESS_ID"]
    except KeyError:
        pass

    date_dir = os.path.join(root_dir, prefix + "_" +
                            time.strftime("%m_%d_%H_%M"))
    if pid:
        date_dir += "_opt_{}".format(pid)

    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    return date_dir


def get_train_val_split_lengths(train_val_split, dataset_length):
    return [int(np.around(train_val_split[0] * 0.01 * dataset_length)),
            int(np.around(train_val_split[1] * 0.01 * dataset_length))]


def load_model(params, resume=None):
    # Load model checkpoint if specified
    model_state_dict,\
        optimizer_state_dict,\
        params["start_epoch"], _ = load_checkpoint(resume)
    model_state_dict = convert_state_dict_from_gpu(model_state_dict)

    # Load the model
    if params["encoder"] == "mobilenet":
        model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
    elif params["encoder"] == "resnet50":
        model = models.ResNetSkipAdd(layers=50, output_size=(224, 224), pretrained=True)
    elif params["encoder"] == "resnet18":
        model = models.ResNetSkipAdd(layers=18, output_size=(224, 224), pretrained=True)
    else:
        model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    return model, optimizer_state_dict


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
        best_loss = 100000  # Very high number

        if model_path:
            print("=> no checkpoint found at '{}'".format(model_path))

    return model_state_dict,\
        optimizer_state_dict,\
        start_epoch,\
        best_loss,\


def get_save_path(epoch, save_dir="./results"):
    save_path = os.path.join(
        save_dir, "model_{}.pth".format(str(epoch).zfill(4)))
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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_result": loss
    }, save_path)


def optimizer_to_gpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.device == "cpu":
                state[k] = v.cuda()


def convert_state_dict_from_gpu(state_dict):
    if state_dict:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if ("module" in k):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        return new_state_dict
    else:
        return state_dict


def save_losses_plot(path, num_epochs, losses, title):
    x = np.arange(1, num_epochs + 1, 1)
    plt.plot(x, losses)
    plt.xticks(x, x)
    plt.xlabel("Epochs")
    plt.ylabel("{} Loss (m)".format(title))
    plt.title("{} Loss".format(title))
    plt.savefig(path, bbox_inches="tight")


def log_comet_metrics(experiment, result, loss, prefix=None, step=None, epoch=None):
    metrics = {
        "loss": loss,
        "rmse": result.rmse,
        "irmse" : result.irmse,
        "mae": result.mae,
        "imae" : result.imae,
        "delta1": result.delta1,
        "delta2" : result.delta2,
        "delta3" : result.delta3,
        "absrel" : result.absrel,
        "lg10" : result.lg10
    }
    experiment.log_metrics(metrics, prefix=prefix, step=step, epoch=epoch, overwrite=True)


def log_image_to_comet(input, target, output, epoch, id, experiment, result, prefix, step=None):
    # Convert to np arrays
    input = tensor_to_rgb(input)
    target = tensor_to_depth(target)
    prediction = tensor_to_depth(output)

    # Log raw image
    stacked = stack_images(input, target, prediction)
    raw = create_raw_image(stacked)
    log_merged_raw_image_to_comet(raw, epoch, id, experiment, prefix, step)

    # Error map
    error_map = calculate_error_map(target.copy(), prediction.copy())

    # Make better depth visualization
    prediction, target = visualize_depth_compare(prediction, target)

    # Merge rgb, target, prediction, and error map
    img_merge = merge_into_row(input * 255, target, prediction, error_map)
    img_merge = write_results(img_merge, result)
    log_merged_image_to_comet(img_merge, epoch, id, experiment, prefix, step)

    return img_merge


def stack_images(input, target, output):
    b, g, r = cv2.split(input)
    return cv2.merge((b, g, r, output, target))


def create_raw_image(image):
    return image.tobytes()    


def log_merged_raw_image_to_comet(raw_image, epoch, id, experiment, prefix, step=None):
    file_name = "{}_epoch_{}_id_{}.raw".format(prefix, epoch, id)
    if step:
        step = int(step)

    experiment.log_asset_data(raw_image, name=file_name, step=step)


def log_merged_image_to_comet(img_merge, epoch, id, experiment, prefix, step=None):
    img_name = "{}_epoch_{}_id_{}".format(prefix, epoch, id)
    if step:
        step = int(step)
    experiment.log_image(img_merge, name=img_name, step=step, overwrite=True)


def flip_depth(outputs, targets, clip=None):
    targets = (1 / targets)
    if clip:
        outputs[outputs < clip] = clip
    outputs = (1 / outputs)
    return outputs, targets


def process_for_loss(outputs, targets, predict_disparity, loss_disparity, disparity_constant, clip=0.1):
    c = disparity_constant if loss_disparity else 1
    if predict_disparity != loss_disparity:
        outputs, targets = flip_depth(outputs, targets, clip)

    return outputs, targets, c


def convert_to_depth(outputs, targets, not_clipped_yet, is_disparity, clip=None):
    clip = clip if not_clipped_yet else None
    if is_disparity:
        outputs, targets = flip_depth(outputs, targets, clip)

    return outputs, targets
