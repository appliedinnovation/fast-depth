import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.getcwd())

import models
import utils

try:
    dataset_path = os.environ["DATASETS_ABS_PATH"]
    sys.path.append(dataset_path)
except KeyError:
    print("Datasets.py absolute path not found in PATH")
import Datasets


def main(args):

    print("Loading config file: ", args.config)
    params = utils.load_config_file(args.config)
    params["dataset_paths"] = utils.format_dataset_path(
        params["dataset_paths"])
    if "nyu" not in params:
        params["nyu"] = False

    # Data loading code
    print("Creating data loaders...")
    if params["nyu"]:
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(params["dataset_paths"], split='val')
    else:
        val_dataset = Datasets.FastDepthDataset(params["dataset_paths"],
                                                split='val',
                                                depth_min=params["depth_min"],
                                                depth_max=params["depth_max"],
                                                input_shape_model=(224, 224),
                                                random_crop=False)

    # set batch size to be 1 for validation
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True)

    # Set GPU
    params["device"] = torch.device(
        "cuda:{}".format(params["device"])
        if params["device"] >= 0 and torch.cuda.is_available() else "cpu")
    print("Using device", params["device"])

    print("Loading models...")
    models = []
    model_names = []
    for model_dict in params["models"]:
        model_names.append(Path(model_dict["model_path"]).stem)
        model, _ = utils.load_model(model_dict, model_dict["model_path"], params["device"])
        model.to(params["device"])
        models.append(model)

    # Create output directory
    output_directory = os.path.join(params["save_folder"], ".".join(model_names))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    params["output_directory"] = output_directory
    print("Saving results to " + output_directory)

    compare_models(params, data_loader, models)


def compare_models(params, loader, models):
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input, target = input.to(params["device"]), target.to(
                params["device"])

            # Inference
            predictions = []
            for model in models:
                prediction = model(input)

                # Clip prediction
                prediction[
                    prediction > params["depth_max"]] = params["depth_max"]
                prediction[
                    prediction < params["depth_min"]] = params["depth_min"]

                predictions.append(prediction)

            # Convert tensors to np arrays
            rgb = utils.tensor_to_rgb(input)
            prediction_images = []
            for prediction in predictions:
                prediction_images.append(utils.tensor_to_depth(prediction))
            gt = utils.tensor_to_depth(target)
            colored_predictions, gt = utils.visualize_depth_compare(prediction_images, gt)
            
            # Combine predictions and ground truth into one image
            combined = np.hstack([rgb * 255, *colored_predictions, gt])

            # Save combined image
            filename = os.path.join(params["output_directory"],
                                    "image_{}.png".format(i))
            utils.save_image(combined, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare two Fastdepth models.')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="config/evaluate.json",
                        help="Path to config JSON.")
    args = parser.parse_args()
    main(args)
