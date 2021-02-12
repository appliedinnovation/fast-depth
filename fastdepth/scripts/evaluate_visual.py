import argparse
import csv
import os
import sys
import time

import numpy as np
from comet_ml import ExistingExperiment, Experiment
import torch

sys.path.append(os.getcwd())

import models
import utils
from metrics import AverageMeter, Result

try:
    dataset_path = os.environ["DATASETS_ABS_PATH"]
except KeyError:
    print("Datasets.py absolute path not found in PATH")
sys.path.append(dataset_path)
import Datasets


def main(args):

    print("Loading config file: ", args.config)
    params = utils.load_config_file(args.config)
    params["test_dataset_paths"] = utils.format_dataset_path(
        params["test_dataset_paths"])
    if "nyu" not in params:
        params["nyu"] = False

    if args.existing_experiment:
        experiment = ExistingExperiment(
            api_key="jBFVYFo9VUsy0kb0lioKXfTmM",
            previous_experiment=args.existing_experiment)
    else:
        experiment = Experiment(api_key="jBFVYFo9VUsy0kb0lioKXfTmM",
                                project_name="fastdepth")

    # Data loading code
    print("Creating data loaders...")
    if params["nyu"]:
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(params["test_dataset_paths"], split='val')
    else:
        val_dataset = Datasets.FastDepthDataset(
            params["test_dataset_paths"],
            split='val',
            depth_min=params["depth_min"],
            depth_max=params["depth_max"],
            input_shape_model=(224, 224),
            random_crop=False)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=params["num_workers"],
                                             pin_memory=True)

    print("Loading model '{}'".format(args.model))
    if not params["nyu"]:
        model, _ = utils.load_model(params, args.model)
    else:
        # Maintain compatibility for fastdepth NYU model format
        state_dict = torch.load(args.model, map_location=params["device"])
        model = models.MobileNetSkipAdd(output_size=(224, 224),
                                        pretrained=True)
        model.load_state_dict(state_dict)
        params["start_epoch"] = 0

    # Set GPU
    params["device"] = torch.device(
        "cuda:{}".format(params["device"])
        if params["device"] >= 0 and torch.cuda.is_available() else "cpu")
    print("Using device", params["device"])

    model.to(params["device"])

    # Create output directory
    output_directory = os.path.join(os.path.dirname(args.model), "images")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    params["experiment_dir"] = output_directory
    print("Saving results to " + output_directory)

    evaluate(params, val_loader, model, experiment)


def evaluate(params, loader, model, experiment):
    print("Testing...")
    with experiment.test() and torch.no_grad():
        average = AverageMeter()
        end = time.time()
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(params["device"]), targets.to(
                params["device"])

            data_time = time.time() - end

            # Predict
            end = time.time()
            outputs = model(inputs)
            gpu_time = time.time() - end

            # Clip prediction
            outputs[outputs > params["depth_max"]] = params["depth_max"]
            outputs[outputs < params["depth_min"]] = params["depth_min"]

            result = Result()
            result.evaluate(outputs.data, targets.data)
            average.update(result, gpu_time, data_time, inputs.size(0))

            # Log images to comet
            img_merged = utils.log_image_to_comet(inputs[0],
                                                  targets[0],
                                                  outputs[0],
                                                  epoch=0,
                                                  id=i,
                                                  experiment=experiment,
                                                  result=result,
                                                  prefix="visual_test")
            if params["save_test_images"]:
                filename = os.path.join(
                    params["experiment_dir"],
                    "image_{}_epoch_{}.png".format(i,
                                                   str(params["start_epoch"])))
                utils.save_image(img_merged, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastDepth evaluation')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help="Model path.")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="config/evaluate.json",
                        help="Path to config JSON.")
    parser.add_argument('-e',
                        '--existing-experiment',
                        type=str,
                        help="Comet Existing Experiment key")
    args = parser.parse_args()
    main(args)
