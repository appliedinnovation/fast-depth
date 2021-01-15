import os
import sys
import json
import argparse
import numpy as np
from comet_ml import Experiment, ExistingExperiment
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.hub
import matplotlib.pyplot as plt

# FastDepth imports
from dataloaders.nyu import NYUDataset
import evaluate
import loss
from metrics import AverageMeter, Result
import models
import optimize
import utils

# Import custom Dataset
try:
    dataset_path = os.environ["DATASETS_ABS_PATH"]
except KeyError:
    print("Datasets.py absolute path not found in PATH")
sys.path.append(dataset_path)
import Datasets


def get_params(file):
    params = utils.load_config_file(file)

    # Convert from JSON format to DataLoader format
    params["training_dataset_paths"] = utils.format_dataset_path(
        params["training_dataset_paths"])
    params["test_dataset_paths"] = utils.format_dataset_path(
        params["test_dataset_paths"])
    return params


def set_up_experiment(params, experiment, resume=None):

    # Log hyper params to Comet
    hyper_params = {
        "learning_rate": params["optimizer"]["lr"],
        "weight_decay": params["optimizer"]["weight_decay"],
        "optimizer": params["optimizer"]["type"],
        "encoder" : params["encoder"],
        "loss": params["loss"],
        "num_epochs": params["num_epochs"],
        "batch_size": params["batch_size"],
        "train_val_split": params["train_val_split"][0],
        "depth_max": params["depth_max"],
        "depth_min": params["depth_min"],
        "disparity": params["predict_disparity"],
        "disparity_constant": params["disparity_constant"],
        "lr_epoch_step_size": params["lr_epoch_step_size"]
    }
    experiment.log_parameters(hyper_params)
    experiment.add_tag(params["loss"])

    # Create experiment directory
    if resume:
        experiment_dir = os.path.split(resume)[0]  # Use existing folder
    else:
        experiment_dir = utils.make_dir_with_date(
            params["save_dir"], "fastdepth")  # New folder
    print("Saving results to ", experiment_dir)
    params["experiment_dir"] = experiment_dir
    experiment.log_other("saved_model_directory", experiment_dir)

    # Log dataset info to Comet
    training_folders = ", ".join(params["training_dataset_paths"])
    test_folders = ", ".join(params["test_dataset_paths"])
    experiment.log_dataset_info(path=training_folders)
    experiment.log_other("test_dataset_info", test_folders)

    ## Dataset ##
    train_loader, val_loader, test_loader = load_dataset(params)

    # Configure GPU
    params["device"] = torch.device("cuda:{}".format(params["device"]) if type(
        params["device"]) is int and torch.cuda.is_available() else "cpu")

    ## Model ##
    model, optimizer_state_dict = utils.load_model(params, resume)

    # Use parallel GPUs if available
    # Specify which GPUs to use on DGX
    try:
        if not os.environ["CUDA_VISIBLE_DEVICES"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if os.environ["USE_MULTIPLE_GPUS"] == "TRUE" and torch.cuda.device_count() > 1:
            print("Let's use", num_gpus, "GPUs!")
            model = nn.DataParallel(model)
    except KeyError:
        pass

    # Send model to GPU(s)
    # This must be done before optimizer is created if a model state_dict is being loaded
    model.to(params["device"])

    print("Encoder: ", params["encoder"])

    ## Loss ##
    criterion = loss.get_loss(params["loss"])
    print("Loss: ", params["loss"])

    ## Optimizer ##
    optimizer = optimize.get_optimizer(model, params)
    print("Optimizer: " , params["optimizer"]["type"])

    experiment.add_tag(params["optimizer"]["type"])
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    # Load optimizer tensors onto GPU if necessary
    utils.optimizer_to_gpu(optimizer)

    ##  LR Scheduler ##
    if resume:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params["lr_epoch_step_size"], gamma=0.1, last_epoch=params["start_epoch"])
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params["lr_epoch_step_size"], gamma=0.1)

    return params, train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler


def load_dataset(params):
    print("Loading the dataset...")

    if params['nyu_dataset']:
        dataset = NYUDataset("../data/nyudepthv2/train", split='train')
        test_dataset = NYUDataset("../data/nyudepthv2/val", split='val')
    else:
        dataset = Datasets.FastDepthDataset(params["training_dataset_paths"],
                                            split='train',
                                            depth_min=params["depth_min"],
                                            depth_max=params["depth_max"],
                                            input_shape_model=(224, 224),
                                            disparity=params["predict_disparity"],
                                            random_crop=params["random_crop"])

        test_dataset = Datasets.FastDepthDataset(params["test_dataset_paths"],
                                                 split='val',
                                                 depth_min=params["depth_min"],
                                                 depth_max=params["depth_max"],
                                                 input_shape_model=(224, 224),
                                                 disparity=params["predict_disparity"],
                                                 random_crop=False)

    # Make training/validation split
    train_val_split_lengths = utils.get_train_val_split_lengths(
        params["train_val_split"], len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, train_val_split_lengths)
    print("Train/val split: ", train_val_split_lengths)
    params["num_training_examples"] = len(train_dataset)
    params["num_validation_examples"] = len(val_dataset)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params["batch_size"],
                                               shuffle=True,
                                               num_workers=params["num_workers"],
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params["batch_size"],
                                             shuffle=True,
                                             num_workers=params["num_workers"],
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=params["batch_size"],
                                              shuffle=False,
                                              num_workers=params["num_workers"],
                                              pin_memory=True)

    return train_loader, val_loader, test_loader


def train(params, train_loader, val_loader, model, criterion, optimizer, scheduler, experiment):

    mean_val_loss = -1
    try:
        train_step = int(np.ceil(
            params["num_training_examples"] / params["batch_size"]) * params["start_epoch"])
        val_step = int(np.ceil(params["num_validation_examples"] /
                           params["batch_size"] * params["start_epoch"]))
        
        for epoch in range(params["num_epochs"] - params["start_epoch"]):
            current_epoch = params["start_epoch"] + epoch + 1

            epoch_loss = 0.0
            running_loss = 0.0
            average = AverageMeter()
            img_idxs = np.random.randint(0, len(train_loader), size=5)

            model.train()
            with experiment.train():
                for i, (inputs, target) in enumerate(train_loader):

                    # Send data to GPU
                    inputs, target = inputs.to(
                        params["device"]), target.to(params["device"])

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Predict
                    prediction = model(inputs)

                    loss = criterion(prediction, target)

                    # Compute loss
                    loss.backward()
                    optimizer.step()

                    # Calculate metrics
                    result = Result()
                    result.evaluate(prediction.data, target.data)
                    average.update(result, 0, 0, inputs.size(0))
                    epoch_loss += loss.item()

                    # Log to Comet
                    utils.log_comet_metrics(
                        experiment, result, loss.item(), step=train_step, epoch=current_epoch)
                    train_step += 1

                    # Log images to Comet
                    if i in img_idxs:
                        utils.log_image_to_comet(
                            inputs[0], target[0], prediction[0], current_epoch, i, experiment, result, "train", train_step)
                        utils.log_raw_image_to_comet(
                            inputs[0], target[0], prediction[0], current_epoch, i, experiment, "train", train_step)

                    # Print statistics
                    running_loss += loss.item()
                    if (i + 1) % params["stats_frequency"] == 0 and i != 0:
                        print('[%d, %5d] loss: %.3f' %
                              (current_epoch, i + 1, running_loss / params["stats_frequency"]))
                        running_loss = 0.0

                # Log epoch metrics to Comet
                mean_train_loss = epoch_loss/len(train_loader)
                utils.log_comet_metrics(experiment, average.average(), mean_train_loss,
                                        prefix="epoch", step=train_step, epoch=current_epoch)

            # Validation each epoch
            epoch_loss = 0.0
            average = AverageMeter()
            with experiment.validate():
                with torch.no_grad():
                    img_idxs = np.random.randint(0, len(val_loader), size=5)
                    model.eval()
                    for i, (inputs, target) in enumerate(val_loader):
                        inputs, target = inputs.to(
                            params["device"]), target.to(params["device"])

                        # Predict
                        prediction = model(inputs)

                        loss = criterion(prediction, target)

                        # Calculate metrics
                        result = Result()
                        result.evaluate(prediction.data, target.data)
                        average.update(result, 0, 0, inputs.size(0))
                        epoch_loss += loss.item()

                        # Log to Comet
                        utils.log_comet_metrics(
                            experiment, result, loss.item(), step=val_step, epoch=current_epoch)
                        val_step += 1

                        # Log images to Comet
                        if i in img_idxs:
                            utils.log_image_to_comet(
                                inputs[0], target[0], prediction[0], current_epoch, i, experiment, result, "val", val_step)
                            utils.log_raw_image_to_comet(
                            inputs[0], target[0], prediction[0], current_epoch, i, experiment, "val", train_step)

                    # Log epoch metrics to Comet
                    mean_val_loss = epoch_loss / len(val_loader)
                    utils.log_comet_metrics(experiment, average.average(), mean_val_loss,
                                            prefix="epoch", step=val_step, epoch=current_epoch)
                    print("Validation Loss [%d]: %.3f" %
                          (current_epoch, mean_val_loss))

            # Save periodically
            if (current_epoch + 1) % params["save_frequency"] == 0:
                save_path = utils.get_save_path(
                    current_epoch, params["experiment_dir"])
                utils.save_model(model, optimizer, save_path, current_epoch,
                                 mean_val_loss, params["max_checkpoints"])
                experiment.log_model(save_path.split("/")[-1], save_path)
                print("Saving new checkpoint")

            experiment.log_epoch_end(current_epoch)
            scheduler.step()

        print("Finished training")

        # Save the final model
        save_path = utils.get_save_path(
            params["num_epochs"], params["experiment_dir"])
        utils.save_model(model, optimizer, save_path, current_epoch,
                         mean_val_loss, params["max_checkpoints"])
        experiment.log_model(save_path.split("/")[-1], save_path)
        print("Model saved to ", os.path.abspath(save_path))

    except KeyboardInterrupt:
        print("Saving model and quitting...")
        save_path = utils.get_save_path(
            current_epoch, params["experiment_dir"])
        utils.save_model(model, optimizer, save_path, current_epoch,
                         mean_val_loss, params["max_checkpoints"])
        experiment.log_model(save_path.split("/")[-1], save_path)
        print("Model saved to ", os.path.abspath(save_path))


def main(args):
    os.environ["USE_MULTIPLE_GPUS"] = "TRUE"

    # Create Comet ML Experiment
    if args.resume:
        experiment_key = input("Enter Comet ML key of experiment to resume:")
        experiment = ExistingExperiment(
            api_key="jBFVYFo9VUsy0kb0lioKXfTmM", previous_experiment=experiment_key)
    elif args.no_comet:
        experiment = Experiment(
            api_key="jBFVYFo9VUsy0kb0lioKXfTmM", project_name="test-runs")
    else:
        experiment = Experiment(
            api_key="jBFVYFo9VUsy0kb0lioKXfTmM", project_name="fastdepth")

    if args.tag:
        experiment.add_tag(args.tag)
    if args.name:
        experiment.set_name(args.name)

    config_file = args.config
    params = get_params(config_file)
    params["nyu_dataset"] = args.nyu

    params, train_loader, val_loader, test_loader, \
        model, criterion, optimizer, scheduler = set_up_experiment(
            params, experiment, args.resume)

    train(params, train_loader, val_loader,
          model, criterion, optimizer, scheduler, experiment)

    evaluate.evaluate(params, test_loader, model, experiment)


if __name__ == "__main__":
    # Parse command args
    parser = argparse.ArgumentParser(description='FastDepth Training')
    parser.add_argument('--config', '-c', type=str, default=None, required=True,
                        help="Filename of parameters configuration JSON.")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to model checkpoint to resume training.")
    parser.add_argument('-t', '--tag', type=str, default=None,
                        help='Extra tag to add to Comet experiment')
    parser.add_argument('-n', '--name', type=str, default=None,
                        help='Comet ML Experiment name')
    parser.add_argument('--nyu', type=int, default=0,
                        help='whether to use NYU Depth V2 dataset.')
    parser.add_argument('--no-comet', action='store_true')
    args = parser.parse_args()
    main(args)
