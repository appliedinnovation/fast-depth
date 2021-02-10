import torch.optim as optim


def get_optimizer(model, params):
    if params["optimizer"]["type"] == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=params["optimizer"]["lr"],
                              momentum=params["optimizer"]["momentum"],
                              weight_decay=params["optimizer"]["weight_decay"])
    elif params["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=params["optimizer"]["lr"])
    elif params["optimizer"]["type"] == "adamw":
        optimizer = optim.Adam(model.parameters(),
                               lr=params["optimizer"]["lr"],
                               weight_decay=params["optimizer"]["weight_decay"])
    
    return optimizer
