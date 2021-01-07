import torch
import torch.nn as nn
import numpy as np
import utils

def get_loss(loss_str):
    loss_dict = {
        "l1": torch.nn.L1Loss(),
        "l2": torch.nn.MSELoss(),
        "silog": SILogLoss(),
        "berhu": BerhuLoss()
    }

    return loss_dict[loss_str]

class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]

        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class BerhuLoss(nn.Module):
    def __init__(self):
        super(BerhuLoss, self).__init__()
        self.name = 'Berhu'

    def forward(self, input, target, mask=None):
        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        # C threshold
        c = 0.2 * torch.max(torch.abs(target - input))

        sum_loss = 0.0
        for pred, depth in zip(input, target):
            for pred_row, depth_row in zip(pred[0], depth[0]):
                for pred_pixel, depth_pixel in zip(pred_row, depth_row):
                    x = torch.abs(pred_pixel - depth_pixel)
                    if x <= c:
                        sum_loss += x
                    else:
                        sum_loss += (1 / (2 * c)) * (torch.square(x) + torch.square(c))
        
        # Calculate the mean
        loss = sum_loss / input.shape[0]

        return loss
