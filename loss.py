import torch
import torch.nn as nn
import numpy as np
import utils

def get_loss(loss_str):
    loss_dict = {
        "l1": torch.nn.L1Loss(),
        "l2": torch.nn.MSELoss(),
        "silog": SILogLoss()
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
