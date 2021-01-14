import torch
import torch.nn as nn
import numpy as np
import kornia.filters as KF
import utils


def get_loss(loss_str):
    loss_dict = {
        "l1": torch.nn.L1Loss(),
        "l2": torch.nn.MSELoss(),
        "silog": SILogLoss(),
        "berhu": BerhuLoss(),
        "sigradient": SIGradientLoss(),
        "normal" : NormalLoss()
    }

    return loss_dict[loss_str]


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=False):
        assert input.shape == target.shape

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
        assert input.shape == target.shape

        if mask is not None:
            input = input[mask]
            target = target[mask]

        # Absolute difference
        diff = torch.abs(input - target)

        # Threshold term
        c = 0.2 * torch.max(diff)

        # Berhu term
        diff_square = (torch.square(diff) + torch.square(c)) / (2 * c)

        diff_square[diff <= c] = 0
        diff_copy = diff.clone()
        diff_copy[diff_copy > c] = 0
        diff_copy += diff_square
        
        loss = torch.mean(diff_copy)
        return loss


class SIGradientLoss(nn.Module):
    def __init__(self):
        super(SIGradientLoss, self).__init__()
        self.name = "SIGradient"

    # Shift input along the specified axis
    # Set to zero the values that were rolled over to the end
    def _roll(self, input, shift, axis):
        input_shifted = torch.roll(input, shifts=shift, dims=axis)
        self._set_end_zero(input_shifted, shift, axis)
        return input_shifted

    def _normalized_gradient(self, input, step, eps=1e-8):
        return (step - input) / torch.abs(step + input + eps)

    def _set_end_zero(self, input, idx_from_end, axis):
        if axis == -1:
            input[..., -idx_from_end:] = 0
        else:
            input[..., -idx_from_end:, :] = 0

    # Calculate the gradient along the rows and columns
    def _calculate_g(self, input, shift):
        input_shifted_rows = torch.as_tensor(self._roll(input, shift, axis=-2))
        input_shifted_cols = torch.as_tensor(self._roll(input, shift, axis=-1))

        gx = self._normalized_gradient(input, input_shifted_rows)
        gy = self._normalized_gradient(input, input_shifted_cols)
        self._set_end_zero(gx, shift, -2)
        self._set_end_zero(gy, shift, -1)
        return gx, gy
    
    def  forward(self, input, target, mask=None):
        assert input.shape == target.shape

        if mask is not None:
            input = input[mask]
            target = target[mask]

        scales = [1, 2, 4, 8, 16]

        losses = torch.empty(size=(len(scales),))
        for i, s in enumerate(scales):

            # Calculate gradients
            gx_input, gy_input = self._calculate_g(input, shift=s)
            gx_target, gy_target = self._calculate_g(target, shift=s)

            # Concatenate gradients
            g_input = torch.cat((gx_input, gy_input), 0)
            g_target = torch.cat((gx_target, gy_target), 0)

            losses[i] = torch.norm(g_input - g_target)

        return torch.mean(losses)


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.name = "Normal"

    def forward(self, input, target, mask=None):
        assert input.shape == target.shape

        if mask is not None:
            input = input[mask]
            target = target[mask]

        # Calculate spatial depth gradient
        grad_input = KF.spatial_gradient(input, mode='sobel')
        grad_target = KF.spatial_gradient(target, mode='sobel')

        # Create homogeneous column vectors
        n_input = torch.cat((-grad_input.view(-1), torch.ones([1,], dtype=torch.float32, device=grad_input.device)))
        n_target = torch.cat((-grad_target.view(-1), torch.ones([1,], dtype=torch.float32, device=grad_target.device)))

        # Inner product of prediction and target
        numerator = torch.dot(n_input, n_target)

        # Normalize by vector magnitudes
        d1 = torch.sqrt(torch.dot(n_input, n_input))
        d2 = torch.sqrt(torch.dot(n_target, n_target))
        denominator = torch.mul(d1, d2)

        losses = 1 - numerator / denominator
        return torch.mean(losses) 
        
