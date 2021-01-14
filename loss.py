import torch
import torch.nn as nn
import numpy as np
import utils


def get_loss(loss_str):
    loss_dict = {
        "l1": torch.nn.L1Loss(),
        "l2": torch.nn.MSELoss(),
        "silog": SILogLoss(),
        "berhu": BerhuLoss(),
        "sigradient": SIGradientLoss()
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


# Read about autograd and make sure gradients can be correctly
# calculated
class SIGradientLoss(nn.Module):
    def __init__(self):
        super(SIGradientLoss, self).__init__()
        self.name = "SIGradient"

    # Shift input along the specified axis
    # Set to zero the values that were rolled over to the end
    def _roll(self, input, shift, axis):
        input_shifted = np.roll(input, shift=shift, axis=axis)
        self._set_end_zero(input_shifted, shift, axis)
        return input_shifted

    def _normalized_gradient(self, input, step):
        return (step - input) / torch.abs(step + input)

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

    def forward(self, input, target, mask=None):
        assert input.shape == target.shape

        if mask is not None:
            input = input[mask]
            target = target[mask]

        scales = [1, 2, 4, 8, 16]

        sum_loss = torch.zeros(size=(1,), requires_grad=True)
        for s in scales:

            # Calculate gradients
            gx_input, gy_input = self._calculate_g(input, shift=s)
            gx_target, gy_target = self._calculate_g(target, shift=s)

            g_input = np.array([gx_input.numpy(), gy_input.numpy()], copy=False)
            g_target = np.array([gx_target.numpy(), gy_target.numpy()], copy=False)

            sum_loss += torch.norm(torch.as_tensor(g_input - g_target))

        return torch.div(sum_loss, len(scales))


# class NormalLoss(nn.Module):
    # def __init__(self):
        # super(NormalLoss, self).__init__()
        # self.name = "Normal"
#
    # def forward(self, input, target, mask=None):
        # if mask is not None:
        # input = input[mask]
        # target = target[mask]
#
        # for pred, depth in zip(input, target):
        # for pred_row, depth_row in zip(pred[0], depth[0]):
        # for pred_pixel, depth_pixel in zip(pred_row, depth_row):
        # normal_pred =
