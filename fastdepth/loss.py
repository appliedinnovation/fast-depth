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
        "normal": NormalLoss(),
        "gfrl": GlobalFocalRelativeLoss()
    }

    return loss_dict[loss_str]


class Loss(nn.Module):
    def __init__(self, loss_dict):
        super(Loss, self).__init__()
        self.name = 'Loss'

        self.loss = {}
        try:
            self.loss["phase_1"] = [get_loss(fn)
                                    for fn in loss_dict["phase_1"]["losses"]]
            self.loss["phase_2"] = [get_loss(fn)
                                    for fn in loss_dict["phase_2"]["losses"]]
            self.loss["phase_3"] = [get_loss(fn)
                                    for fn in loss_dict["phase_3"]["losses"]]
        except KeyError:
            pass

        if not self.loss and not self.loss["phase_1"]:
            raise ValueError(
                "No loss function specified. There must be at least one"
                "loss function specified in phase_1.")

        self.constants = {
            "k1": 1.0,
            "k2": 1.0,
            "k3": 1.0
        }
        try:
            self.constants["k1"] = loss_dict["phase_1"]["k"]
            self.constants["k2"] = loss_dict["phase_2"]["k"]
            self.constants["k3"] = loss_dict["phase_3"]["k"]
        except KeyError:
            pass

    def forward(self, input, target, mask=None, phase_list=["phase_1"]):

        # Phase 1
        loss = sum([criterion(input, target, mask)
                    for criterion in self.loss["phase_1"]]) * self.constants["k1"]

        # Phase 2
        if "phase_2" in phase_list and "phase_2" in self.loss:
            loss += sum([criterion(input, target, mask)
                         for criterion in self.loss["phase_2"]]) * self.constants["k2"]

        # Phase 3
        if "phase_3" in phase_list and "phase_3" in self.loss:
            loss += sum([criterion(input, target, mask)
                         for criterion in self.loss["phase_3"]]) * self.constants["k3"]

        return loss


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

    def forward(self, input, target, mask=None):
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
        n_input = torch.cat(
            (-grad_input.view(-1), torch.ones([1, ], dtype=torch.float32, device=grad_input.device)))
        n_target = torch.cat(
            (-grad_target.view(-1), torch.ones([1, ], dtype=torch.float32, device=grad_target.device)))

        # Inner product of prediction and target
        numerator = torch.dot(n_input, n_target)

        # Normalize by vector magnitudes
        d1 = torch.sqrt(torch.dot(n_input, n_input))
        d2 = torch.sqrt(torch.dot(n_target, n_target))
        denominator = torch.mul(d1, d2)

        losses = 1 - numerator / denominator
        return torch.mean(losses)


class GlobalFocalRelativeLoss(nn.Module):
    def __init__(self):
        super(GlobalFocalRelativeLoss, self).__init__()
        self.name = "GFRL"

    def _create_pixel_pairs(self, input):
        batch_size = input.shape[0]

        # Divide image into nxn blocks
        n = 16
        unfold = torch.nn.Unfold(kernel_size=(n, n), stride=n)
        input_blocks = unfold(input)  # [B, 256, 196]
        num_patches = input_blocks.shape[-1]

        flattened_blocks = input_blocks.view(-1)
        total_patches = batch_size * input_blocks.shape[-1]

        # Randomly sample 1 pixel from each block
        random_pixel_idxs = torch.Tensor(
            [256 * i + torch.randint(0, 256, size=(1,)) for i in range(total_patches)])
        pixel_samples = flattened_blocks[random_pixel_idxs.type(torch.long)]
        pixel_samples_batched = pixel_samples.view(batch_size, num_patches)

        # Create combinations of each pixel pair index
        pixel_pairs = torch.stack(
            ([torch.combinations(pixel_samples_batched[i]) for i in range(batch_size)]))
        pixel_pairs = pixel_pairs.view(-1, 2)

        return pixel_pairs[..., 0], pixel_pairs[..., 1]

    def _masked_split(self, input, mask):
        mask_selection = torch.masked_select(input, mask)
        mask_selection_opposite = torch.masked_select(
            input, torch.logical_not(mask))
        return mask_selection, mask_selection_opposite

    def forward(self, input, target, mask=None):
        assert input.shape == target.shape

        if mask is not None:
            input = input[mask]
            target = target[mask]

        # Create pairs of random pixels for comparing ordinal relations
        d1_input, d2_input = self._create_pixel_pairs(input)
        d1_target, d2_target = self._create_pixel_pairs(target)

        # Depth equality mask
        # Depth values are equal if depth difference ratio is less than 0.02
        equal_mask = torch.abs(torch.sub(d1_target, d2_target)) < 0.02

        # Split up into equal and nonequal pixel pairs
        d1_input_equal, d1_input_nonequal = self._masked_split(
            d1_input, equal_mask)
        d2_input_equal, d2_input_nonequal = self._masked_split(
            d2_input, equal_mask)
        d1_target_equal, d1_target_nonequal = self._masked_split(
            d1_target, equal_mask)
        d2_target_equal, d2_target_nonequal = self._masked_split(
            d2_target, equal_mask)

        # Create ordinal masks with target ordinal relations
        lesser_mask = torch.lt(d1_target_nonequal, d2_target_nonequal)
        greater_mask = torch.gt(d1_target_nonequal, d2_target_nonequal)

        # rk = -1 if d1 < d2, rk = 1 if d1 > d2
        ordinal_mask = lesser_mask.type(
            torch.int8) * -1 + greater_mask.type(torch.int8) * 1

        # Calculate Wk modulating factor
        ord_factor = 1 + \
            torch.exp(torch.mul(-ordinal_mask,
                                d1_input_nonequal - d2_input_nonequal))
        wk = 1 - 1 / ord_factor

        # Ordinal loss, rk != 0
        gam = 2  # Modulating exponent
        ordinal_loss = torch.mul(torch.pow(wk, gam), torch.log(ord_factor))

        # MSE Loss, rk == 0
        mse_loss = torch.square(d1_input_equal - d2_input_equal)

        # Sum the average losses
        loss = torch.mean(ordinal_loss) + torch.mean(mse_loss)

        return loss
