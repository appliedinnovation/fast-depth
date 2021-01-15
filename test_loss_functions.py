import torch
import numpy as np
import loss

# Test predictin
prediction = torch.tensor([[2.5, -3], [1.3, -8]])

# Test target
target = torch.tensor([[2.3, -2.8], [2.0, -5.6]])

# Berhu loss
berhu_loss = loss.BerhuLoss()
berhu_loss_output = berhu_loss(prediction, target)
assert (torch.abs(berhu_loss_output - 1.848)) < 0.001

prediction = torch.tensor([[2.5, 3], [1.3, 8]])
target = torch.tensor([[2.3, 2.8], [2.0, 5.6]])

# SILog Loss
silog_loss = loss.SILogLoss()
silog_loss_output = silog_loss(prediction, target)
assert (torch.abs(silog_loss_output - 3.282)) < 0.001

# SIGradient Loss
prediction = torch.tensor([[[1.2, 4.3, 6.5, 3.2],
                            [3.2, 1.8, 3.4, 1.9],
                            [1.5, 1.9, 3.9, 0.4],
                            [1.4, 3.2, 6.5, 7.3]],
                           [[1.3, 2.0, 3.0, 2.0],
                            [0.4, 0.8, 0.4, 0.1],
                            [4.3, 5.6, 7.4, 2.6],
                            [2.4, 5.6, 4.7, 2.4]]])
target = torch.tensor([[[1.4, 4.8, 3.2, 0.9],
                        [2.8, 1.9, 1.2, 3.2],
                        [0.3, 0.8, 1.3, 1.8],
                        [4.5, 2.4, 7.8, 3.5]],
                       [[1.8, 2.1, 0.4, 4.9],
                        [0.7, 0.4, 1.2, 9.8],
                        [1.4, 3.4, 5.4, 6.5],
                        [0.2, 0.3, 0.5, 1.4]]])
                        
sigradient_loss = loss.SIGradientLoss()
sigradient_loss_output = sigradient_loss(prediction, target)
assert torch.abs(sigradient_loss_output - 1.727) < 0.001

# Normal loss
prediction = torch.unsqueeze(prediction, 1)
target = torch.unsqueeze(target, 1)

normal_loss = loss.NormalLoss()
normal_loss_output = normal_loss(prediction, target)
assert torch.abs(normal_loss_output - 0.835) < 0.001

# Global Focal Relative Loss
prediction = torch.rand(size=(4, 1, 224, 224))
target = torch.rand(size=(4, 1, 224, 224))

gfrl_loss = loss.GlobalFocalRelativeLoss()
gfrl_loss_output = gfrl_loss(prediction, target)
print(gfrl_loss_output)

print("All losses passed")