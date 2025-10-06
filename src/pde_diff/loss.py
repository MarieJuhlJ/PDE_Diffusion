import torch.nn as nn
from pde_diff.utils import LossRegistry

@LossRegistry.register("mse")
class MSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, model_output, target):
        return nn.functional.mse_loss(model_output, target)
