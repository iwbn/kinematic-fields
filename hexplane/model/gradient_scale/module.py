from .functional import scalegrad
import torch
from torch import nn

class GradientScale(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return scalegrad(x, self.alpha)