"""
MobileNetV3 differs from previous versions by:
- Using Squeeze and excitation blocks inside the inverted
  residual blocks.
- Uses the swish activation function - x*sigmoid(x)
  but to reduce computation uses a "hard" version
  by scaling relu6 - x*(relu6(x+3)/6)
- Uses a reduction ratio of 4 in the SE blocks.
"""

import torch
from torch import nn
import torch.nn.functional as F

from components.models.senet import SqueezeExcite
from components.models.mobilenet import PointWiseConv2d, DepthWiseConv2d
from components.core import SequentialEx


class Swish(nn.Module):
    def forward(self, x):
        return x*torch.sigmoid(x)


class HardSwish(nn.Module):
    def forward(self, x):
        return x*(F.relu6(x+3)/6)


class InvertedResidualV3(nn.Module):
    """
    MobileNetV3 inverted residual block with squeeze excite block embedded into
    residual layer, after the depthwise conv. Uses the HardSwish activation.
    """
    def __init__(self, cin, cout, ks=3, stride=1, expansion_ratio=4, squeeze_reduction_ratio=None, act=HardSwish()):
        super().__init__()
        self.stride = stride
        self.same_c = cin == cout
        self.exp = int(cin*expansion_ratio)
        self.block = SequentialEx(
            PointWiseConv2d(cin, self.exp),
            nn.BatchNorm2d(self.exp),
            act,
            DepthWiseConv2d(self.exp, st=stride),
            nn.BatchNorm2d(self.exp),
            act,
            SqueezeExcite(self.exp, reduction_ratio=squeeze_reduction_ratio) if squeeze_reduction_ratio is not None else None,
            PointWiseConv2d(self.exp, cout),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        residual = self.block(x)
        if self.stride != 1 or not self.same_c:
            return residual
        else:
            return x + residual
