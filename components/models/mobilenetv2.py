from torch import nn
from components.models.mobilenet import PointWiseConv2d, DepthWiseConv2d

class InvertedResidualBlock(nn.Module):
    """
    Inverted residual block from mobilenetv2.

    The lack of non-linearity between the first pointwise conv
    and the depthwise conv is due to losing representational
    power from the use of a non-linearity, particularly in
    higher dimensions.

    For strides>1 or cin!=cout, the input does not get added to the residual.
    """
    def __init__(self, cin, cout, stride=1, expansion_ratio=6, act=nn.ReLU6(True)):
        super().__init__()
        self.stride = stride
        self.same_c = cin == cout
        self.block = nn.Sequential(
            PointWiseConv2d(cin, cin*expansion_ratio),
            nn.BatchNorm2d(cin*expansion_ratio),
            act,
            DepthWiseConv2d(cin*expansion_ratio, st=stride),
            nn.BatchNorm2d(cin*expansion_ratio),
            act,
            PointWiseConv2d(cin*expansion_ratio, cout),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        residual = self.block(x)
        if self.stride != 1 or not self.same_c:
            return residual
        else:
            return x + residual


class ShortcutResidualBlock(nn.Module):
    """
    A generalisable residual block that uses a pointwise conv
    and pooling in the shortcut connection so that the input can
    be added to the residual.
    """
    def __init__(self, cin, cout, stride=1, expansion_ratio=6, act=nn.ReLU6(True)):
        super().__init__()
        self.block = nn.Sequential(
            PointWiseConv2d(cin, cin*expansion_ratio),
            DepthWiseConv2d(cin*expansion_ratio, st=stride),
            act,
            PointWiseConv2d(cin*expansion_ratio, cout),
            act
        )

        self.downsample = nn.AvgPool2d(2, stride=stride) if stride != 1 else None
        self.bottle = PointWiseConv2d(cin, cout) if cin != cout else None

    def forward(self, x):
        residual = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)

        if self.bottle is not None:
            x = self.bottle(x)
        return x + residual
