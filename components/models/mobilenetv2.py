from torch import nn
from components.models.mobilenet import PointWiseConv2d, DepthWiseConv2d

class InvertedResidualBlock(nn.Module):
    """
    Inverted residual block from mobilenetv2.

    The lack of non-linearity between the first pointwise conv
    and the depthwise conv is due to losing representational
    power from the use of a non-linearity, particularly in
    higher dimensions.
    """
    def __init__(self, cin, cout, expansion_ratio=4, act=nn.ReLU6(True), stride=1):
        super().__init__()
        self.block = nn.Sequential(
            PointWiseConv2d(cin, cin*expansion_ratio),
            act, # Does this go here??
            DepthWiseConv2d(cin*expansion_ratio, st=stride),
            act,
            PointWiseConv2d(cin*expansion_ratio, cin),
        )

        if stride != 1:
            self.downsample = nn.AvgPool2d(2, stride=stride)

    def forward(self, x):
        residual = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual


class InvertedResidualBlockWithStride(nn.Module):
    """
    Inverted residual block with stride doesn't
    use a residual connection from mobilenetv2.
    """
    def __init__(self, cin, cout, expansion_ratio=4, act=nn.ReLU6(True), stride=2):
        super().__init__()
        self.block = nn.Sequential(
            PointWiseConv2d(cin, cin*expansion_ratio),
            DepthWiseConv2d(cin*expansion_ratio, st=stride),
            act,
            PointWiseConv2d(cin*expansion_ratio, cin),
            act
        )

    def forward(self, x):
        return self.block(x)
