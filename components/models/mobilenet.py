from torch import nn
from components import conv

class DepthWiseConv2d(nn.Conv2d):
    "Depth-wise convolution operation"
    def __init__(self, channels, ks=3, st=1):
        super().__init__(channels, channels, ks, stride=st, padding=ks//2, groups=channels)

class PointWiseConv2d(nn.Conv2d):
    "Point-wise (1x1) convolution operation."
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 1, stride=1)

class MobileNetConv(nn.Module):
    "The MobileNetV1 convolutional layer"
    def __init__(self, cin, cout, ks=3, st=1, act=nn.ReLU(True)):
        super().__init__()
        self.dwconv = DepthWiseConv2d(cin, ks=ks, st=st)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act = act
        self.pwconv = PointWiseConv2d(cin, cout)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.bn1(self.act(self.dwconv(x)))
        x = self.bn2(self.act(self.pwconv(x)))
        return x
