from torch import nn
from components.conv import conv_bn_act, conv_bn

class BasicResBlock(nn.Module):
    def __init__(self, cin, cout, expansion=1, st=1, ks=3, act=nn.ReLU(True)):
        super().__init__()
        self.st = st
        self.act = act

        self.conv1 = conv_bn_act(cin, cin, ks, st, act=act)
        self.conv2 = conv_bn(cin, cout*expansion)

        self.shortcut = None if cout*expansion==cin else conv_bn(cin, cout*expansion, ks=1, st=st)

    def forward(self, x):
        inp = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.shortcut is not None:
            inp = self.shortcut(inp)

        return self.act(x + inp)


class BottleneckResBlock(nn.Module):
    def __init__(self, cin, cout, expansion=4, st=1, ks=3, act=nn.ReLU(True)):
        super().__init__()
        self.st = st
        self.act = act

        self.conv1 = conv_bn_act(cin, cout, 1, 1, act=act)
        self.conv2 = conv_bn_act(cout, cout, ks, st, act=act)
        self.conv3 = conv_bn(cout, cout*expansion, ks=1)

        self.shortcut = None if cout*expansion==cin else conv_bn(cin, cout*expansion, ks=1, st=st)
        self.extras = extras

    def forward(self, x):
        inp = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.shortcut is not None:
            inp = self.shortcut(inp)

        return self.act(x + inp)
