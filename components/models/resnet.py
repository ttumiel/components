from torch import nn

from components.conv import conv_bn_act, conv_bn, init_cnn_
from components.models import ModelBuilder


class ResBlock(nn.Module):
    def __init__(self, cin, cout, expansion=1, st=1, ks=3, act=nn.ReLU(True), extras=None):
        super().__init__()
        self.st = st
        self.act = act

        if expansion == 1:
            self.conv = nn.Sequential(
                conv_bn_act(cin, cout, ks, st, act=act),
                conv_bn(cout, cout*expansion)
            )
        else:
            self.conv = nn.Sequential(
                conv_bn_act(cin, cout, 1, 1, act=act),
                conv_bn_act(cout, cout, ks, st, act=act),
                conv_bn(cout, cout*expansion, ks=1)
            )

        self.shortcut = None if cout*expansion==cin else conv_bn(cin, cout*expansion, ks=1, st=st)
        self.extras = extras

    def forward(self, x):
        inp = x

        x = self.conv(x)

        if self.extras is not None:
            x = self.extras(x)

        if self.shortcut is not None:
            inp = self.shortcut(inp)

        return self.act(x + inp)


resnet_layers = {
    18: [1,1,1,1],
    34: [2,3,5,2],
    50: [2,3,5,2],
    101: [2,3,22,2],
    152: [2,7,35,2],
}

def n(param, n=1):
    return [param]*n

class ResNet(ModelBuilder):
    def __init__(self, layers, num_classes=1000):
        stem = nn.Sequential(
            conv_bn_act(3,64, ks=7, st=2, act=nn.ReLU(inplace=True)),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        exp = (1 if layers<=34 else 4)
        base = self.get_params(resnet_layers[layers], expansion=exp)
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512*exp, 1000)
        )
        super().__init__(ResBlock, base, stem=stem, head=head)
        self.apply(init_cnn_)

    def get_params(self, mults, expansion=1):
        base_arch = [
            # cin, cout, st, ks
            (64, 64, expansion),
            *n((64*expansion, 64, expansion), mults[0]),
            (64*expansion,128,expansion,2),
            *n((128*expansion,128,expansion), mults[1]),
            (128*expansion,256,expansion,2),
            *n((256*expansion,256,expansion), mults[2]),
            (256*expansion,512,expansion,2),
            *n((512*expansion,512,expansion), mults[3]),
        ]
        return base_arch
