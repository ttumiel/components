"Convolutional utilities."

from torch import nn

def conv(cin, cout, ks=3, st=1, act=nn.ReLU(True)):
    "Sequential conv, relu, bn."
    return nn.Sequential(
        nn.Conv2d(cin, cout, ks, st, padding=ks//2, bias=False),
        act,
        nn.BatchNorm2d(cout)
    )

def convT(cin, cout, ks=4, st=2, act=nn.LeakyReLU(0.2, True)):
    "Sequential transpose conv, relu, conv. Upsamples by 2."
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, ks, st, padding=1),
        act,
        nn.BatchNorm2d(cout)
    )

class ResBlock(nn.Module):
    def __init__(self, cin, cout, st=1, ks=3, act=nn.ReLU(True)):
        super().__init__()
        self.st = st

        self.conv1 = conv(cin, cout, ks, st, act=act)
#         self.conv2 = conv(cout, cout, ks, st, act=act)

        if cin != cout:
            self.bot = nn.Conv2d(cin, cout, 1, bias=False)
        else:
            self.bot = None

    def forward(self, x):
        inp = x

#         if self.st != 1:
#             inp = nn.Bilinear()

        x = self.conv1(x)
#         x = self.conv2(x)

        if self.bot is not None:
            inp = self.bot(inp)

        return x + inp
