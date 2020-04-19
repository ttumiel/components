"""
Components for building a GAN.
"""

from components import convT, conv, ResBlock
from torch import nn

datasets = [
    'https://storage.cloud.google.com/cartoonset_public_files/cartoonset100k.tgz',
    'https://drive.google.com/open?id=1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP', # FFHQ
]

class BasicGenerator(nn.Module):
    "A basic generator architecture that will upsize to 32x32 px"
    def __init__(self, latent_dim=100, cout=3):
        super().__init__()
        self.main = nn.Sequential(
            convT(latent_dim, 512),
            convT(512, 256),
            convT(256, 128),
            convT(128, 64),
            convT(64, cout),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class BasicDiscriminator(nn.Module):
    "A basic discriminator that will take in any size image into a binary value."
    def __init__(self, cin=3):
        super().__init__()
        self.main = nn.Sequential(
            conv(cin, 32, st=2),
            conv(32, 64, st=2),
            conv(64, 128, st=2),
            conv(128, 256, st=2),
            conv(256, 512, st=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, input):
        return self.main(input)


class ResDiscriminator(nn.Module):
    "A basic discriminator that will take in any size image into a binary value."
    def __init__(self, cin=3, act=nn.LeakyReLU(0.2, True)):
        super().__init__()
        self.main = nn.Sequential(
            ResBlock(cin, 32, act=act),
            conv(32,64,act=act),
            ResBlock(64, 64, act=act),
            conv(64,128,act=act),
            ResBlock(128, 128, act=act),
            conv(128,256,act=act),
            ResBlock(256, 256, act=act),
            conv(256,512,act=act),
            ResBlock(512, 512, act=act),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        return self.main(input)
