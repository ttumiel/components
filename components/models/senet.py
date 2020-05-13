from torch import nn

class SqueezeExcite(nn.Module):
    "Squeeze and excitation module."
    def __init__(self, cin, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, cin//reduction_ratio, 1),
            nn.ReLU(True),
            nn.Conv2d(cin//reduction_ratio, cin, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.squeeze(x)
        return x * squeeze
