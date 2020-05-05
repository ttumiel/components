from torch import nn

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes, smooth=0.0, dim=-1):
        super().__init__()
        self.smooth = smooth
        self.c = classes
        self.dim = dim

    def forward(self, preds, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smooth / (self.c - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1-self.smooth)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
