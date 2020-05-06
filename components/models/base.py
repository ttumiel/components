from torch import nn

def basic_model_head(cin, n_classes):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(cin, n_classes)
    )

def large_model_head(cin, n_classes, p=0.5, n_hidden=512):
    return nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.Linear(cin*2, n_hidden),
        nn.BatchNorm1d(n_hidden),
        nn.Dropout(p=p),
        nn.ReLU(True),
        nn.Linear(n_hidden, n_classes)
    )

class ModelBuilder(nn.Module):
    def __init__(self, blocktype, param_list, head=None, stem=None, extras=None):
        super().__init__()

        if not isinstance(blocktype, list):
            if not isinstance(stem, nn.Module) and callable(stem):
                blocktype = [stem]+[blocktype]*(len(param_list)-1)
            else:
                blocktype = [blocktype]*len(param_list)

        blocks = [block(*params) for block,params in zip(blocktype,param_list)]

        if extras is not None:
            blocks.append(extras)

        if isinstance(stem, nn.Module):
            self.stem = stem
        else:
            self.stem = None
        self.body = nn.Sequential(*blocks)
        self.head = head

    def forward(self, *inputs):
        if self.stem is not None:
            inputs = self.stem(*inputs)
            features = self.body(inputs)
        else:
            features = self.body(*inputs)
        return features if self.head is None else self.head(features)
