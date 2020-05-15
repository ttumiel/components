from torch import nn


class SequentialEx(nn.Sequential):
    """
    Extends nn.Sequential with additional functionality.

    - Allows the model to be sliced easily.
    - If any of the modules passed to init are None, they
      are ignored.
    - Add extended indexing features (find by str)
    """
    def __init__(self, *args):
        super().__init__(*[a for a in args if a is not None])

    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)):
            return super().__getitem__(idx)
        elif isinstance(idx, str):
            layers = idx.split("/")
            l = self
            for layer in layers:
                l = getattr(l, layer)
            return l
        elif isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self) if m]
        else:
            return [self[i] for i in idx]

    def __setitem__(self, name, item):
        if "/" in name:
            root, name = name.rsplit('/', maxsplit=1)
            root_module = self[root]
            setattr(root_module, name, item)
        else:
            setattr(self, name, item)

    def freeze(self, bn=False):
        def _inner(m):
            if bn or not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(False)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(False)
        self.apply(_inner)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
