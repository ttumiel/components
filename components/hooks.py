import torch
from functools import partial

class Hook():
    "Utility class for creating and destroying a pytorch hook."
    def __init__(self, module, hook_fn, forward_pass=True, detach=True, clone=False):
        self.module = module
        self.hook_fn = hook_fn

        def _hook(m,i,o):
            self.hook_fn(self, m, clone_detach(i, detach, clone), clone_detach(o, detach, clone))

        self._hook = _hook
        if forward_pass:
            self.hook = self.module.register_forward_hook(_hook)
        else:
            self.hook = self.module.register_backward_hook(_hook)

    def remove(self):
        self.hook.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __repr__(self):
        return f"Hook: {self.hook_fn.__name__} on {self.module.__class__.__name__}"

class Hooks(Hook):
    "Create hooks on each item of a list of modules."
    def __init__(self, modules, hook_fn, forward_pass=True, detach=True, clone=False):
        self.module = modules
        self.hook_fn = hook_fn
        self.hooks = [Hook(m, hook_fn, forward_pass, detach, clone) for m in self.module]

    def remove(self):
        for h in self.hooks:
            h.remove()


def capture_activations_hook(hook_cls, m, i, o, display=False):
    if not hasattr(hook_cls, 'act_mean'): hook_cls.act_mean = []
    if not hasattr(hook_cls, 'act_std'): hook_cls.act_std = []
    if m.training:
        hook_cls.act_mean.append(o.mean())
        hook_cls.act_std.append(o.std())

    if display:
        data = [o.mean(), o.std()]
        print("{:^16}".format(m.__class__.__name__), ("|{:^10.5f}"*len(data)).format(*data))


def capture_params_hook(hook_cls, m, i, o, display=False):
    if not hasattr(hook_cls, 'param_mean'): hook_cls.param_mean = []
    if not hasattr(hook_cls, 'param_std'): hook_cls.param_std = []
    if m.training:
        hook_cls.param_mean.append(m.weight.detach().mean())
        hook_cls.param_std.append(m.weight.detach().std())

    if display:
        data = [m.weight.mean(), m.weight.std()]
        if bias and hasattr(m, 'bias') and m.bias is not None:
            data += [m.bias.mean(), m.bias.std()]
        print("{:^16}".format(m.__class__.__name__), ("|{:^10.5f}"*len(data)).format(*data))



def clone_detach(tensor, detach, clone):
    if isinstance(tensor, tuple):
        if detach:
            tensor = tuple(v.detach() for v in tensor)
        if clone:
            tensor = tuple(v.clone() for v in tensor)
    elif isinstance(tensor, torch.Tensor):
        if detach:
            tensor = tensor.detach()
        if clone:
            tensor = tensor.clone()
    return tensor
