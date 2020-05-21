class Hook():
    "Utility class for creating and destroying a pytorch hook."
    def __init__(self, module, hook_fn, forward_pass=True):
        self.module = module
        self.hook_fn = hook_fn
        if forward_pass:
            self.hook = self.module.register_forward_hook(hook_fn)
        else:
            self.hook = self.module.register_backward_hook(hook_fn)

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
    def __init__(self, modules, hook_fn, forward_pass=True):
        self.module = modules
        self.hook_fn = hook_fn
        if forward_pass:
            self.hooks = [m.register_forward_hook(hook_fn) for m in self.module]
        else:
            self.hooks = [m.register_backward_hook(hook_fn) for m in self.module]

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

