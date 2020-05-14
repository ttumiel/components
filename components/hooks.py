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
