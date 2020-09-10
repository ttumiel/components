from collections import OrderedDict
import matplotlib.pyplot as plt

from components.utils import find_all
from components.hooks import Hooks, capture_activations_hook, capture_params_hook

class Callback():
    def __init__(self, module, method):
        self.module = module
        self.method = method

    def __call__(self):
        self.method(self, self.module)

class Callbacks():
    def __init__(self, modules, method):
        self.hooks = [Callback(m, method) for m in modules]
    def __call__(self):
        for m in self.hooks:
            m()

def grad_callback(cb, module):
    if not hasattr(cb, 'grad_mean'): cb.grad_mean = []
    if not hasattr(cb, 'grad_std'): cb.grad_std = []
    cb.grad_mean.append(module.weight.grad.mean())
    cb.grad_std.append(module.weight.grad.std())


class Telemetry():
    def __init__(self, module, entry, activations=True, parameters=True, grads=True):
        self.module = module
        self.entries = find_all(self.module, entry)
        self.hooks = OrderedDict()
        self.names = [c.__class__.__name__+'_'+str(i) for i,c in enumerate(self.entries)]
        if activations:
            self.hooks['act'] = Hooks(self.entries, capture_activations_hook)
        if parameters:
            self.hooks['param'] = Hooks(self.entries, capture_params_hook)
        if grads:
            self.hooks['grad'] = Callbacks(self.entries, grad_callback)
            def on_after_backward(): self.hooks['grad']()
            self.module.on_after_backward = on_after_backward

    def remove(self):
        for k,v in self.hooks.items():
            if k != 'grad':
                v.remove()

    def __del__(self):
        self.remove()

    def plot(self):
        f,ax = plt.subplots(len(self.hooks),2, figsize=(15,25))
        ax = ax.flatten()

        for i,(k,hs) in enumerate(self.hooks.items()):
            for j,h in enumerate(hs.hooks):
                means = getattr(h, k+'_mean')
                stds = getattr(h, k+'_std')

                ax[i*2].plot(means, label=self.names[j] if i == 0 else None)
                ax[i*2].set_title(k.capitalize()+' Means')
                ax[i*2+1].plot(stds)
                ax[i*2+1].set_title(k.capitalize()+' Stds')

        ax[0].legend()
