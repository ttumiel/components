import numpy as np

def plot_with_error(x, ys, *args, ax=None, alpha=0.3, **kwargs):
    "Plots a graph with a 1 standard deviation error outline."
    if not isinstance(ys, np.ndarray):
        ys = np.stack(ys)

    std = np.std(ys, axis=0)
    mean = np.mean(ys, axis=0)
    if ax is None:
        ax = plt
    ax.plot(x,mean, *args, **kwargs)
    ax.fill_between(x, mean-std, mean+std, alpha=alpha)
