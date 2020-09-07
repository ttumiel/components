import torch
from sklearn.metrics import cohen_kappa_score
from functools import partial


def wrap_metric(metric, detach=True, clone=False, cpu=False, numpy=False, pred_func=None, target_func=None):
    """Decorates a metric function and applies various transforms to the
    predictions and targets passed into the function.
    """
    def _inner(p, y):
        if detach: p,y = p.detach(),y.detach()
        if clone: p,y = p.clone(),y.clone()
        if cpu: p,y = p.cpu(),y.cpu()
        if numpy: p,y = p.numpy(),y.numpy()
        if pred_func is not None:
            p = pred_func(p)
        if target_func is not None:
            y = target_func(y)
        return metric(p,y)

    _inner.__name__ = metric.__name__
    return _inner


@wrap_metric
def accuracy(preds, target):
    """
    The average number of correct classes. Only useful
    if the classes are balanced and mostly independent:
    something like ImageNet classification.
    """
    if preds.ndim>1: preds = preds.argmax(1)
    assert preds.shape == target.shape, f"preds.shape ({preds.shape}) != target.shap ({target.shape})"
    return torch.mean((preds==target).float())


@partial(wrap_metric, cpu=True, numpy=True)
def quadratic_kappa(preds, target):
    """
    Quadratic kappa score. Used for regression tasks to quadratically penalize
    values that are far from the correct answer. Works well to test inter-
    observer variability. See https://en.wikipedia.org/wiki/Cohen%27s_kappa
    """
    assert preds.shape == target.shape, f"preds.shape ({preds.shape}) != target.shap ({target.shape})"
    return torch.tensor(cohen_kappa_score(preds, target, weights='quadratic'))
