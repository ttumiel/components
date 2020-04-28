import torch
from sklearn.metrics import cohen_kappa_score


def accuracy(preds, target):
    """
    The average number of correct classes. Only useful
    if the classes are balanced and mostly independent:
    something like ImageNet classification.
    """
    assert preds.shape == target.shape, f"preds.shape ({preds.shape}) != target.shap ({target.shape})"
    return torch.mean((preds==target).float())


def quadratic_kappa(preds, target):
    """
    Quadratic kappa score. Used for regression tasks to quadratically penalize
    values that are far from the correct answer. Works well to test inter-
    observer variability. See https://en.wikipedia.org/wiki/Cohen%27s_kappa
    """
    assert preds.shape == target.shape, f"preds.shape ({preds.shape}) != target.shap ({target.shape})"
    return torch.tensor(cohen_kappa_score(preds.detach().cpu().numpy(), target.detach().cpu().numpy(), weights='quadratic'))
