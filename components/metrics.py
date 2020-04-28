import torch
def accuracy(preds, target):
    """
    The average number of correct classes. Only useful
    if the classes are balanced and mostly independent:
    something like ImageNet classification.
    """
    assert preds.shape == target.shape, f"preds.shape ({preds.shape}) != target.shap ({target.shape})"
    return torch.mean((preds==target).float())
