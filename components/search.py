"Perform a grid search over a dictionary of hyper parameters."

def is_search(hparams):
    return any(isinstance(p, list) for p in hparams.values())

def _grid(hparams, *args):
    if len(hparams) == 1:
        for v in hparams[0]:
            yield (*args, v)
    else:
        for v in hparams[0]:
            yield from _grid(hparams[1:], *args, v)

def grid_search(hparams):
    """Performs a grid search over a dictionary of
    hyper-parameters for each item that is a list.
    """
    others, keys, searches = {},[],[]
    for k,v in hparams.items():
        if isinstance(v, list):
            keys.append(k)
            searches.append(v)
        else:
            others[k] = v

    for vs in _grid(searches):
        out = {k:v for k,v in zip(keys, vs)}
        out.update(others)
        yield out
