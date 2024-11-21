from tqdm.auto import tqdm


def vtqdm(x, verbose=False, **kwargs):
    if verbose:
        return tqdm(x, **kwargs)
    else:
        return x
