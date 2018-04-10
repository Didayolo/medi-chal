import numpy as np
from scipy.spatial.distance import pdist, squareform


def euclidean(x, y, axis=None):
    """
        Compute the euclidean distance.
    """
    return np.linalg.norm(x - y, ord=2, axis=axis)


def maximum(x, y, axis=None):
    """
        Compute the maximum distance.
    """
    return np.linalg.norm(x - y, ord='inf', axis=axis)


def minimum(x, y, axis=None):
    """
        Compute the minimum distance.
    """
    return np.linalg.norm(x - y, ord='-inf', axis=axis)


def manhattan(x, y, axis=None):
    """
        Compute the manhattan distance.
    """
    return np.linalg.norm(x - y, ord=1, axis=axis)


def distcorr(X, Y):
    """
        Compute the distance correlation function.
        Works with X and Y of different dimensions (but same number of samples mandatory).
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
