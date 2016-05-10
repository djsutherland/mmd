import logging

import numpy as np

from ._mmd import _rbf_mmk
from .utils import ProgressLogger


################################################################################
### Logging setup
logger = logging.getLogger(__name__)

progress_logger = logging.getLogger(__name__ + '.progress')
progress_logger.propagate = False
progress_logger.addHandler(logging.NullHandler())

def plog(it, name=None, total=None):
    return ProgressLogger(progress_logger, name=name)(it, total=total)


################################################################################
### Actual functions

def rbf_mmk(X, Y=None, gamma=1, get_X_diag=False, get_Y_diag=False,
            n_jobs=1, method='el-by-el'):
    if method == 'el-by-el':
        return _el_by_el_rbf_mmk(
            X, Y, gamma=gamma, get_X_diag=get_X_diag, get_Y_diag=get_Y_diag,
            n_jobs=n_jobs)
    elif method == 'at-once':
        return _atonce_rbf_mmk(
            X, Y, gamma=gamma, get_X_diag=get_X_diag, get_Y_diag=get_Y_diag)
    else:
        raise ValueError("unknown method '{}'".format(method))


def _atonce_rbf_mmk(X, Y=None, gamma=1, get_X_diag=False, get_Y_diag=False):
    from sklearn.metrics.pairwise import rbf_kernel

    X_stacked = np.vstack(X)
    X_n_samps = np.array([len(x) for x in X], dtype=np.int32)
    X_bounds = np.r_[0, np.cumsum([x for x in X_n_samps])]

    if Y is None or Y is X:
        X_is_Y = True
        Y_stacked = X_stacked
        Y_n_samps = X_n_samps
        Y_bounds = X_bounds
    else:
        X_is_Y = False
        Y_stacked = np.vstack(Y)
        Y_n_samps = np.array([len(y) for y in Y], dtype=np.int32)
        Y_bounds = np.r_[0, np.cumsum([y for y in Y_n_samps])]

    pointwise = rbf_kernel(X_stacked, Y_stacked, gamma=gamma)
    K = np.empty((len(X_n_samps), len(Y_n_samps)))
    for i in range(len(X_n_samps)):
        for j in range(len(Y_n_samps)):
            K[i, j] = pointwise[X_bounds[i]:X_bounds[i + 1],
                                Y_bounds[j]:Y_bounds[j + 1]].mean()

    if get_X_diag or get_Y_diag:
        ret = (K,)
        if X_is_Y:
            diag = np.diagonal(K).copy()
            if get_X_diag:
                ret += (diag,)
            if get_Y_diag:
                ret += (diag,)
        else:
            if get_X_diag:
                ret += (np.array([
                    rbf_kernel(x, gamma=gamma).mean() for x in X
                ]),)
            if get_Y_diag:
                ret += (np.array([
                    rbf_kernel(y, gamma=gamma).mean() for y in Y
                ]),)
        return ret
    else:
        return K


def _el_by_el_rbf_mmk(X, Y=None, gamma=1, get_X_diag=False, get_Y_diag=False,
                      n_jobs=1):
    X_stacked = np.vstack(X)
    X_n_samps = np.array([len(x) for x in X], dtype=np.int32)

    if Y is None or Y is X:
        Y_stacked = X_stacked
        Y_n_samps = X_n_samps
        X_is_Y = True
    else:
        Y_stacked = np.vstack(Y)
        Y_n_samps = np.array([len(y) for y in Y], dtype=np.int32)
        X_is_Y = False

    if all(isinstance(h, logging.NullHandler)
            for h in progress_logger.handlers):
        log_obj = None # don't send progress messages that'll just be dropped
    else:
        log_obj = ProgressLogger(progress_logger, name="RBF mean map kernel")

    return _rbf_mmk(
        X_stacked, X_n_samps, Y_stacked, Y_n_samps, gamma, n_jobs,
        log_obj, X_is_Y, get_X_diag, get_Y_diag)


def rbf_mmd(X, Y=None, gamma=1, squared=False, n_jobs=1,
            X_diag=None, Y_diag=None, method='el-by-el'):
    res = rbf_mmk(X, Y, gamma, n_jobs=n_jobs,
                  subsample_gamma=subsample_gamma,
                  get_X_diag=X_diag is None, get_Y_diag=Y_diag is None)
    K = res[0]
    if X_diag is None:
        X_diag = res[1]
    if Y_diag is None:
        Y_diag = res[-1]

    K *= -2
    K += X_diag[:, None]
    K += Y_diag[None, :]
    if not squared:
        np.sqrt(K, out=K)
    return K

