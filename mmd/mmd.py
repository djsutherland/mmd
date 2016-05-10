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

def rbf_mmk(X, Y=None, gammas=1, get_X_diag=False, get_Y_diag=False,
            n_jobs=1, method='el-by-el'):
    if method == 'el-by-el':
        return _el_by_el_rbf_mmk(
            X, Y, gammas=gammas, get_X_diag=get_X_diag, get_Y_diag=get_Y_diag,
            n_jobs=n_jobs)
    elif method == 'at-once':
        return _atonce_rbf_mmk(
            X, Y, gammas=gammas, get_X_diag=get_X_diag, get_Y_diag=get_Y_diag)
    else:
        raise ValueError("unknown method '{}'".format(method))


def _atonce_rbf_mmk(X, Y=None, gammas=1, get_X_diag=False, get_Y_diag=False):
    from sklearn.metrics.pairwise import euclidean_distances

    X_stacked = np.vstack(X)
    X_n_samps = np.array([len(x) for x in X], dtype=np.int32)
    X_bounds = np.r_[0, np.cumsum([x for x in X_n_samps])]
    X_n = len(X_n_samps)

    if Y is None or Y is X:
        Y = X
        X_is_Y = True
        Y_stacked = X_stacked
        Y_n_samps = X_n_samps
        Y_bounds = X_bounds
        Y_n = X_n
    else:
        X_is_Y = False
        Y_stacked = np.vstack(Y)
        Y_n_samps = np.array([len(y) for y in Y], dtype=np.int32)
        Y_bounds = np.r_[0, np.cumsum([y for y in Y_n_samps])]
        Y_n = len(Y_n_samps)

    gammas = np.asarray(gammas, dtype=X_stacked.dtype)
    scalar_gamma = gammas.ndim == 0
    if scalar_gamma:
        gammas = np.ascontiguousarray(gammas[None])

    pointwise = (-gammas[:, None, None]) * \
        euclidean_distances(X_stacked, Y_stacked, squared=True)[None, :, :]
    np.exp(pointwise, out=pointwise)

    K = np.empty((len(gammas), X_n, Y_n))
    for i in range(X_n):
        for j in range(Y_n):
            K[:, i, j] = pointwise[
                :, X_bounds[i]:X_bounds[i + 1], Y_bounds[j]:Y_bounds[j + 1]
            ].mean(axis=(1, 2))

    if get_X_diag or get_Y_diag:
        ret = (K,)
        if X_is_Y:
            diag = K[:, xrange(X_n), xrange(X_n)].copy()
            if get_X_diag:
                ret += (diag,)
            if get_Y_diag:
                ret += (diag,)
        else:
            if get_X_diag:
                X_diag = np.empty((len(gammas), X_n))
                for i, x in enumerate(X):
                    pointwise = (-gammas[:, None, None]) * \
                        euclidean_distances(x, squared=True)[None, :, :]
                    np.exp(pointwise, out=pointwise)
                    X_diag[:, i] = pointwise.mean(axis=(1, 2))
                ret += (X_diag,)

            if get_Y_diag:
                Y_diag = np.empty((len(gammas), Y_n))
                for i, y in enumerate(Y):
                    pointwise = (-gammas[:, None, None]) * \
                        euclidean_distances(y, squared=True)[None, :, :]
                    np.exp(pointwise, out=pointwise)
                    Y_diag[:, i] = pointwise.mean(axis=(1, 2))
                ret += (Y_diag,)

        return tuple(r[0] for r in ret) if scalar_gamma else ret
    else:
        return K[0] if scalar_gamma else K


def _el_by_el_rbf_mmk(X, Y=None, gammas=1, get_X_diag=False, get_Y_diag=False,
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

    gammas = np.asarray(gammas, dtype=X_stacked.dtype)
    scalar_gamma = gammas.ndim == 0
    if scalar_gamma:
        gammas = np.ascontiguousarray(gammas[None])

    res = _rbf_mmk(
        X_stacked, X_n_samps, Y_stacked, Y_n_samps, gammas, n_jobs,
        log_obj, X_is_Y, get_X_diag, get_Y_diag)

    if scalar_gamma:
        if get_X_diag or get_Y_diag:
            return tuple(x[0] for x in res)
        else:
            return res[0]
    return res


def rbf_mmd(X, Y=None, gammas=1, squared=False, n_jobs=1,
            X_diag=None, Y_diag=None, method='el-by-el'):
    res = rbf_mmk(X, Y, gammas,
                  get_X_diag=X_diag is None, get_Y_diag=Y_diag is None,
                  n_jobs=n_jobs, method=method)
    K = res[0]
    if X_diag is None:
        X_diag = res[1]
    if Y_diag is None:
        Y_diag = res[-1]

    K *= -2
    K += X_diag[..., :, None]
    K += Y_diag[..., None, :]
    if not squared:
        np.sqrt(K, out=K)
    return K

