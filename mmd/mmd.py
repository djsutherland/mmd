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

def rbf_mmk(X, Y=None, gamma=1, n_jobs=1, subsample_gamma=1000,
            get_X_diag=False, get_Y_diag=False):
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

    return _rbf_mmk(
        X_stacked, X_n_samps, Y_stacked, Y_n_samps, gamma, n_jobs,
        ProgressLogger(progress_logger, name="RBF mean map kernel"),
        X_is_Y, get_X_diag, get_Y_diag)
    # TODO: if there's no non-null handler attached, don't send a logger obj

def rbf_mmd(X, Y=None, gamma=1, squared=False, n_jobs=1, subsample_gamma=1000,
            X_diag=None, Y_diag=None):
    if Y is None or Y is X:
        K = rbf_mmk(X, None, gamma, n_jobs=n_jobs,
                    subsample_gamma=subsample_gamma)
        X_diag = Y_diag = np.diagonal(K).copy()
    else:
        get_X_diag = X_diag is None
        get_Y_diag = Y_diag is None
        res = rbf_mmk(X, Y, gamma, n_jobs=n_jobs,
                      subsample_gamma=subsample_gamma,
                      get_X_diag=get_X_diag, get_Y_diag=get_Y_diag)
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

