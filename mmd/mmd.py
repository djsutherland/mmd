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

def rbf_mmk(samps, gamma, n_jobs=1):
    stacked = np.vstack(samps)
    n_samps = np.array([len(x) for x in samps], dtype=np.int32)
    return _rbf_mmk(
        stacked, n_samps, gamma, n_jobs,
        ProgressLogger(progress_logger, name="RBF mean map kernel"))

def rbf_mmd(samps, gamma, squared=False, n_jobs=1):
    K = rbf_mmk(samps, gamma, n_jobs=n_jobs)
    diag = np.diagonal(K).copy()
    K *= -2
    K += diag[:, None]
    K += diag[None, :]
    if not squared:
        np.sqrt(K, out=K)
    return K

