import numpy as np

from ._mmd import _rbf_mmk

def rbf_mmk(samps, gamma, n_jobs=1):
    stacked = np.vstack(samps)
    n_samps = np.array([len(x) for x in samps], dtype=np.int32)
    return _rbf_mmk(stacked, n_samps, gamma, n_jobs)

def rbf_mmd(samps, gamma, n_jobs=1):
    K = rbf_mmk(samps, gamma, n_jobs=n_jobs)
    diag = np.diagonal(K).copy()
    K *= -2
    K += diag[:, None]
    K += diag[None, :]
    return K

