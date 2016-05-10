from __future__ import division, print_function

from cython cimport boundscheck, floating, wraparound
from cython.parallel import prange, threadid
from cpython.exc cimport PyErr_CheckSignals

import numpy as np
from .utils import _get_n_jobs

cdef extern from "math.h":
    double exp(double x) nogil
    float expf(float x) nogil

cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

# TODO: support other kernels
# TODO: support computing multiple gammas at once

@boundscheck(False)
@wraparound(False)
def _rbf_mmk(floating[:, ::1] X_stacked not None, int[::1] X_n_samps not None,
             floating[:, ::1] Y_stacked not None, int[::1] Y_n_samps not None,
             floating gamma, int n_jobs, log_progress,
             bint X_is_Y, bint get_X_diag, bint get_Y_diag):
    # NOTE: use longs for indices in the number of bags and especially the
    #       product of the number of bags, since that actually might overflow
    #       an int -- but use ints for the number of samples per bag (and their
    #       product), since the BLAS interface only works with ints
    cdef object float_t = np.float32 if floating is float else np.float64
    n_jobs = _get_n_jobs(n_jobs)
    cdef long i, j

    cdef long X_n = X_n_samps.shape[0]
    cdef long X_N = X_stacked.shape[0]
    cdef int d = X_stacked.shape[1]
    cdef long[::1] X_bounds = np.r_[0, np.cumsum(X_n_samps)]

    cdef floating[::1] XXs_stacked = np.empty(X_N, dtype=float_t)
    for i in range(X_N):
        XXs_stacked[i] = 0
        for j in range(d):
            XXs_stacked[i] += X_stacked[i, j] ** 2

    cdef long Y_n, Y_N
    cdef long[::1] Y_bounds
    cdef floating[::1] YYs_stacked

    if X_is_Y:
        Y_n = X_n
        Y_N = X_N
        Y_bounds = X_bounds
        YYs_stacked = XXs_stacked
    else:
        Y_n = Y_n_samps.shape[0]
        Y_N = Y_stacked.shape[0]
        assert Y_stacked.shape[1] == d
        Y_bounds = np.r_[0, np.cumsum(Y_n_samps)]
        YYs_stacked = np.empty(Y_N, dtype=float_t)
        for i in range(Y_N):
            YYs_stacked[i] = 0
            for j in range(d):
                YYs_stacked[i] += Y_stacked[i, j] ** 2

    # allocate work buffer for each thread
    cdef int max_pts = max(np.max(X_n_samps), np.max(Y_n_samps))
    cdef floating[:, ::1] tmps = np.empty(
            (n_jobs, max_pts * max_pts), dtype=float_t)

    cdef floating[:, ::1] K = np.empty((X_n, Y_n), dtype=float_t)
    cdef floating[::1] X_diag, Y_diag

    cdef long num_main_jobs
    cdef long[::1] main_job_is, main_job_js
    if X_is_Y:  # only do the upper half of the array...
        main_job_is, main_job_js = [
                np.ascontiguousarray(a) for a in np.triu_indices(X_n)]
        num_main_jobs = main_job_is.shape[0]
    else:
        num_main_jobs = X_n * Y_n

    cdef long total_to_do = num_main_jobs
    if not X_is_Y:
        if get_X_diag:
            X_diag = np.empty(X_n, dtype=float_t)
            total_to_do += X_n
        if get_Y_diag:
            Y_diag = np.empty(Y_n, dtype=float_t)
            total_to_do += Y_n

    cdef object pbar = log_progress
    cdef bint do_progress = bool(pbar)
    cdef long jobs_since_last_tick_val
    cdef long * jobs_since_last_tick = &jobs_since_last_tick_val
    cdef long[::1] num_done  # total things done by each thread
    if do_progress:
        num_done = np.zeros(n_jobs, dtype=np.int64)
        pbar.start(total=total_to_do)

    cdef long job_idx, tid, i_start, i_end, j_start, j_end
    cdef floating minus_gamma = -gamma
    with nogil:
        for job_idx in prange(total_to_do, num_threads=n_jobs,
                              schedule='static'):
            tid = threadid()
            if tid == 0:
                with gil:
                    PyErr_CheckSignals()  # allow ^C to interrupt us
                if do_progress:
                    handle_pbar(pbar, jobs_since_last_tick, num_done)

            if job_idx < num_main_jobs:
                if X_is_Y:
                    i = main_job_is[job_idx]
                    j = main_job_js[job_idx]
                else:
                    i = job_idx // Y_n
                    j = job_idx % Y_n
                i_start = X_bounds[i]
                i_end = X_bounds[i + 1]
                j_start = Y_bounds[j]
                j_end = Y_bounds[j + 1]

                K[i, j] = _mean_rbf_kernel(
                    X_stacked[i_start:i_end, :],
                    Y_stacked[j_start:j_end, :],
                    XXs_stacked[i_start:i_end],
                    YYs_stacked[j_start:j_end],
                    minus_gamma, tmps[tid, :])

                if X_is_Y:
                    K[j, i] = K[i, j]

            else:
                i = job_idx - num_main_jobs
                if get_X_diag and i < X_n:
                    i_start = X_bounds[i]
                    i_end = X_bounds[i + 1]
                    X_diag[i] = _mean_rbf_kernel(
                        X_stacked[i_start:i_end, :],
                        X_stacked[i_start:i_end, :],
                        XXs_stacked[i_start:i_end],
                        XXs_stacked[i_start:i_end],
                        minus_gamma, tmps[tid, :])
                else:
                    if get_X_diag:
                        i = i - X_n
                    i_start = Y_bounds[i]
                    i_end = Y_bounds[i + 1]
                    Y_diag[i] = _mean_rbf_kernel(
                        Y_stacked[i_start:i_end, :],
                        Y_stacked[i_start:i_end, :],
                        YYs_stacked[i_start:i_end],
                        YYs_stacked[i_start:i_end],
                        minus_gamma, tmps[tid, :])

            if do_progress:
                num_done[tid] += 1

    if do_progress:
        pbar.finish()

    if get_X_diag or get_Y_diag:
        if X_is_Y:
            X_diag = Y_diag = np.diagonal(K).copy()

        ret = (np.asarray(K),)
        if get_X_diag:
            ret += (np.asarray(X_diag),)
        if get_Y_diag:
            ret += (np.asarray(Y_diag),)
        return ret
    else:
        return np.asarray(K)


@boundscheck(False)
@wraparound(False)
cdef floating _mean_rbf_kernel(
            floating[:, ::1] X, floating[:, ::1] Y,
            floating[::1] XX, floating[::1] YY,
            floating minus_gamma, floating[::1] tmp) nogil:
    cdef int num_X = X.shape[0], num_Y = Y.shape[0], dim = X.shape[1]
    cdef int num_prod = num_X * num_Y
    cdef floating zero_f = 0, minus_two_f = -2
    cdef int one_i = 1
    cdef int i, j

    # put  -2 X Y^T  into tmp
    # X, Y are stored as row-major feature arrays; blas expects col-major
    if floating is float:
        blas.sgemm('T', 'N', &num_X, &num_Y, &dim, &minus_two_f,
                   &X[0, 0], &dim,
                   &Y[0, 0], &dim,
                   &zero_f, &tmp[0], &num_X)
    else:
        blas.dgemm('T', 'N', &num_X, &num_Y, &dim, &minus_two_f,
                   &X[0, 0], &dim,
                   &Y[0, 0], &dim,
                   &zero_f, &tmp[0], &num_X)

    # add row norms - NOTE: not vectorized...
    for i in range(num_X):
        for j in range(num_Y):
            tmp[i + num_X * j] += XX[i] + YY[j]

    # scale by -gamma
    if floating is float:
        blas.sscal(&num_prod, &minus_gamma, &tmp[0], &one_i)
    else:
        blas.dscal(&num_prod, &minus_gamma, &tmp[0], &one_i)

    # exponentiate and mean - NOTE: not vectorized...
    cdef floating res = 0
    for i in range(num_X):
        for j in range(num_Y):
            if floating is float:
                res += expf(tmp[i + num_X * j])
            else:
                res += exp(tmp[i + num_X * j])
    return res / num_prod


@boundscheck(False)
@wraparound(False)
cdef bint handle_pbar(object pbar, long * jobs_since_last_tick,
                      long[::1] num_done) nogil except 1:
    jobs_since_last_tick[0] += 1
    cdef long done_count = 0

    if jobs_since_last_tick[0] >= 20:  # TODO: tweak? do it based on time?
        for k in range(num_done.shape[0]):
            done_count += num_done[k]

        with gil:
            pbar.update(done_count)
        jobs_since_last_tick[0] = 0
