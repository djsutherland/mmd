from __future__ import division

import numpy as np
from cython cimport boundscheck, floating
from cython.parallel import prange, threadid
from sklearn.utils import _get_n_jobs

cdef extern from "math.h":
    double exp(double x) nogil
    float expf(float x) nogil

cimport scipy.linalg.cython_blas as blas 
cimport scipy.linalg.cython_lapack as lapack 


@boundscheck(False)
cpdef _rbf_mmk(floating[:, ::1] stacked, int[::1] n_samps,
               floating gamma, int n_jobs):
    cdef object float_t = np.float32 if floating is float else np.float64
    n_jobs = _get_n_jobs(n_jobs)
  
    cdef int n = n_samps.shape[0]
    cdef int N = stacked.shape[0]
    cdef int d = stacked.shape[1]
    cdef int[::1] bounds = np.r_[0, np.cumsum(n_samps)].astype(np.int32)
    cdef int i, j, k, l

    cdef floating[:, ::1] K = np.empty((n, n), dtype=float_t)

    # allocate work buffer for each thread
    cdef int max_pts = np.max(n_samps)
    cdef floating[:, ::1] tmps = np.empty(
            (n_jobs, max_pts * max_pts), dtype=float_t)
  
    cdef floating[::1] XXs_stacked = np.empty(N, dtype=float_t)
    for i in range(N):
        XXs_stacked[i] = 0
        for j in range(d):
            XXs_stacked[i] += stacked[i, j] ** 2
  
    cdef int one_i = 1
    cdef floating zero_f = 0, one_f = 1, minus_two_f = -2
    cdef floating minus_gamma = -gamma

    cdef int i_start, i_end, num_i
    cdef int j_start, j_end, num_j
    cdef int num_prod, job_i, tid
    cdef floating inc

    with nogil:
        for job_i in prange(n ** 2, num_threads=n_jobs, schedule='static'):
            i = job_i // n
            j = job_i % n
            tid = threadid()

            i_start = bounds[i]
            i_end = bounds[i + 1]
            num_i = i_end - i_start

            j_start = bounds[j]
            j_end = bounds[j + 1]
            num_j = j_end - j_start
          
            num_prod = num_i * num_j

            # TODO: avoid code duplication here...
            # TODO: support other kernels
            # TODO: optional progress info like in skl-groups
            if floating is float:
                # put  X Y^T  into tmps
                blas.sgemm('N', 'T', &num_i, &num_j, &d, &one_f,
                           &stacked[i_start, 0], &num_i,
                           &stacked[j_start, 0], &num_j,
                           &zero_f, &tmps[tid, 0], &num_i)
              
                # scale by -2
                blas.sscal(&num_prod, &minus_two_f, &tmps[tid, 0], &one_i)
              
                # add row norms - NOTE: not vectorized...
                for k in range(num_i):
                    for l in range(num_j):
                        tmps[tid, k + num_i * l] += (
                                XXs_stacked[i_start + k]
                              + XXs_stacked[j_start + l])
              
                # scale by -gamma
                blas.sscal(&num_prod, &minus_gamma, &tmps[tid, 0], &one_i)
              
                # exponentiate and mean - NOTE: not vectorized...
                K[i, j] = 0
                for k in range(num_i):
                    for l in range(num_j):
                        inc = expf(tmps[tid, k + num_i * l])
                        K[i, j] += inc
              
                K[i, j] /= num_prod
            else:
                # put  X Y^T  into tmps
                blas.dgemm('N', 'T', &num_i, &num_j, &d, &one_f,
                           &stacked[i_start, 0], &num_i,
                           &stacked[j_start, 0], &num_j,
                           &zero_f, &tmps[tid, 0], &num_i)
              
                # scale by -2
                blas.dscal(&num_prod, &minus_two_f, &tmps[tid, 0], &one_i)
              
                # add row norms - NOTE: not vectorized...
                for k in range(num_i):
                    for l in range(num_j):
                        tmps[tid, k + num_i * l] += (
                                XXs_stacked[i_start + k]
                              + XXs_stacked[j_start + l])
              
                # scale by -gamma
                blas.dscal(&num_prod, &minus_gamma, &tmps[tid, 0], &one_i)
              
                # exponentiate and mean - NOTE: not vectorized...
                K[i, j] = 0
                for k in range(num_i):
                    for l in range(num_j):
                        inc = exp(tmps[tid, k + num_i * l])
                        K[i, j] += inc
              
                K[i, j] /= num_prod
    return np.asarray(K)
