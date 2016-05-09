from multiprocessing import cpu_count


# stolen from sklearn.utils
def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.
    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.
    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.
    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.
    Examples
    --------
    >>> from sklearn.utils import _get_n_jobs
    >>> _get_n_jobs(4)
    4
    >>> jobs = _get_n_jobs(-2)
    >>> assert jobs == max(cpu_count() - 1, 1)
    >>> _get_n_jobs(0)
    Traceback (most recent call last):
    ...
    ValueError: Parameter n_jobs == 0 has no meaning.
    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs
