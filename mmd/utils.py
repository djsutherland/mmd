from __future__ import print_function

import json
import logging
from multiprocessing import cpu_count
import sys


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


################################################################################
# Progress bar handling
# Taken wholesale from skl_groups.utils

class ProgressLogger(object):
    '''
    A little class to use to wrap logging progress to a logger object.

    Parameters
    ----------
    logger : :class:`logging.Logger`
        The logger instance to use.

    name : string, optional, default None
        The name of the process.

    Notes
    -----
    Instances of this class can be used as callables. In that case,
    they take an iterable and return a generator that yields each element, with
    appropriate progress messages in between. For example::

        ProgressLogger(logger)([x ** 2 for x in range(10)])
        ProgressLogger(logger)((x ** 2 for x in range(10)), total=10)

    will each yield the integers 1, 4, 9, ... while logging progress messages
    to ``logger``. In the second case, we passed the ``total`` argument to
    communicate the length of the iterable, which is not available via
    :func:`len`.

    A gotcha here is that :meth:`finish` will only be called if the iterator
    actually terminates. For example::

        for x, y in zip(things, ProgressLogger(logger)(f(x) for x in things)):
            pass

    won't ever :meth:`finish`, because :func:`zip` will see that ``things``
    ends first and not continue iterating over the generator.
    '''
    def __init__(self, logger, name=None):
        self.logger = logger
        self.name = name

    def start(self, total):
        '''
        Signal the start of the process.

        Parameters
        ----------
        total : int
            The total number of steps in the process, or None if unknown.
        '''
        self.logger.info(json.dumps(['START', self.name, total]))

    def update(self, idx):
        '''
        Update the current state.

        Parameters
        ----------
        idx : int
            The current state through the process.
        '''
        # json format, but since we might call this a lot do it a little faster
        self.logger.info('["SET", {}]'.format(idx))

    def finish(self):
        '''
        Marks the process as done.
        '''
        self.logger.info(json.dumps(['DONE']))

    def __call__(self, it, total=None):
        if total is None:
            try:
                total = len(it)
            except TypeError:
                total = None
        self.start(total)
        for i, thing in enumerate(it, 1):
            self.update(i)
            yield thing
        self.finish()


class ProgressBarHandler(logging.Handler):
    '''
    A logging handler that uses the progressbar module to show progress from
    a :class:`ProgressLogger`.

    Takes the same parameters as :class:`progressbar.ProgressBar`,
    but gives a default for ``widgets`` that applies only when maxval is
    available; you'll need to pass different widgets if not.
    '''
    def __init__(self, widgets=None, **kwargs):
        import progressbar as pb

        logging.Handler.__init__(self)

        if widgets is None:
            class CommaProgress(pb.widgets.WidgetBase):
                def __call__(self, progress, data):
                    return '{value:,} of {max_value:,}'.format(**data)

            widgets = [' ', CommaProgress(), ' (', pb.Percentage(), ') ',
                       pb.Bar(), ' ', pb.ETA()]

        self.pbar_args = {'widgets': widgets}
        self.pbar_args.update(kwargs)

    def emit(self, record):
        import progressbar as pb

        msg = json.loads(record.msg)
        # print(msg)
        if msg[0] == 'SET':
            pass
            self.pbar.update(msg[1])
        elif msg[0] == 'START':
            print(msg[1] + ':', file=sys.stderr)
            self.pbar = pb.ProgressBar(maxval=msg[2], **self.pbar_args)
            self.pbar.start()
        elif msg[0] == 'DONE':
            self.pbar.finish()
            del self.pbar
            print('', file=sys.stderr)


def show_progress(name, **kwargs):
    '''
    Sets up a :class:`ProgressBarHandler` to handle progess logs for
    a given module.

    Parameters
    ----------
    name : string
        The module name of the progress logger to use. For example,
        :meth:`mmd.mmd.rbf_mmk`
        uses ``'mmd.mmd.progress'``.

    * : anything
        Other keyword arguments are passed to the :class:`ProgressBarHandler`.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(ProgressBarHandler(**kwargs))
