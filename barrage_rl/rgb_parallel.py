"""Window-local RGB parallelism; restore the planner's thread mask per call."""
from contextlib import contextmanager
from contextvars import ContextVar
import os

DEFAULT_RGB_WORKERS = 4
_workers = ContextVar('window_rgb_workers', default=0)

@contextmanager
def rgb_workers(workers):
    token = _workers.set(max(0, int(workers)))
    try:
        yield
    finally:
        _workers.reset(token)

def dispatch(serial, parallel, *args):
    requested = _workers.get()
    if requested <= 1 or parallel is None:
        return serial(*args)
    from numba import get_num_threads, set_num_threads, config
    count = min(requested, os.cpu_count() or 1, config.NUMBA_NUM_THREADS)
    if count <= 1:
        return serial(*args)
    previous = get_num_threads()
    try:
        if previous != count:
            set_num_threads(count)
        return parallel(*args)
    finally:
        if previous != count:
            set_num_threads(previous)
