"""Timing buffer accounting, export compatibility, and exception unwinding."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.window_latency_breakdown import WindowProfiler


def test_nested_exclusive_accounting_and_growth():
    target = SimpleNamespace()
    target.child = lambda: 7
    target.parent = lambda: target.child()
    profiler = WindowProfiler(initial_capacity=1)
    profiler.wrap(target, 'child', 'child')
    profiler.wrap(target, 'parent', 'parent')
    with patch('tools.window_latency_breakdown.time.perf_counter', side_effect=[0.,1.,3.,5.,6.,7.]):
        profiler._begin()
        assert target.parent() == 7
        profiler._finish(5000.)
        profiler._begin()
        assert target.child() == 7
        profiler._finish(1000.)
    assert profiler.rows == [
        {'total_ms':5000.,'stages':{'child':[2000.,2000.],'parent':[5000.,3000.]}},
        {'total_ms':1000.,'stages':{'child':[1000.,1000.]}}]
    report = profiler.report()
    assert report['decisions'] == 2
    assert report['uninstrumented_ms'] == 0
    assert profiler._capacity == 2
    profiler.close()


def test_exception_restores_depth_and_repeated_calls_accumulate():
    def fail():
        raise ValueError('sentinel')
    target=SimpleNamespace(call=fail)
    profiler=WindowProfiler(initial_capacity=1)
    profiler.wrap(target,'call','failure')
    with patch('tools.window_latency_breakdown.time.perf_counter',side_effect=[0.,1.,2.,3.]):
        profiler._begin()
        for _ in range(2):
            with pytest.raises(ValueError,match='sentinel'):
                target.call()
        assert profiler._depth == 0
        profiler._finish(3000.)
    assert profiler.rows[0]['stages']['failure'] == [2000.,2000.]
    profiler.close()
    assert target.call is fail
