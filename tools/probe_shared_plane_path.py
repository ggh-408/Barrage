"""Isolated scenario assessment with one plane trajectory per candidate route."""
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diagnostics/planner_readiness_20260921'))
from readiness_kernel_parallel import assess_paths
from numba import set_num_threads


def candidate():
    source = textwrap.dedent(inspect.getsource(assess_paths.py_func))
    source = source.replace('        terminal=plane.copy()',
        '        positions=np.empty((lengths[i]*4,2),plane.dtype)\n        terminal=plane.copy()')
    marker = '        metrics[i,4]=min('
    assert source.count(marker) == 1
    source = source.replace(marker, '            positions[step]=terminal\n' + marker)
    begin = source.index('                a=paths[i,step//4]', source.index('for scenario'))
    end = source.index('                t=np.float32', begin)
    source = source[:begin] + '                p=positions[step]\n' + source[end:]
    namespace = dict(assess_paths.py_func.__globals__)
    exec(compile(source, '<shared_plane_path>', 'exec'), namespace)
    return namespace['assess_paths']


def main():
    from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig
    from barrage_rl.runtime_core import ACTION_VECTORS
    set_num_threads(1)
    guard = PixelGuard(PixelGuardConfig())
    optimized = candidate()
    rng = np.random.default_rng(92561)
    rows = 0
    for case in range(240):
        count = (0, 1, 20, 100, 300)[case % 5]
        depth = (1, 4, 15)[case % 3]
        routes = (1, 9, 18, 27)[case % 4]
        plane = rng.uniform(9, 811, 2).astype(np.float32)
        if case >= 120:
            plane = plane.astype(np.float64)
        if case % 7 == 0:
            plane[:] = [9, 811]
        bullets = rng.uniform(0, 820, (count, 2)).astype(np.float32)
        if count and case % 2 == 0:
            bullets[:min(count, 20)] = plane + rng.uniform(-25, 25, (min(count, 20), 2))
        velocity = rng.uniform(-240, 240, (count, 2)).astype(np.float32)
        error = rng.uniform(2, 240, count).astype(np.float32)
        paths = rng.integers(0, 9, (routes, depth), dtype=np.int64)
        lengths = rng.integers(0, depth + 1, routes, dtype=np.int64)
        args = (plane, guard.half_size, bullets, velocity, error, guard.table,
                ACTION_VECTORS, paths, lengths)
        original_inputs = [value.tobytes() for value in args]
        expected, actual = assess_paths(*args), optimized(*args)
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert expected.tobytes() == actual.tobytes(), case
        assert original_inputs == [value.tobytes() for value in args]
        rows += routes
    result = dict(cases=240, route_rows=rows, plane_dtypes=['float32', 'float64'], metrics_bitwise_equal=True,
                  inputs_unchanged=True, deployment_modified=False,
                  timing_pending=True, full_replay_pending=True)
    output = ROOT / 'diagnostics/shared_plane_path_20260925'
    output.mkdir(exist_ok=True)
    (output / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
