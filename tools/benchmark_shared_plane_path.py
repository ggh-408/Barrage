"""Interleaved warmed kernel timings; run without competing evaluation workloads."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_shared_plane_path import candidate, assess_paths
from numba import set_num_threads


def main():
    from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig
    from barrage_rl.runtime_core import ACTION_VECTORS
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 3, 6, 9])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.repeats < 2 or any(n < 1 for n in options.workers):
        parser.error('Positive workers and at least two repeats are required')
    if options.output.exists():
        raise FileExistsError(options.output)
    guard = PixelGuard(PixelGuardConfig())
    optimized = candidate()
    rng = np.random.default_rng(92562)
    pool = []
    for routes in (1, 9, 18, 27):
        for crowded in (False, True):
            plane = np.array([410, 410], dtype=np.float32)
            bullets = rng.uniform(0, 820, (300, 2)).astype(np.float32)
            if crowded:
                bullets[:20] = plane + rng.uniform(-25, 25, (20, 2))
            velocity = rng.uniform(-240, 240, (300, 2)).astype(np.float32)
            error = rng.uniform(2, 240, 300).astype(np.float32)
            paths = rng.integers(0, 9, (routes, 15), dtype=np.int64)
            lengths = np.full(routes, 15, dtype=np.int64)
            pool.append((crowded, (plane, guard.half_size, bullets, velocity,
                         error, guard.table, ACTION_VECTORS, paths, lengths)))
    results = []
    for workers in options.workers:
        set_num_threads(workers)
        for crowded, args in pool:
            expected, actual = assess_paths(*args), optimized(*args)
            assert expected.dtype == actual.dtype and expected.shape == actual.shape
            assert expected.tobytes() == actual.tobytes()
            timings = [[], []]
            for repeat in range(options.repeats):
                for variant in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    function = (assess_paths, optimized)[variant]
                    start = time.perf_counter_ns()
                    for _ in range(40):
                        function(*args)
                    timings[variant].append((time.perf_counter_ns() - start) / 40e6)
            before, after = (float(np.median(values)) for values in timings)
            results.append(dict(workers=workers, routes=len(args[-1]), crowded=crowded,
                                before_median_ms=before, after_median_ms=after,
                                change_percent=100 * (after / before - 1), samples_ms=timings))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(dict(results=results, bullets=300, depth=15,
        metrics_bitwise_equal=True, scope='Synthetic kernel timings; end-to-end confirmation required'),
        indent=2), encoding='utf-8')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
