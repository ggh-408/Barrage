"""Isolated exact candidate sharing interval and exposure path traversal."""
from pathlib import Path
import sys
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diagnostics/planner_readiness_20260921'))
from tools.pixel_search_kernel import njit, search_step
from numba import prange, set_num_threads


@njit(cache=False, parallel=True)
def assess_interval_exposure(plane, half_size, bullets, velocity, error,
                            table, integral, vectors, paths, lengths):
    prefix = np.zeros(len(paths), np.float64)
    exposure = np.zeros(len(paths), np.float64)
    for i in prange(len(paths)):
        p = plane.reshape(1, 2).copy()
        survived = 0
        failed = False
        for step in range(lengths[i] * 4):
            t = np.float32((step + 1) / 120)
            movement = (vectors[paths[i, step // 4]] * np.float32(2)).reshape(1, 2)
            p, hit, mass = search_step(p, movement, half_size, bullets + velocity * t,
                np.float32(1) + error * t, table, integral, True)
            if not failed:
                if np.any(mass > 0):
                    failed = True
                else:
                    survived += 1
            exposure[i] += np.sum(mass)
        prefix[i] = survived / max(lengths[i] * 4, 1)
    return prefix, exposure


def main():
    from readiness_kernel_parallel import assess_path_intervals, assess_path_exposure
    from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig
    from barrage_rl.runtime_core import ACTION_VECTORS
    set_num_threads(9)
    guard = PixelGuard(PixelGuardConfig())
    rng = np.random.default_rng(92531)
    pool = []
    for case in range(120):
        count = (0, 1, 20, 100, 300)[case % 5]
        depth = (1, 4, 15)[case % 3]
        routes = (1, 9, 18, 27)[case % 4]
        plane = rng.uniform(9, 811, 2).astype(np.float32)
        if case % 7 == 0:
            plane[:] = [9, 811]
        bullets = rng.uniform(0, 820, (count, 2)).astype(np.float32)
        if count and case % 2 == 0:
            bullets[:min(count, 20)] = plane + rng.uniform(-25, 25, (min(count, 20), 2))
        velocity = rng.uniform(-240, 240, (count, 2)).astype(np.float32)
        error = rng.uniform(2, 240, count).astype(np.float32)
        paths = rng.integers(0, 9, (routes, depth), dtype=np.int64)
        lengths = rng.integers(0, depth + 1, routes, dtype=np.int64)
        if case % 3 == 0:
            lengths[:] = depth
        pool.append((plane, guard.half_size, bullets, velocity, error,
                     guard.table, guard.integral, ACTION_VECTORS, paths, lengths))
    start = time.perf_counter()
    for args in pool:
        old = (assess_path_intervals(*args), assess_path_exposure(*args))
        new = assess_interval_exposure(*args)
        for a, b in zip(old, new):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    validation_seconds = time.perf_counter() - start
    dense = [a for a in pool if len(a[2]) == 300 and a[-2].shape[1] == 15]
    timings = [[], []]
    for repeat in range(8):
        for variant in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            start = time.perf_counter()
            for _ in range(20):
                for args in dense:
                    if variant:
                        assess_interval_exposure(*args)
                    else:
                        assess_path_intervals(*args)
                        assess_path_exposure(*args)
            timings[variant].append((time.perf_counter() - start) * 1000 / (20 * len(dense)))
    result = dict(cases=len(pool), route_rows=sum(len(a[-1]) for a in pool),
        interval_and_exposure_bitwise_equal=True, validation_and_compile_seconds=validation_seconds,
        original_median_ms=float(np.median(timings[0])), shared_median_ms=float(np.median(timings[1])),
        samples_ms=timings, workers=9, benchmark_bullets=300, benchmark_depth=15,
        deployment_modified=False, scope='Kernel-only synthetic test; no end-to-end claim')
    output = ROOT / 'diagnostics/velocity_batch_20260925/shared_path_probe.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
