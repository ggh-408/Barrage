"""Exact ranking inputs for combined passes and retained certificate reuse."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def test_window_metrics_match_recorded_controller_for_varied_paths():
    from barrage_rl.window_planner import WindowPixelGuard
    from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig
    from barrage_rl import window_planner_kernel
    from numba import set_num_threads
    set_num_threads(9)
    root = Path(__file__).resolve().parents[1]
    directory = root/'diagnostics/planner_readiness_20260921'
    spec = importlib.util.spec_from_file_location('metric_reference_controller', directory/'candidate_parallel.py')
    reference = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(reference)
        calibration = PixelGuard(PixelGuardConfig())
        guard = SimpleNamespace(table=calibration.table, integral=calibration.integral)
        rng = np.random.default_rng(572)
        for case in range(24):
            paths = rng.integers(0, 9, (27, 15), dtype=np.int64)
            lengths = rng.integers(1, 16, 27, dtype=np.int64)
            plane = rng.uniform(20, 800, 2).astype(np.float32)
            bullets = plane + rng.uniform(-100, 100, (case*3, 2)).astype(np.float32)
            velocity = rng.uniform(-240, 240, bullets.shape).astype(np.float32)
            error = rng.uniform(2, 240, len(bullets)).astype(np.float32)
            args = (guard, plane, calibration.half_size, bullets, velocity, error, paths, lengths)
            expected = reference.ContinuationMixin._assess(*args)
            actual = WindowPixelGuard._assess(*args)
            certificate = WindowPixelGuard._certificate(*args)
            reused = WindowPixelGuard._assess(*args, certificate=certificate)
            assert expected.tobytes() == actual.tobytes() == reused.tobytes()
            assert expected[:, 4:6].tobytes() == certificate[:, 4:6].tobytes()
    finally:
        sys.path.remove(str(directory))
