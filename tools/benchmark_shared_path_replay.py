"""Measure paired decision latency of the exact path-reuse candidates."""
from pathlib import Path
import sys
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_tracker_exact as validation
from tools.probe_shared_path_assessment import assess_interval_exposure
from tools.reuse_retained_assessment import install


def configure(controllers):
    from readiness_kernel_parallel import assess_paths
    from barrage_rl.runtime_core import ACTION_VECTORS
    guard = controllers[1].agent._receding_pixel_guard
    install(guard)
    def assess(self, plane, half_size, bullets, velocity, error, paths, lengths):
        scenarios = assess_paths(plane, half_size, bullets, velocity, error,
                                 self.table, ACTION_VECTORS, paths, lengths)
        intervals, exposure = assess_interval_exposure(plane, half_size, bullets,
            velocity, error, self.table, self.integral, ACTION_VECTORS, paths, lengths)
        return np.column_stack((scenarios, intervals, exposure))
    guard._assess = types.MethodType(assess, guard)
    # Compilation precedes the decision timing; no state or counters are touched.
    guard._assess(np.array([410., 410.], np.float32), guard.half_size,
        np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), np.ones(1, np.float32),
        np.zeros((1, 15), np.int64), np.array([15], np.int64))


def main():
    output = ROOT / 'diagnostics/shared_path_timing_20260925'
    output.mkdir(exist_ok=True)
    if (output / 'paired_replay.json').exists():
        raise FileExistsError(output / 'paired_replay.json')
    validation.BASE = output
    validation.replay([], ROOT / 'diagnostics/tracker_exact_20260925/after/image_detections.pkl',
                      configure, verify_planner_state=True)


if __name__ == '__main__':
    main()
