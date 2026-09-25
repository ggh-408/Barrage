"""Replay the isolated shared-plane candidate with exact per-call metrics checks."""
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_tracker_exact as validation
from tools.probe_shared_plane_path import candidate, assess_paths


def main():
    output = ROOT / 'diagnostics/shared_plane_path_20260925/replay'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'paired_replay.json').exists():
        raise FileExistsError(output / 'paired_replay.json')
    validation.BASE = output
    optimized = candidate()
    counts = dict(calls=0, route_rows=0, all_metrics_bitwise_equal=True)

    def configure(controllers):
        from readiness_kernel_parallel import assess_path_intervals, assess_path_exposure
        from barrage_rl.runtime_core import ACTION_VECTORS
        guard = controllers[1].agent._receding_pixel_guard

        def assess(self, plane, half_size, bullets, velocity, error, paths, lengths):
            args = (plane, half_size, bullets, velocity, error, self.table,
                    ACTION_VECTORS, paths, lengths)
            expected = assess_paths(*args)
            actual = optimized(*args)
            validation.exact(expected, actual)
            counts['calls'] += 1
            counts['route_rows'] += len(paths)
            interval_args = (*args[:6], self.integral, *args[6:])
            return np.column_stack((actual, assess_path_intervals(*interval_args),
                                    assess_path_exposure(*interval_args)))

        guard._assess = types.MethodType(assess, guard)

    validation.replay([], ROOT / 'diagnostics/tracker_exact_20260925/after/image_detections.pkl',
                      configure, verify_planner_state=True)
    assert counts['calls'] > 0
    counts['timing_valid'] = False
    counts['timing_limitation'] = 'Reference metrics execute inside candidate calls for exact verification.'
    (output / 'metric_checks.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
