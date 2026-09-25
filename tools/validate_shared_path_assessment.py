"""Replay real detections while checking every shared path metric exactly."""
import json
import argparse
import copy
from pathlib import Path
import sys
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_tracker_exact as validation
from tools.probe_shared_path_assessment import assess_interval_exposure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-retained', action='store_true')
    args = parser.parse_args()
    output = ROOT / ('diagnostics/shared_path_reuse_20260925' if args.reuse_retained
                     else 'diagnostics/shared_path_20260925')
    output.mkdir(exist_ok=True)
    if (output / 'paired_replay.json').exists():
        raise FileExistsError(output / 'paired_replay.json')
    validation.BASE = output
    checks = dict(calls=0, routes=0, all_metrics_bitwise_equal=True)
    traces = [[], []]
    def configure(controllers):
        from readiness_kernel_parallel import assess_paths
        from barrage_rl.runtime_core import ACTION_VECTORS
        guard = controllers[1].agent._receding_pixel_guard
        for index, controller in enumerate(controllers):
            controller.agent._receding_pixel_guard._decision_observer = (
                lambda key, row, index=index: traces[index].append(copy.deepcopy((key, row))))
        if args.reuse_retained:
            from tools.reuse_retained_assessment import install
            install(guard)
        original = guard._assess
        def assess(self, plane, half_size, bullets, velocity, error, paths, lengths):
            scenarios = assess_paths(plane, half_size, bullets, velocity, error,
                                     self.table, ACTION_VECTORS, paths, lengths)
            intervals, exposure = assess_interval_exposure(plane, half_size, bullets,
                velocity, error, self.table, self.integral, ACTION_VECTORS, paths, lengths)
            result = np.column_stack((scenarios, intervals, exposure))
            expected = original(plane, half_size, bullets, velocity, error, paths, lengths)
            validation.exact(expected, result)
            checks['calls'] += 1
            checks['routes'] += len(paths)
            return result
        guard._assess = types.MethodType(assess, guard)
    validation.replay([], ROOT / 'diagnostics/tracker_exact_20260925/after/image_detections.pkl',
                      configure, verify_planner_state=True)
    validation.exact(traces[0], traces[1])
    checks['decision_traces_bitwise_equal'] = True
    checks['decisions_traced'] = len(traces[0])
    (output / 'metric_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
