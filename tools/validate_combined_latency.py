"""Paired replay of exact normalization, path reuse and prediction handoff."""
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_tracker_exact as validation
from tools.benchmark_shared_path_replay import configure as configure_paths
from tools.probe_shared_predictions import shared_hints, build_shared_update
from tools.validate_velocity_batch import candidate as velocity_candidate
from barrage_rl.runtime_core import tracker_prediction_hints
from barrage_rl.image_oracle import PersistentImageTracker
from barrage_rl.live_screen import LiveVisualController


def shared_act():
    source = textwrap.dedent(inspect.getsource(LiveVisualController._act_detections))
    source = source.replace('*, decision_steps: int = 1',
                            '*, decision_steps: int = 1, source_predictions=None', 1)
    old = '''self.tracked_extractor.step_detections(
            detections.bullet_positions,
            detections.plane_position,
            decision_steps=decision_steps,
        )'''
    new = '''_shared_step(self.tracked_extractor,
            detections.bullet_positions, detections.plane_position,
            source_predictions, decision_steps=decision_steps)'''
    assert source.count(old) == 1, 'Controller source differs from reviewed version'
    namespace = dict(LiveVisualController._act_detections.__globals__,
                     _shared_step=build_shared_update())
    exec(compile(source.replace(old, new), '<shared_prediction_controller>', 'exec'), namespace)
    return namespace['_act_detections']


def main():
    output = ROOT / 'diagnostics/combined_latency_20260925'
    output.mkdir(exist_ok=True)
    if (output / 'paired_replay.json').exists():
        raise FileExistsError(output / 'paired_replay.json')
    validation.BASE = output
    hints = [[], []]
    optimized_act = shared_act()
    def configure(controllers):
        configure_paths(controllers)
        old = controllers[0]._act_detections
        def reference(controller, detections, **kwargs):
            value = tracker_prediction_hints(controller.tracked_extractor.tracker,
                                             (820, 820, 3), kwargs.get('decision_steps', 1))
            hints[0].append(value)
            return old(detections, **kwargs)
        def optimized(controller, detections, **kwargs):
            value, source = shared_hints(controller.tracked_extractor.tracker,
                                        (820, 820, 3), kwargs.get('decision_steps', 1))
            hints[1].append(value)
            return optimized_act(controller, detections, source_predictions=source, **kwargs)
        controllers[0]._act_detections = types.MethodType(reference, controllers[0])
        controllers[1]._act_detections = types.MethodType(optimized, controllers[1])
    original, modified = velocity_candidate()
    items = [(PersistentImageTracker, '_fit_velocities', original, modified)]
    try:
        validation.replay(items, ROOT / 'diagnostics/tracker_exact_20260925/after/image_detections.pkl',
                          configure, verify_planner_state=True)
        validation.exact(hints[0], hints[1])
        (output / 'hint_checks.json').write_text(json.dumps(dict(
            updates=len(hints[0]), all_pre_detection_hints_bitwise_equal=True,
            scope='Hints computed from prior state, then handed directly to tracker association; RGB detection itself is outside replay timing.'
        ), indent=2), encoding='utf-8')
    finally:
        PersistentImageTracker._fit_velocities = original


if __name__ == '__main__':
    main()
