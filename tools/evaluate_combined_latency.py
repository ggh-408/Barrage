"""Existing fixed-pool evaluation with exact checks of all latency candidates."""
import copy
from dataclasses import fields
import functools
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import types
import weakref
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_velocity_batch as runner
from tools.benchmark_tracker_exact import exact
from tools.probe_shared_predictions import shared_hints, build_shared_update

_velocity_worker = runner.checked_worker


def checked_worker(*args, **kwargs):
    from barrage_rl.tracked_collection import _RenderedRGBObservation
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    from barrage_rl.image_oracle import PersistentImageTracker
    from barrage_rl.runtime_core import tracker_prediction_hints
    pending = weakref.WeakKeyDictionary()
    checks = dict(pid=os.getpid(), hint_calls=0, reused_predictions=0, all_exact=True)
    original_frame = _RenderedRGBObservation._frame
    original_step = TrackedFeatureExtractor.step_detections
    shared_step = build_shared_update()
    # Preserve the exact current association expression as the independent reference.
    source = textwrap.dedent(inspect.getsource(PersistentImageTracker._update_measurements))
    start = source.index('    positions = np.asarray([track.position for track in self.tracks])')
    end = source.index('    occlusion_distance_squared =', start)
    body = source[start:end]
    namespace = dict(PersistentImageTracker._update_measurements.__globals__)
    exec('def reference(self, steps):\n    elapsed=self.decision_dt*max(1,int(steps))\n' + body +
         '    return predictions\n', namespace)
    reference = namespace['reference']
    def hints(tracker, shape, decision_steps=1):
        old = tracker_prediction_hints(tracker, shape, decision_steps)
        new, positions = shared_hints(tracker, shape, decision_steps)
        exact(old, new)
        if tracker.tracks:
            exact(reference(tracker, decision_steps), positions)
        pending[tracker] = (tracker._step, decision_steps, positions)
        checks['hint_calls'] += 1
        return new
    namespace = dict(original_frame.__globals__, tracker_prediction_hints=hints)
    exec(compile(textwrap.dedent(inspect.getsource(original_frame)), '<paired_rgb_frame>', 'exec'), namespace)
    _RenderedRGBObservation._frame = namespace['_frame']
    def step(extractor, bullets, plane, *, decision_steps=1):
        value = pending.pop(extractor.tracker, None)
        if value is None:
            return original_step(extractor, bullets, plane, decision_steps=decision_steps)
        timestamp, steps, positions = value
        assert timestamp == extractor.tracker._step and steps == decision_steps
        checks['reused_predictions'] += 1
        return shared_step(extractor, bullets, plane, positions, decision_steps=decision_steps)
    TrackedFeatureExtractor.step_detections = step
    try:
        _velocity_worker(*args, **kwargs)
    except BaseException:
        checks['all_exact'] = False
        raise
    finally:
        TrackedFeatureExtractor.step_detections = original_step
        _RenderedRGBObservation._frame = original_frame
        (Path(os.environ['BARRAGE_VELOCITY_AUDIT_DIR']) / f'prediction_{os.getpid()}.json').write_text(
            json.dumps(checks, indent=2), encoding='utf-8')


def main():
    from tools import train_targeted_dagger as entry
    from barrage_rl import deployment
    from tools.benchmark_shared_path_replay import configure as configure_paths
    original_install = entry.install_runtime
    planner = dict(decisions=0, all_actions_and_state_exact=True)
    def install_runtime():
        original_install()
        original_configure = deployment.configure_image_controller
        @functools.wraps(original_configure)
        def configure(agent, *args, **kwargs):
            result = original_configure(agent, *args, **kwargs)
            guard = agent._receding_pixel_guard
            original_apply, original_assess = guard.apply, guard._assess
            configure_paths([None, types.SimpleNamespace(agent=agent)])
            optimized_apply, optimized_assess = guard.apply, guard._assess
            def apply(selection, objects, masks, globals_):
                before = copy.deepcopy(guard._plans), guard.counters.copy(), guard.elapsed_seconds
                guard._assess = original_assess
                expected = original_apply(selection, objects, masks, globals_)
                expected_plans, expected_counters = copy.deepcopy(guard._plans), guard.counters.copy()
                guard._plans, guard.counters, guard.elapsed_seconds = before
                guard._assess = optimized_assess
                actual = optimized_apply(selection, objects, masks, globals_)
                for field in fields(expected):
                    a, b = getattr(expected, field.name), getattr(actual, field.name)
                    if isinstance(a, torch.Tensor):
                        exact(a.detach().cpu().numpy(), b.detach().cpu().numpy())
                    else:
                        exact(a, b)
                exact(expected_plans, guard._plans)
                exact(expected_counters, guard.counters)
                planner['decisions'] += len(objects)
                return actual
            guard.apply = apply
            return result
        deployment.configure_image_controller = configure
    entry.install_runtime = install_runtime
    runner.checked_worker = checked_worker
    runner.CANDIDATE_NAME = 'normalization, shared predictions, shared path traversal, retained-route reuse'
    runner.ADDITIONAL_SOURCES = ('tools/evaluate_combined_latency.py', 'tools/probe_shared_predictions.py',
        'tools/probe_shared_path_assessment.py', 'tools/reuse_retained_assessment.py',
        'tools/benchmark_shared_path_replay.py')
    runner.main()
    audit = Path(os.environ['BARRAGE_VELOCITY_AUDIT_DIR'])
    checks = [json.loads(path.read_text()) for path in audit.glob('prediction_*.json')]
    assert checks and all(c['all_exact'] for c in checks)
    assert sum(c['reused_predictions'] for c in checks) > 0
    (audit.parent / 'combined_checks.json').write_text(json.dumps(dict(planner=planner,
        hint_calls=sum(c['hint_calls'] for c in checks),
        reused_predictions=sum(c['reused_predictions'] for c in checks),
        all_prediction_checks_exact=True), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
