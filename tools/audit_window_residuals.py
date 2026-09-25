"""Observe residual work in the pinned window runtime, without deployment edits."""
import collections
import functools
import inspect
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    torch.set_num_threads(10)
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.window_runtime import CONTROLLER, configure_window_controller
    from tools.pixel_guard_candidate import PixelGuard
    counts = collections.Counter()
    seconds = collections.Counter()
    events = collections.Counter()
    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                counts[label] += 1
                seconds[label] += time.perf_counter() - start
        setattr(owner, name, call)
    wrap(torch, 'load', 'checkpoint_load')
    wrap(PixelGuard, '__init__', 'calibration_init')
    controller = LiveVisualController(str(ROOT/'best.pt'),
        experimental_controller=CONTROLLER)
    configure_window_controller(controller.agent)
    guard = controller.agent._receding_pixel_guard
    wrap(guard.commit_guard, 'hazards', 'short_hazards')
    wrap(guard, 'hazards', 'long_hazards')
    wrap(controller.agent.model.teacher_cost_head, 'forward', 'teacher_cost_head')
    wrap(controller.agent._action_selector, 'select', 'selector')
    wrap(guard, '_certificate', 'certificate')
    wrap(guard, '_assess', 'assessment')
    original_rectangle = guard.commit_guard.rectangle_hits
    def rectangle(lower, upper):
        caller = inspect.currentframe().f_back
        events[f'rectangle:{Path(caller.f_code.co_filename).name}:{caller.f_lineno}'] += 1
        return original_rectangle(lower, upper)
    guard.commit_guard.rectangle_hits = rectangle
    original_trace = guard._trace_decision
    def trace(key, reason, incumbent, action, **details):
        events['branch:'+reason] += 1
        return original_trace(key, reason, incumbent, action, **details)
    guard._trace_decision = trace
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as stream:
        records = pickle.load(stream)[:600]
    for detection, kwargs, expected in records:
        assert controller._act_detections(detection, **kwargs) == expected
    extra_modules = [name for name in ('tools.train_targeted_dagger', 'barrage_rl.train_tracked_policy',
        'barrage_rl.parallel_evaluation', 'barrage_rl.distilled_student', 'readiness_kernel_parallel',
        'tools.pixel_receding_kernel') if name in sys.modules]
    result = dict(updates=len(records), all_recorded_actions_equal=True,
        configuration=dict(interval_consensus=guard.recovery_config.interval_consensus,
            preserve_plans=guard.recovery_config.preserve_plans, teacher_cost_weight=controller.agent.teacher_cost_ranking_weight,
            analytic_shield=controller.agent.analytic_shield, observer_enabled=getattr(guard,'_decision_observer',None) is not None),
        calls=dict(counts), total_seconds=dict(seconds), events=dict(events),
        inactive_modules_loaded=extra_modules,
        active_legacy_bases=[f'{c.__module__}.{c.__name__}' for c in type(guard).__mro__],
        calibration_tables_equal=bool((guard.table==guard.commit_guard.table).all()),
        calibration_tables_distinct=guard.table is not guard.commit_guard.table,
        max_track_history=max((len(t.history) for t in controller.tracked_extractor.tracker.tracks), default=0),
        stored_timing_samples=dict(tracking=len(controller._stage_tracking_ms), model=len(controller._stage_model_ms)),
        counters=guard.counters,
        scope='600 existing image-detection records on the current pinned window controller. No RGB decoding, live window, training, or success evaluation. Timing includes audit probes.')
    path=ROOT/'diagnostics/window_cleanup_20260925/residual_audit.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
