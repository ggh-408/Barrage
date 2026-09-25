"""Measure fresh-process initialization of the pinned window runtime."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    torch.set_num_threads(10)
    calls = []
    original = torch.load
    def load(*args, **kwargs):
        calls.append(str(args[0]))
        return original(*args, **kwargs)
    torch.load = load
    started = time.perf_counter()
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.window_runtime import CONTROLLER, configure_window_controller
    imported = time.perf_counter()
    from barrage_rl.window_pixel_geometry import WindowPixelGeometry
    calibration_calls = []
    initialize = WindowPixelGeometry.__init__
    def calibration(self):
        calibration_calls.append(1)
        return initialize(self)
    WindowPixelGeometry.__init__ = calibration
    controller = LiveVisualController(str(ROOT/'best.pt'),
        experimental_controller=CONTROLLER)
    loaded = time.perf_counter()
    configure_window_controller(controller.agent)
    finished = time.perf_counter()
    extra = [name for name in ('tools.train_targeted_dagger', 'barrage_rl.train_tracked_policy',
        'barrage_rl.parallel_evaluation', 'barrage_rl.distilled_student', 'readiness_kernel_parallel',
        'tools.pixel_receding_kernel', 'tools.pixel_guard_receding', 'gymnasium',
        'barrage_rl.env', 'barrage_rl.baselines', 'barrage_rl.recovery_planner') if name in sys.modules]
    assert len(calls) == 1, calls
    assert not extra, extra
    assert len(calibration_calls) == 1
    import pickle
    observed_calls = dict(teacher_cost_head=0, action_selector=0)
    for owner, method, key in ((controller.agent.model.teacher_cost_head, 'forward', 'teacher_cost_head'),
                               (controller.agent._action_selector, 'select', 'action_selector')):
        original_method = getattr(owner, method)
        def observe(*args, _original=original_method, _key=key, **kwargs):
            observed_calls[_key] += 1
            return _original(*args, **kwargs)
        setattr(owner, method, observe)
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as stream:
        records = pickle.load(stream)[:120]
    for detection, kwargs, expected in records:
        assert controller._act_detections(detection, **kwargs) == expected
    assert observed_calls == dict(teacher_cost_head=0, action_selector=0)
    assert not controller._stage_tracking_ms and not controller._stage_model_ms
    assert max(len(t.history) for t in controller.tracked_extractor.tracker.tracks) <= 8
    result = dict(checkpoint_load_count=len(calls), checkpoint=calls[0],
        calibration_count=len(calibration_calls),
        runtime_import_seconds=imported-started, controller_seconds=loaded-imported,
        guard_seconds=finished-loaded, total_after_torch_import_seconds=finished-started,
        inactive_modules_loaded=extra, replay_actions_equal=len(records), inactive_hot_calls=observed_calls,
        normal_mode_timing_samples=0, history_limit=8,
        controller_manifest=controller.agent._receding_pixel_guard.manifest())
    path = ROOT/'diagnostics/window_residual_cleanup_20260925/startup_after.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
