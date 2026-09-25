"""Foreground before/after timing of the isolated combined exact candidates."""
import functools
import inspect
from pathlib import Path
import runpy
import sys
import textwrap
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_candidate():
    from tools.train_targeted_dagger import install_runtime
    from barrage_rl import deployment
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.image_oracle import PersistentImageTracker
    from tools.validate_velocity_batch import candidate
    from tools.validate_combined_latency import shared_act
    from tools.probe_shared_predictions import shared_hints
    from tools.benchmark_shared_path_replay import configure as configure_paths
    install_runtime()
    original_configure = deployment.configure_image_controller
    @functools.wraps(original_configure)
    def configure(agent, *args, **kwargs):
        result = original_configure(agent, *args, **kwargs)
        configure_paths([None, SimpleNamespace(agent=agent)])
        return result
    source = textwrap.dedent(inspect.getsource(LiveVisualController._observe_due_rgb))
    old = 'predictions, radii = self._prediction_hints(rgb, decision_steps)'
    assert source.count(old) == 1
    source = source.replace(old, '(predictions, radii), source_predictions = _shared_hints(\n'
        '        self.tracked_extractor.tracker, tuple(rgb.shape), decision_steps)')
    old = 'self._act_detections(\n        detections, decision_steps=decision_steps\n    )'
    assert source.count(old) == 1
    source = source.replace(old, '_shared_act(self, detections, decision_steps=decision_steps,\n'
        '        source_predictions=source_predictions)')
    namespace = dict(LiveVisualController._observe_due_rgb.__globals__,
                     _shared_hints=shared_hints, _shared_act=shared_act())
    exec(compile(source, '<shared_prediction_observation>', 'exec'), namespace)
    original_observe = LiveVisualController._observe_due_rgb
    original_fit, modified_fit = candidate()
    deployment.configure_image_controller = configure
    LiveVisualController._observe_due_rgb = namespace['_observe_due_rgb']
    PersistentImageTracker._fit_velocities = modified_fit
    def restore():
        deployment.configure_image_controller = original_configure
        LiveVisualController._observe_due_rgb = original_observe
        PersistentImageTracker._fit_velocities = original_fit
    return restore


def main():
    output = ROOT / 'diagnostics/combined_window_20260925'
    for name in ('before', 'after'):
        if (output / name).exists():
            raise FileExistsError(output / name)
    previous = sys.argv
    restore = None
    try:
        for name in ('before', 'after'):
            if name == 'after':
                restore = install_candidate()
            sys.argv = [str(ROOT / 'tools/diagnose_window_pacing.py'), '--seconds', '120',
                        '--record-window-state', '--output-dir', str(output / name)]
            runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = previous
        if restore:
            restore()


if __name__ == '__main__':
    main()
