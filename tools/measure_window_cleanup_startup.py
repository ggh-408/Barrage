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
    controller = LiveVisualController(str(ROOT/'best.pt'),
        experimental_controller=CONTROLLER)
    loaded = time.perf_counter()
    configure_window_controller(controller.agent)
    finished = time.perf_counter()
    extra = [name for name in ('tools.train_targeted_dagger', 'barrage_rl.train_tracked_policy',
        'barrage_rl.parallel_evaluation', 'barrage_rl.distilled_student', 'readiness_kernel_parallel',
        'tools.pixel_receding_kernel') if name in sys.modules]
    assert len(calls) == 1, calls
    assert not extra, extra
    result = dict(checkpoint_load_count=len(calls), checkpoint=calls[0],
        runtime_import_seconds=imported-started, controller_seconds=loaded-imported,
        guard_seconds=finished-loaded, total_after_torch_import_seconds=finished-started,
        inactive_modules_loaded=extra, controller_manifest=controller.agent._receding_pixel_guard.manifest())
    path = ROOT/'diagnostics/window_cleanup_20260925/startup_after.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
