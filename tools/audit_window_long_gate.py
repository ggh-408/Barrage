"""Counterfactual audit of the current long gate, isolated from deployment."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from tools.train_targeted_dagger import install_runtime, CONTROLLER
    install_runtime()
    from barrage_rl.deployment import configure_image_controller
    from barrage_rl.live_screen import LiveVisualController
    from tools.benchmark_tracker_exact import exact
    torch.set_num_threads(10)
    controllers = []
    for _ in range(2):
        controller = LiveVisualController(str(ROOT/'best.pt'),
            device_name='cpu', experimental_controller=CONTROLLER)
        configure_image_controller(controller.agent, 'receding', search_workers=9)
        controllers.append(controller)
    guard = controllers[1].agent._receding_pixel_guard
    def unconditional_gate(objects, masks, globals_):
        return np.zeros((len(objects),9),bool), np.ones((len(objects),9),bool)
    # Leave short commitment checks, certificates, route ranking and all model
    # computations intact. This process-local audit changes diagnostic counters.
    guard.hazards = unconditional_gate
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:
        records = pickle.load(f)
    for index, (detection, kwargs, expected) in enumerate(records):
        actions = [c._act_detections(detection, **kwargs) for c in controllers]
        assert actions[0] == actions[1] == expected, (index, actions, expected)
        exact(controllers[0].agent._receding_pixel_guard._plans, guard._plans)
        if (index+1) % 1200 == 0:
            print(f'{index+1}/{len(records)} identical actions and plan memory', flush=True)
    before = controllers[0].agent._receding_pixel_guard.counters
    after = guard.counters
    report = dict(updates=len(records), all_actions_equal=True, all_plan_memory_bitwise_equal=True,
        counter_changes={key: [before[key],after[key]] for key in before if before[key]!=after[key]},
        production_files_modified=False,
        scope='Complete recorded image-detection sequence with the pinned v53 window controller; replaces the 16-step diagnostic gate with unconditional triggering only in this process. No live deployment, training, or success evaluation.')
    path = ROOT/'diagnostics/window_current_audit_20260925/long_gate_counterfactual.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
