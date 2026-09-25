"""Observe the pinned window path without changing production code or training."""
import functools
import inspect
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'diagnostics/window_current_audit_20260925'


def main():
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    import torch
    torch_import_seconds = time.perf_counter()-started
    spans = {}
    phase = 'startup'
    restores = []
    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                key = phase+':'+label
                row = spans.setdefault(key, dict(calls=0, seconds=0.0))
                row['calls'] += 1
                row['seconds'] += time.perf_counter()-start
        setattr(owner, name, call)
        restores.append((owner, name, original))

    checkpoints = []
    original_load = torch.load
    def load(*args, **kwargs):
        start = time.perf_counter()
        checkpoint = original_load(*args, **kwargs)
        checkpoints.append(dict(path=str(args[0]), seconds=time.perf_counter()-start,
            model_version=checkpoint.get('model_version'),
            spec=checkpoint.get('tracked_policy_spec'), hparams=checkpoint.get('model_hparams'),
            distilled_student_present='distilled_student' in checkpoint,
            experimental_controller=checkpoint.get('experimental_controller')))
        return checkpoint
    torch.load = load
    wrap(torch.nn.init, 'orthogonal_', 'orthogonal_initialization')
    start = time.perf_counter()
    from tools.train_targeted_dagger import install_runtime, CONTROLLER
    training_entry_import_seconds = time.perf_counter()-start
    start = time.perf_counter()
    install_runtime()
    runtime_install_seconds = time.perf_counter()-start
    from barrage_rl.live_screen import LiveVisualController
    from tools.pixel_guard_candidate import PixelGuard
    from tools.pixel_guard_receding import _StatelessPixelGuard
    from tools.pixel_guard_continuation import ContinuationMixin
    for cls in (PixelGuard, _StatelessPixelGuard, ContinuationMixin):
        wrap(cls, '__init__', cls.__name__+'.__init__')
    torch.set_num_threads(min(10, __import__('os').cpu_count() or 1))
    start = time.perf_counter()
    controller = LiveVisualController(str(ROOT/'best.pt'),
        device_name='cpu', experimental_controller=CONTROLLER)
    controller_init_seconds = time.perf_counter()-start
    from barrage_rl.deployment import configure_image_controller
    start = time.perf_counter()
    configure_image_controller(controller.agent, 'receding', search_workers=9)
    guard_init_seconds = time.perf_counter()-start
    agent = controller.agent
    guard = agent._receding_pixel_guard
    from tools import pixel_receding_kernel as old_kernel
    import readiness_kernel_parallel as current_kernel
    signatures = {
        'old': {name: len(getattr(old_kernel, name).signatures)
                for name in ('beam_costs', 'beam_paths', 'assess_paths', 'assess_path_intervals')},
        'current': {name: len(getattr(current_kernel, name).signatures)
                    for name in ('beam_paths', 'assess_paths', 'assess_path_intervals', 'assess_path_exposure')},
    }
    imported_extras = sorted(name for name in sys.modules if name in (
        'barrage_rl.train_tracked_policy', 'barrage_rl.tracked_collection',
        'barrage_rl.parallel_evaluation', 'barrage_rl.distilled_student',
        'barrage_rl.plot', 'matplotlib', 'tools.train_targeted_dagger'))
    configuration = dict(agent_type=type(agent).__name__, device=str(agent.device),
        action_delay_steps=controller.action_delay_steps,
        decision_interval=controller.decision_interval,
        analytic_shield=agent.analytic_shield,
        teacher_cost_ranking_weight=agent.teacher_cost_ranking_weight,
        action_hysteresis_bonus=agent.action_hysteresis_bonus,
        continuation_head_present=agent.model.continuation_head is not None,
        continuation_weight=agent.model.continuation_weight,
        guard_apply_source=inspect.getsourcefile(guard.apply),
        guard_assess_source=inspect.getsourcefile(guard._assess),
        interval_consensus=guard.recovery_config.interval_consensus,
        search_depth=guard.recovery_config.search_depth,
        duplicate_guard_tables_equal=bool((guard.table==guard.commit_guard.table).all()),
        duplicate_guard_tables_distinct=guard.table is not guard.commit_guard.table)
    for name in ('forward_with_geometry', 'prepare_action_geometry'):
        wrap(agent.model, name, 'model.'+name)
    wrap(agent.model.teacher_cost_head, 'forward', 'model.teacher_cost_head')
    wrap(agent._action_selector, 'select', 'selector.select')
    wrap(guard, 'hazards', 'guard.long_hazards')
    wrap(guard.commit_guard, 'hazards', 'guard.short_hazards')
    for name in ('beam_costs', 'beam_paths', 'assess_paths', 'assess_path_intervals'):
        # These old dispatchers are no longer referenced by the installed guard.
        wrap(old_kernel, name, 'old_kernel.'+name)
    for name in ('beam_paths', 'assess_paths', 'assess_path_intervals', 'assess_path_exposure'):
        wrap(current_kernel, name, 'current_kernel.'+name)
    assessment_calls = []
    old_assess = guard._assess
    def assess(*args, **kwargs):
        line = sys._getframe(1).f_lineno
        begin = time.perf_counter()
        metrics = old_assess(*args, **kwargs)
        assessment_calls.append(dict(caller_line=line, paths=len(args[-2]),
                                     seconds=time.perf_counter()-begin))
        return metrics
    guard._assess = assess
    reasons = {}
    def observe(_key, detail):
        reason = detail['reason']
        reasons[reason] = reasons.get(reason, 0)+1
    guard._decision_observer = observe
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:
        records = pickle.load(f)[:600]
    phase = 'decisions'
    actions_match = True
    start = time.perf_counter()
    for detections, kwargs, expected in records:
        action = controller._act_detections(detections, **kwargs)
        actions_match &= action == expected
    replay_seconds = time.perf_counter()-start
    for owner, name, original in reversed(restores):
        setattr(owner, name, original)
    torch.load = original_load
    report = dict(checkpoint_loads=checkpoints, configuration=configuration,
        torch_import_seconds=torch_import_seconds,
        training_entry_import_seconds=training_entry_import_seconds,
        runtime_install_seconds=runtime_install_seconds,
        controller_init_seconds=controller_init_seconds, guard_init_seconds=guard_init_seconds,
        startup_compiled_signatures=signatures, imported_extras=imported_extras,
        spans=spans, replay_decisions=len(records), replay_seconds=replay_seconds,
        all_recorded_actions_equal=bool(actions_match), branch_reasons=reasons,
        assessment_calls=assessment_calls,
        stage_list_lengths=dict(detection=len(controller._stage_detection_ms),
            tracking=len(controller._stage_tracking_ms), model=len(controller._stage_model_ms)),
        scope='Current pinned window controller startup and 600 recorded image-detection inputs; no training, collection benchmark, or success evaluation. Timings include audit hooks and serve only to locate executed work.')
    (OUT/'runtime_observations.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='assessment_calls'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
