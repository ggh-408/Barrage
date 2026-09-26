"""The pinned window controller, independent of training and legacy guards."""
from dataclasses import asdict
from functools import wraps
import numpy as np
import torch
from .action_selector import WindowActionSelection

CONTROLLER = 'planner_readiness_20260921_parallel'


def configure_window_controller(agent, *, search_workers=9):
    from .window_planner import WindowPixelGuard, WindowPlannerConfig
    from .tracked_policy import ActionQueryPolicy
    config = WindowPlannerConfig(search_workers=int(search_workers))
    existing = getattr(agent, '_receding_pixel_guard', None)
    if existing is not None:
        if not isinstance(existing, WindowPixelGuard) or existing.recovery_config != config:
            raise ValueError('A different window controller is already installed')
        return existing.manifest()
    guard = WindowPixelGuard(config)
    agent._receding_pixel_guard = guard
    original_select = agent._select_actions
    def select(objects, masks, globals_, *, deterministic):
        selection = original_select(objects, masks, globals_, deterministic=deterministic)
        return guard.apply(selection, objects.detach().cpu().numpy(), masks.detach().cpu().numpy(),
            globals_.detach().cpu().numpy())
    agent._select_actions = select
    original_act = agent.act_features
    @torch.inference_mode()
    def act(objects, masks, globals_, deterministic=True, episode_indices=None, **kwargs):
        model = agent.model
        if (not deterministic or not isinstance(model, ActionQueryPolicy) or model.training
                or agent.device.type != 'cpu' or agent.analytic_shield
                or agent._action_selector.teacher_cost_ranking_weight != 0
                or agent.action_hysteresis_bonus != 0 or model.continuation_head is not None):
            return original_act(objects, masks, globals_, deterministic=deterministic,
                episode_indices=episode_indices, **kwargs)
        policy = model.forward_policy(torch.as_tensor(objects, device=agent.device, dtype=torch.float32),
            torch.as_tensor(masks, device=agent.device, dtype=torch.bool),
            torch.as_tensor(globals_, device=agent.device, dtype=torch.float32))
        raw = policy.argmax(dim=1)
        selection = guard.apply(WindowActionSelection(raw, raw), objects, masks, globals_)
        actions = selection.actions.cpu().numpy()
        agent.overridden_decision_count += int(np.count_nonzero(actions != raw.cpu().numpy()))
        agent.decision_count += len(actions)
        return actions
    def with_context(original, episode_argument):
        @wraps(original)
        def invoke(*args, **kwargs):
            previous = guard._context_indices, guard._context_steps
            guard._context_indices = kwargs.get('episode_indices', args[episode_argument] if len(args)>episode_argument else None)
            guard._context_steps = kwargs.get('decision_steps', 1)
            try:
                return original(*args, **kwargs)
            finally:
                guard._context_indices, guard._context_steps = previous
        return invoke
    agent.act_features = with_context(act, 4)
    agent.act_features_with_diagnostics = with_context(agent.act_features_with_diagnostics, 3)
    original_reset = agent.reset_state
    def reset(episode_indices=None):
        guard.reset(episode_indices)
        return original_reset(episode_indices)
    agent.reset_state = reset
    return dict(kind='receding', algorithm='receding_continuation', config=asdict(config),
        experimental_controller=CONTROLLER, ranking_variant='commit_safe_ranking',
        ranking_safety_basis='nominal_pixel',
        trigger_statistics_basis='short_commitment_and_certificate',
        uncertainty_semantics='observed-track sensitivity; excludes future births and association errors')
