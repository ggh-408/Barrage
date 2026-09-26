"""Training-only finite-scenario exact teacher for pixel quantization ambiguity.

The nine coherent scenarios cover the relative-offset grid, not the Cartesian
product of independent offsets for every bullet. They are sensitivity labels,
not a continuous-set safety certificate. No new deployment observation is used.
"""
from dataclasses import dataclass, replace
import numpy as np
from .baselines import privileged_planner_supervision


@dataclass(frozen=True)
class QuantizedTeacherConfig:
    plane_half_width_pixels: float = .5001
    bullet_half_width_pixels: float = .5001
    aggregation: str = 'maximum_action_cost'
    scenario_layout: str = 'coherent_opposed_3x3'

    def __post_init__(self):
        widths = (self.plane_half_width_pixels, self.bullet_half_width_pixels)
        if not all(np.isfinite(x) and x >= 0 for x in widths):
            raise ValueError('Position half widths must be finite and nonnegative')
        if self.aggregation != 'maximum_action_cost' or self.scenario_layout != 'coherent_opposed_3x3':
            raise ValueError('Unsupported teacher scenario or aggregation')

    @property
    def relative_half_width_pixels(self):
        return self.plane_half_width_pixels + self.bullet_half_width_pixels


DEFAULT_CONFIG = QuantizedTeacherConfig()


def scenario_positions(env, config=DEFAULT_CONFIG):
    """Center on canonical rendered pixels, with one shared plane offset.

    Exact simulator sprite centers identify the pixel cells for this offline
    teacher. This models quantization only, not detector/association outliers.
    """
    plane = np.asarray(env.plane_surface.get_rect(center=tuple(env.plane_position)).center, np.float32)
    bullets = np.asarray([env.bullet_surface.get_rect(center=tuple(p)).center
                          for p in env.bullet_positions], np.float32).reshape(-1, 2)
    for x, y in ((0,0),(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        offset = np.asarray((x,y),np.float32)
        yield (np.clip(plane+offset*config.plane_half_width_pixels,
                       env.plane_size/2, np.array((env.screen_width,env.screen_height))-env.plane_size/2),
               bullets-offset*config.bullet_half_width_pixels)


def quantized_planner_supervision(env, *, uncertainty=DEFAULT_CONFIG, **kwargs):
    if uncertainty.plane_half_width_pixels == uncertainty.bullet_half_width_pixels == 0:
        return privileged_planner_supervision(env, **kwargs)
    base = env.capture_state()
    positions = list(scenario_positions(env, uncertainty))
    labels = []
    try:
        for plane, bullets in positions:
            env.restore_state(base)
            env.plane_position[:] = plane
            env.bullet_positions[:] = bullets
            labels.append(privileged_planner_supervision(env, **kwargs))
    finally:
        env.restore_state(base)
    costs = np.max(np.stack([s.action_costs for s in labels]),axis=0)
    action = int(np.argmin(costs))
    safety = np.max(np.stack([s.safety_targets for s in labels]),axis=0)
    viable = None if any(s.sequence_viable is None for s in labels) else np.all(
        np.stack([s.sequence_viable for s in labels]),axis=0)
    terminal = None if any(s.terminal_viable_action_count is None for s in labels) else np.min(
        np.stack([s.terminal_viable_action_count for s in labels]),axis=0)
    return replace(labels[0], action=action, action_costs=costs.astype(np.float32),
        regrets=np.clip(costs-costs[action],0,20).astype(np.float32),
        safety_targets=safety.astype(np.float32), collision_mask=safety[0].astype(bool),
        urgent=bool(np.any(safety[:,action]) or np.any(safety[0])),
        sequence_viable=viable,terminal_viable_action_count=terminal,
        used_sequence_search=any(s.used_sequence_search for s in labels),
        sequence_nodes_expanded=sum(s.sequence_nodes_expanded for s in labels),
        sequence_beam_width=max(s.sequence_beam_width for s in labels),
        greedy_survival_seconds=min(s.greedy_survival_seconds for s in labels))


def install_collection_teacher():
    """Only change training collection's binding, including in spawned workers."""
    from . import tracked_collection
    tracked_collection.privileged_planner_supervision = quantized_planner_supervision
