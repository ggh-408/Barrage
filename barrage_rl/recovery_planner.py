"""Vectorized privileged recovery planner for dense high-quality labels."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .baselines import PlannerSupervision
from .env import BarrageVisionEnv
from .runtime_core import OPENING_BATCH_COUNT


def _wall_distance(
    positions: np.ndarray, half_plane: np.ndarray, width: float, height: float
) -> np.ndarray:
    return np.minimum.reduce((
        positions[..., 0] - half_plane[0],
        width - half_plane[0] - positions[..., 0],
        positions[..., 1] - half_plane[1],
        height - half_plane[1] - positions[..., 1],
    ))


def _candidate_actions(
    plane: np.ndarray,
    bullets_next_decision: np.ndarray,
    actions: np.ndarray,
    movement_per_decision: float,
    half_plane: np.ndarray,
    width: float,
    height: float,
    collision_radius: float,
    wall_margin: float,
) -> np.ndarray:
    candidates = plane[:, None, :] + actions[None, :, :] * movement_per_decision
    candidates[..., 0] = np.clip(
        candidates[..., 0], half_plane[0], width - half_plane[0]
    )
    candidates[..., 1] = np.clip(
        candidates[..., 1], half_plane[1], height - half_plane[1]
    )
    clearances = np.linalg.norm(
        bullets_next_decision[None, None, :, :] - candidates[:, :, None, :],
        axis=-1,
    ) - collision_radius
    nearest_count = min(3, clearances.shape[-1])
    nearest = np.partition(clearances, nearest_count - 1, axis=-1)[
        ..., :nearest_count
    ]
    proximity = np.exp(-np.maximum(nearest, 0.0) / 34.0).mean(axis=-1)
    collision = (clearances <= 0.0).any(axis=-1).astype(np.float32) * 1000.0
    walls = _wall_distance(candidates, half_plane, width, height)
    wall_penalty = np.square(
        np.maximum(float(wall_margin) - walls, 0.0)
        / max(float(wall_margin), 1.0)
    )
    movement_tie_break = (
        (np.arange(len(actions), dtype=np.float32) != 0.0) * 1e-5
    )[None, :]
    return np.argmin(
        collision + proximity + 0.35 * wall_penalty + movement_tie_break,
        axis=1,
    )


def vectorized_recovery_supervision(
    env: BarrageVisionEnv,
    horizon_seconds: float = 1.5,
    reaction_seconds: float = 0.30,
    wall_margin: float = 80.0,
    wall_penalty_weight: float = 0.35,
    safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
) -> PlannerSupervision:
    """Evaluate nine first actions with vectorized receding recovery.

    The live environment is never mutated.  Bullets follow their observed
    linear trajectories during the short horizon; off-screen bullets are
    ignored because a respawned edge bullet cannot become an immediate central
    threat before the next real decision refreshes the plan.
    """
    horizons = np.sort(
        np.asarray(tuple(float(value) for value in safety_horizons), np.float32)
    )
    if horizons.ndim != 1 or not len(horizons) or np.any(horizons <= 0):
        raise ValueError("safety_horizons must contain positive values")
    maximum_horizon = max(float(horizon_seconds), float(horizons[-1]))
    if maximum_horizon <= 0.0:
        raise ValueError("horizon_seconds must be positive")

    actions = np.asarray(env.ACTIONS, dtype=np.float32)
    action_count = len(actions)
    physics_dt = float(env.delta_time)
    action_repeat = int(env.action_repeat)
    total_substeps = max(1, int(np.ceil(maximum_horizon / physics_dt)))
    reaction_decisions = max(
        1, int(np.ceil(float(reaction_seconds) / env.decision_dt))
    )
    half_plane = env.plane_size.astype(np.float32) / 2.0
    width = float(env.screen_width)
    height = float(env.screen_height)
    collision_radius = 0.5 * (
        max(float(env.plane_size[0]), float(env.plane_size[1]))
        + max(env.bullet_surface.get_size())
    )

    substep_times = (
        np.arange(1, total_substeps + 1, dtype=np.float32) * physics_dt
    )
    bullets = (
        env.bullet_positions[None, :, :]
        + substep_times[:, None, None] * env.bullet_velocities[None, :, :]
    )
    active = (
        (bullets[..., 0] >= 0.0)
        & (bullets[..., 0] <= width)
        & (bullets[..., 1] >= 0.0)
        & (bullets[..., 1] <= height)
    )

    # Constant-action collision targets use physics-substep resolution.
    constant_plane = (
        env.plane_position[None, None, :]
        + actions[:, None, :] * float(env.speed)
        * substep_times[None, :, None]
    )
    constant_plane[..., 0] = np.clip(
        constant_plane[..., 0], half_plane[0], width - half_plane[0]
    )
    constant_plane[..., 1] = np.clip(
        constant_plane[..., 1], half_plane[1], height - half_plane[1]
    )
    constant_distance = np.linalg.norm(
        bullets[None, :, :, :] - constant_plane[:, :, None, :], axis=-1
    )
    constant_collision = (
        (constant_distance <= collision_radius) & active[None, :, :]
    )
    safety = np.zeros((len(horizons), action_count), dtype=np.float32)
    for horizon_index, horizon in enumerate(horizons):
        steps = min(
            total_substeps,
            max(1, int(np.ceil(float(horizon) / physics_dt))),
        )
        safety[horizon_index] = constant_collision[:, :steps].any(axis=(1, 2))

    planes = np.repeat(
        env.plane_position.astype(np.float32)[None, :], action_count, axis=0
    )
    collision_step = np.full(action_count, -1, dtype=np.int32)
    danger_sum = np.zeros(action_count, dtype=np.float32)
    current_actions = np.arange(action_count, dtype=np.int64)
    movement_per_decision = float(env.speed) * float(env.decision_dt)
    decision_count = int(np.ceil(total_substeps / action_repeat))
    for decision in range(decision_count):
        start = decision * action_repeat
        stop = min(total_substeps, start + action_repeat)
        if decision >= reaction_decisions:
            # `_reactive_recovery_action` evaluates the bullet geometry one
            # decision from the current branch state.  `stop - 1` is exactly
            # that point; adding another action_repeat looked two decisions
            # ahead and changed evasive turns on dense crossings.
            lookahead_index = min(total_substeps - 1, stop - 1)
            current_actions = _candidate_actions(
                planes,
                bullets[lookahead_index],
                actions,
                movement_per_decision,
                half_plane,
                width,
                height,
                collision_radius,
                wall_margin,
            )
        directions = actions[current_actions]
        for substep in range(start, stop):
            planes += directions * float(env.speed) * physics_dt
            planes[:, 0] = np.clip(
                planes[:, 0], half_plane[0], width - half_plane[0]
            )
            planes[:, 1] = np.clip(
                planes[:, 1], half_plane[1], height - half_plane[1]
            )
            distances = np.linalg.norm(
                bullets[substep][None, :, :] - planes[:, None, :], axis=-1
            ) - collision_radius
            distances = np.where(active[substep][None, :], distances, np.inf)
            collision_now = (distances <= 0.0).any(axis=1)
            newly_collided = (collision_step < 0) & collision_now
            collision_step[newly_collided] = substep + 1
        # Match the exact rollout cost: the single minimum clearance drives
        # long-horizon danger.  The recovery policy itself still averages the
        # nearest three threats, as `_reactive_recovery_action` does.
        nearest = distances.min(axis=1)
        danger_sum += np.exp(-np.maximum(nearest, 0.0) / 34.0)

    collision_penalty = np.zeros(action_count, dtype=np.float32)
    collided = collision_step >= 0
    collision_penalty[collided] = 1000.0 + 100.0 * (
        total_substeps - collision_step[collided] + 1
    ) / total_substeps
    walls = _wall_distance(planes, half_plane, width, height)
    wall_fraction = np.maximum(float(wall_margin) - walls, 0.0) / max(
        float(wall_margin), 1.0
    )
    if env.opening_spawned_batches < OPENING_BATCH_COUNT:
        inertia = 0.03 * np.square(actions).sum(axis=1)
    else:
        current = env.plane_velocity.astype(np.float32) / max(
            float(env.speed), 1e-6
        )
        inertia = 0.03 * np.square(actions - current[None, :]).sum(axis=1)
    movement = 0.005 * np.square(actions).sum(axis=1)
    action_costs = (
        collision_penalty
        + danger_sum / max(decision_count, 1)
        + float(wall_penalty_weight) * np.square(wall_fraction)
        + inertia
        + movement
    ).astype(np.float32)

    initial_wall = float(
        _wall_distance(
            env.plane_position[None, :], half_plane, width, height
        )[0]
    )
    if initial_wall < wall_margin:
        first_positions = env.plane_position[None, :] + actions * movement_per_decision
        first_positions[:, 0] = np.clip(
            first_positions[:, 0], half_plane[0], width - half_plane[0]
        )
        first_positions[:, 1] = np.clip(
            first_positions[:, 1], half_plane[1], height - half_plane[1]
        )
        first_walls = _wall_distance(first_positions, half_plane, width, height)
        action_costs += np.maximum(initial_wall + 1.0 - first_walls, 0.0)

    action = int(np.argmin(action_costs))
    regrets = np.maximum(action_costs - float(action_costs[action]), 0.0)
    collision_mask = safety[0].astype(np.bool_)
    return PlannerSupervision(
        action=action,
        regrets=np.clip(regrets, 0.0, 20.0).astype(np.float32),
        collision_mask=collision_mask,
        safety_targets=safety,
        action_costs=action_costs,
        urgent=bool(np.any(safety[:, action]) or np.any(collision_mask)),
    )
