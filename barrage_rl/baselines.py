"""Diagnostic baselines for checking whether the barrage task is learnable.

The privileged planner deliberately reads simulator state. It is never used as
an observation or reward input during PPO training; it only establishes an
upper baseline and catches environment/reward designs where no-op is optimal.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from .env import BarrageVisionEnv


Policy = Callable[[BarrageVisionEnv, np.random.Generator], int]


@dataclass(frozen=True)
class PlannerSupervision:
    action: int
    regrets: np.ndarray
    collision_mask: np.ndarray
    safety_targets: np.ndarray
    action_costs: np.ndarray
    urgent: bool


def noop_policy(env: BarrageVisionEnv, rng: np.random.Generator) -> int:
    del env, rng
    return 0


def random_policy(env: BarrageVisionEnv, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(env.ACTIONS)))


def privileged_planner_action(
    env: BarrageVisionEnv,
    horizon_seconds: float = 1.5,
    reaction_seconds: float = 0.30,
    wall_margin: float = 80.0,
    wall_penalty_weight: float = 0.35,
    safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
) -> int:
    action, _ = privileged_planner_target(
        env,
        horizon_seconds=horizon_seconds,
        reaction_seconds=reaction_seconds,
        wall_margin=wall_margin,
        wall_penalty_weight=wall_penalty_weight,
        safety_horizons=safety_horizons,
    )
    return action


def privileged_planner_target(
    env: BarrageVisionEnv,
    horizon_seconds: float = 1.5,
    reaction_seconds: float = 0.30,
    wall_margin: float = 80.0,
    temperature: float = 0.08,
    wall_penalty_weight: float = 0.35,
    safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
) -> tuple[int, np.ndarray]:
    supervision = privileged_planner_supervision(
        env,
        horizon_seconds=horizon_seconds,
        reaction_seconds=reaction_seconds,
        wall_margin=wall_margin,
        wall_penalty_weight=wall_penalty_weight,
        safety_horizons=safety_horizons,
    )
    target = np.zeros(len(env.ACTIONS), dtype=np.float32)
    if not supervision.urgent or temperature <= 0.0:
        target[supervision.action] = 1.0
        return supervision.action, target
    logits = -supervision.regrets / float(temperature)
    logits[supervision.collision_mask] = -80.0
    logits -= float(np.max(logits))
    target = np.exp(logits)
    target /= float(target.sum())
    return supervision.action, target.astype(np.float32, copy=False)


def privileged_planner_supervision(
    env: BarrageVisionEnv,
    horizon_seconds: float = 1.5,
    reaction_seconds: float = 0.30,
    wall_margin: float = 80.0,
    wall_penalty_weight: float = 0.35,
    safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
) -> PlannerSupervision:
    """Return exact-dynamics receding-horizon supervision without mutation.

    Nine first actions are simulated with the real substep, target respawn, RNG,
    clipping and mask collision.  Later decisions use a cheap privileged
    recovery rule and are still advanced by the exact simulator.  This keeps
    the teacher practical enough for online DAgger while avoiding the old
    straight-line approximation and forced-stay label conflict.
    """
    horizons = np.asarray(tuple(float(value) for value in safety_horizons), np.float32)
    if horizons.ndim != 1 or len(horizons) == 0 or np.any(horizons <= 0):
        raise ValueError("safety_horizons must contain positive values")
    horizons = np.sort(horizons)
    maximum_horizon = max(float(horizon_seconds), float(horizons[-1]))
    total_decisions = max(1, int(np.ceil(maximum_horizon / env.decision_dt)))
    reaction_decisions = max(
        1, int(np.ceil(float(reaction_seconds) / env.decision_dt))
    )
    horizon_steps = np.maximum(1, np.ceil(horizons / env.decision_dt).astype(int))
    action_count = len(env.ACTIONS)
    base = env.capture_state()
    action_costs = np.full(action_count, np.inf, np.float32)
    safety = np.ones((len(horizons), action_count), dtype=np.float32)
    try:
        for first in range(action_count):
            # Safety means "keep this action for the stated horizon", not
            # "a privileged controller repairs it on the next 33 ms tick".
            # The latter made almost every action look safe and left the risk
            # head with <2% positives.
            env.restore_state(base)
            constant_collision_step: Optional[int] = None
            for decision in range(int(horizon_steps[-1])):
                if env.simulate_action(first):
                    constant_collision_step = decision + 1
                    break
            for horizon_index, step in enumerate(horizon_steps):
                safety[horizon_index, first] = float(
                    constant_collision_step is not None
                    and constant_collision_step <= step
                )

            # Cost uses exact receding-horizon dynamics, but holds the first
            # command for the configured reaction time before privileged
            # recovery.  This matches deployment latency and makes adjacent
            # labels temporally coherent.
            env.restore_state(base)
            collision_step: Optional[int] = None
            danger_sum = 0.0
            for decision in range(total_decisions):
                action = first if decision < reaction_decisions else _reactive_recovery_action(
                    env, wall_margin
                )
                terminated = env.simulate_action(action)
                clearance = _minimum_center_clearance(env)
                danger_sum += float(np.exp(-max(clearance, 0.0) / 34.0))
                if terminated:
                    collision_step = decision + 1
                    break
            half_plane = env.plane_size / 2.0
            wall_distance = min(
                float(env.plane_position[0] - half_plane[0]),
                float(env.screen_width - half_plane[0] - env.plane_position[0]),
                float(env.plane_position[1] - half_plane[1]),
                float(env.screen_height - half_plane[1] - env.plane_position[1]),
            )
            collision_penalty = 0.0
            if collision_step is not None:
                collision_penalty = 1000.0 + 100.0 * (
                    total_decisions - collision_step + 1
                ) / total_decisions
            wall_fraction = max(float(wall_margin) - wall_distance, 0.0) / max(
                float(wall_margin), 1.0
            )
            action_costs[first] = (
                collision_penalty + danger_sum / total_decisions
                + float(wall_penalty_weight) * wall_fraction * wall_fraction
                + _action_change_cost(env, base, first)
            )
    finally:
        env.restore_state(base)
    half_plane = env.plane_size / 2.0
    initial_wall_distance = min(
        float(base.plane_position[0] - half_plane[0]),
        float(env.screen_width - half_plane[0] - base.plane_position[0]),
        float(base.plane_position[1] - half_plane[1]),
        float(env.screen_height - half_plane[1] - base.plane_position[1]),
    )
    if initial_wall_distance < wall_margin:
        # The first action must itself move inward; promising to recover only
        # after a stay still teaches wall-sticking to the student.
        for first in range(action_count):
            position = base.plane_position + env.ACTIONS[first] * env.speed * env.decision_dt
            first_wall_distance = min(
                float(position[0] - half_plane[0]),
                float(env.screen_width - half_plane[0] - position[0]),
                float(position[1] - half_plane[1]),
                float(env.screen_height - half_plane[1] - position[1]),
            )
            action_costs[first] += max(
                initial_wall_distance + 1.0 - first_wall_distance, 0.0
            )
    action = int(np.argmin(action_costs))
    regrets = action_costs - float(action_costs[action])
    collision_mask = safety[0].astype(np.bool_)
    urgent = bool(np.any(safety[:, action]) or np.any(collision_mask))
    return PlannerSupervision(
        action=action,
        regrets=np.clip(regrets, 0.0, 20.0).astype(np.float32),
        collision_mask=collision_mask,
        safety_targets=safety.astype(np.float32),
        action_costs=action_costs.astype(np.float32),
        urgent=urgent,
    )


def fast_planner_supervision(
    env: BarrageVisionEnv,
    horizon_seconds: float = 0.60,
    reaction_seconds: float = 0.30,
    wall_margin: float = 80.0,
    wall_penalty_weight: float = 0.35,
    safety_horizons: Sequence[float] = (0.10,),
) -> PlannerSupervision:
    """Vectorized short-horizon teacher used for dense stage-one collection.

    Bullets are linear within this short horizon; an off-screen respawn cannot
    become an immediate threat before the next decision.  The candidate command
    is held for the deployment reaction window and the plane then stops.  This
    gives the image policy a learnable, deterministic target without performing
    thousands of Python physics substeps for every training state.
    """
    horizons = np.sort(
        np.asarray(tuple(float(value) for value in safety_horizons), np.float32)
    )
    if horizons.ndim != 1 or not len(horizons) or np.any(horizons <= 0):
        raise ValueError("safety_horizons must contain positive values")
    decision_dt = float(env.decision_dt)
    steps = max(1, int(np.ceil(float(horizon_seconds) / decision_dt)))
    times = np.arange(1, steps + 1, dtype=np.float32) * decision_dt
    held_time = np.minimum(times, float(reaction_seconds))
    actions = np.asarray(env.ACTIONS, dtype=np.float32)
    plane = (
        env.plane_position[None, None, :]
        + actions[:, None, :] * float(env.speed) * held_time[None, :, None]
    )
    half_plane = env.plane_size.astype(np.float32) / 2.0
    plane[:, :, 0] = np.clip(
        plane[:, :, 0], half_plane[0], env.screen_width - half_plane[0]
    )
    plane[:, :, 1] = np.clip(
        plane[:, :, 1], half_plane[1], env.screen_height - half_plane[1]
    )
    bullets = (
        env.bullet_positions[None, :, :]
        + times[:, None, None] * env.bullet_velocities[None, :, :]
    )
    radius = 0.5 * (
        max(float(env.plane_size[0]), float(env.plane_size[1]))
        + max(env.bullet_surface.get_size())
    )
    clearance = np.linalg.norm(
        bullets[None, :, :, :] - plane[:, :, None, :], axis=-1
    ) - radius
    nearest_count = min(3, clearance.shape[-1])
    nearest = np.partition(clearance, nearest_count - 1, axis=-1)[
        :, :, :nearest_count
    ]
    time_weight = np.exp(-times / max(float(horizon_seconds), decision_dt))
    danger = (
        np.exp(-np.maximum(nearest, 0.0) / 34.0)
        * time_weight[None, :, None]
    ).mean(axis=(1, 2))
    collision_by_time = (clearance <= 0.0).any(axis=-1)
    collision_penalty = np.zeros(len(actions), dtype=np.float32)
    for action_index in range(len(actions)):
        hit = np.flatnonzero(collision_by_time[action_index])
        if len(hit):
            collision_penalty[action_index] = 1000.0 + 100.0 * (
                steps - int(hit[0])
            ) / steps
    wall_distance = np.minimum.reduce(
        (
            plane[:, :, 0] - half_plane[0],
            env.screen_width - half_plane[0] - plane[:, :, 0],
            plane[:, :, 1] - half_plane[1],
            env.screen_height - half_plane[1] - plane[:, :, 1],
        )
    ).min(axis=1)
    wall_fraction = np.maximum(float(wall_margin) - wall_distance, 0.0) / max(
        float(wall_margin), 1.0
    )
    current = env.plane_velocity.astype(np.float32) / max(float(env.speed), 1e-6)
    inertia = 0.03 * np.square(actions - current[None, :]).sum(axis=1)
    movement = 0.005 * np.square(actions).sum(axis=1)
    action_costs = (
        collision_penalty + danger
        + float(wall_penalty_weight) * np.square(wall_fraction)
        + inertia + movement
    ).astype(np.float32)
    action = int(np.argmin(action_costs))
    # Avoid direction chatter when moving offers no material safety gain.
    if action_costs[0] <= action_costs[action] + 0.01:
        action = 0
    regrets = action_costs - float(action_costs[action])
    regrets = np.maximum(regrets, 0.0)

    safety = np.zeros((len(horizons), len(actions)), dtype=np.float32)
    for horizon_index, horizon in enumerate(horizons):
        horizon_step = min(
            steps - 1, max(0, int(np.ceil(float(horizon) / decision_dt)) - 1)
        )
        safety[horizon_index] = collision_by_time[:, : horizon_step + 1].any(axis=1)
    collision_mask = safety[0].astype(np.bool_)
    return PlannerSupervision(
        action=action,
        regrets=np.clip(regrets, 0.0, 20.0).astype(np.float32),
        collision_mask=collision_mask,
        safety_targets=safety,
        action_costs=action_costs,
        urgent=bool(np.any(collision_mask)),
    )


def _reactive_recovery_action(env: BarrageVisionEnv, wall_margin: float) -> int:
    """Choose the next exact rollout action from privileged current geometry."""
    half_plane = env.plane_size / 2.0
    dt = env.decision_dt
    candidates = (
        env.plane_position[None, :] + env.ACTIONS * env.speed * dt
    )
    candidates[:, 0] = np.clip(
        candidates[:, 0], half_plane[0], env.screen_width - half_plane[0]
    )
    candidates[:, 1] = np.clip(
        candidates[:, 1], half_plane[1], env.screen_height - half_plane[1]
    )
    future_bullets = env.bullet_positions + env.bullet_velocities * dt
    radius = 0.5 * (
        max(float(env.plane_size[0]), float(env.plane_size[1]))
        + max(env.bullet_surface.get_size())
    )
    clearances = np.linalg.norm(
        future_bullets[None, :, :] - candidates[:, None, :], axis=2
    ) - radius
    nearest = np.partition(clearances, min(2, clearances.shape[1] - 1), axis=1)
    proximity = np.exp(-np.maximum(nearest[:, : min(3, clearances.shape[1])], 0.0) / 34.0).mean(axis=1)
    wall_distance = np.minimum.reduce(
        (
            candidates[:, 0] - half_plane[0],
            env.screen_width - half_plane[0] - candidates[:, 0],
            candidates[:, 1] - half_plane[1],
            env.screen_height - half_plane[1] - candidates[:, 1],
        )
    )
    wall_penalty = np.square(
        np.maximum(float(wall_margin) - wall_distance, 0.0)
        / max(float(wall_margin), 1.0)
    )
    collision = (clearances <= 0.0).any(axis=1).astype(np.float32) * 1000.0
    movement_tie_break = (np.arange(len(env.ACTIONS)) != 0) * 1e-5
    return int(np.argmin(collision + proximity + 0.35 * wall_penalty + movement_tie_break))


def _action_change_cost(
    env: BarrageVisionEnv, base: "object", action: int
) -> float:
    """Small observable inertia prior that suppresses 30 Hz teacher jitter."""
    current = np.asarray(base.plane_velocity, dtype=np.float32) / max(
        float(env.speed), 1e-6
    )
    difference = env.ACTIONS[int(action)] - current
    return 0.03 * float(np.dot(difference, difference))


def _minimum_center_clearance(env: BarrageVisionEnv) -> float:
    radius = 0.5 * (
        max(float(env.plane_size[0]), float(env.plane_size[1]))
        + max(env.bullet_surface.get_size())
    )
    return float(
        np.min(np.linalg.norm(env.bullet_positions - env.plane_position, axis=1) - radius)
    )


def privileged_policy(env: BarrageVisionEnv, rng: np.random.Generator) -> int:
    del rng
    return privileged_planner_action(env)


POLICIES: Dict[str, Policy] = {
    "noop": noop_policy,
    "random": random_policy,
    "privileged": privileged_policy,
}


def evaluate_policy(
    policy: Policy,
    bullet_count: int,
    episodes: int,
    seed: int = 20_000,
) -> np.ndarray:
    env = BarrageVisionEnv(bullet_count=bullet_count)
    rng = np.random.default_rng(seed + 1_000_000)
    survival_times = []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            info = {"survival_seconds": 0.0}
            while not (terminated or truncated):
                action = policy(env, rng)
                _, _, terminated, truncated, info = env.step(action)
            survival_times.append(float(info["survival_seconds"]))
    finally:
        env.close()
    return np.asarray(survival_times, dtype=np.float32)


def run_benchmark(
    bullet_counts: Iterable[int],
    episodes: int,
    seed: int,
) -> None:
    for bullet_count in bullet_counts:
        results: Dict[str, np.ndarray] = {}
        for name, policy in POLICIES.items():
            values = evaluate_policy(policy, bullet_count, episodes, seed)
            results[name] = values
            print(
                "bullets=%d policy=%s episodes=%d mean=%.3fs median=%.3fs"
                % (
                    bullet_count,
                    name,
                    episodes,
                    float(values.mean()),
                    float(np.median(values)),
                )
            )
        ratio = float(results["privileged"].mean() / results["noop"].mean())
        print("bullets=%d privileged/noop=%.3fx" % (bullet_count, ratio))


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate barrage diagnostic baselines")
    parser.add_argument("--bullets", type=int, nargs="+", default=[10, 25])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_benchmark(args.bullets, args.episodes, args.seed)


if __name__ == "__main__":
    main()
