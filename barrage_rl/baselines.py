"""Privileged teachers used only to label training data and measure ceilings."""

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np

from .env import BarrageVisionEnv
from .runtime_core import OPENING_BATCH_COUNT


def _simulate_teacher_action(env: BarrageVisionEnv, action: int) -> bool:
    """Advance only dynamics observable at the current teacher boundary."""
    return env.simulate_action(
        action,
        include_scheduled_opening=False,
        include_respawns=False,
    )


@dataclass(frozen=True)
class PlannerSupervision:
    action: int
    regrets: np.ndarray
    collision_mask: np.ndarray
    safety_targets: np.ndarray
    action_costs: np.ndarray
    urgent: bool
    sequence_viable: Optional[np.ndarray] = None
    terminal_viable_action_count: Optional[np.ndarray] = None
    used_sequence_search: bool = False
    sequence_nodes_expanded: int = 0
    sequence_beam_width: int = 0
    greedy_survival_seconds: float = float("inf")


@dataclass(frozen=True)
class _SequenceNode:
    root_action: int
    last_action: int
    snapshot: Any
    survived_decisions: int
    minimum_clearance: float
    danger_sum: float
    action_changes: int


@dataclass(frozen=True)
class _RolloutResult:
    root_action: int
    survived_decisions: int
    collision: bool
    terminal_viable_action_count: int
    minimum_clearance: float
    terminal_wall_distance: float
    danger_mean: float
    action_changes: int
    snapshot: Any


def privileged_planner_supervision(
    env: BarrageVisionEnv,
    horizon_seconds: float = 1.5,
    reaction_seconds: float = 0.30,
    wall_margin: float = 80.0,
    wall_penalty_weight: float = 0.35,
    safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
    sequence_beam_width: int = 9,
    sequence_search_seconds: float = 0.60,
    strong_sequence_beam_width: int = 9,
    strong_sequence_search_seconds: float = 1.20,
    terminal_reserve_seconds: float = 0.30,
    force_sequence_search: bool = False,
) -> PlannerSupervision:
    """Return deterministic exact-dynamics sequence supervision without mutation.

    Every root action receives an exact greedy continuation.  States that are
    close to an inevitable collision additionally run a diverse beam over
    short action blocks.  Every branch carries a complete environment snapshot,
    including RNG state, so targeted respawns remain action-dependent and exact.
    Constant-action ``safety_targets`` retain their historical meaning.
    """
    horizons = np.asarray(tuple(float(value) for value in safety_horizons), np.float32)
    if horizons.ndim != 1 or len(horizons) == 0 or np.any(horizons <= 0):
        raise ValueError("safety_horizons must contain positive values")
    if sequence_beam_width <= 0 or strong_sequence_beam_width <= 0:
        raise ValueError("sequence beam widths must be positive")
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
    greedy_results: list[_RolloutResult] = []
    try:
        for first in range(action_count):
            # Safety means "keep this action for the stated horizon", not
            # "a privileged controller repairs it on the next 33 ms tick".
            # The latter made almost every action look safe and left the risk
            # head with <2% positives.
            env.restore_state(base)
            constant_collision_step: Optional[int] = None
            for decision in range(int(horizon_steps[-1])):
                if _simulate_teacher_action(env, first):
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
            minimum_clearance = float("inf")
            previous_action = int(first)
            action_changes = 0
            for decision in range(total_decisions):
                action = (
                    first
                    if decision < reaction_decisions
                    else _reactive_recovery_action(env, wall_margin)
                )
                action_changes += int(decision > 0 and int(action) != previous_action)
                previous_action = int(action)
                terminated = _simulate_teacher_action(env, action)
                clearance = _minimum_center_clearance(env)
                minimum_clearance = min(minimum_clearance, clearance)
                danger_sum += float(np.exp(-max(clearance, 0.0) / 34.0))
                if terminated:
                    collision_step = decision + 1
                    break
            wall_distance = _wall_distance(env)
            terminal = env.capture_simulation_state()
            # Terminal reserve checks are deliberately deferred.  Running a
            # nine-action reserve rollout for every root makes the common,
            # clearly safe path almost as expensive as the emergency beam.
            terminal_viable = 0 if collision_step is not None else -1
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
            greedy_results.append(_RolloutResult(
                root_action=int(first),
                survived_decisions=(
                    int(collision_step - 1)
                    if collision_step is not None
                    else int(total_decisions)
                ),
                collision=collision_step is not None,
                terminal_viable_action_count=int(terminal_viable),
                minimum_clearance=float(minimum_clearance),
                terminal_wall_distance=float(wall_distance),
                danger_mean=float(danger_sum / total_decisions),
                action_changes=int(action_changes),
                snapshot=terminal,
            ))

        greedy_action = int(np.argmin(action_costs))
        selected = greedy_results[greedy_action]
        all_greedy_collide = all(result.collision for result in greedy_results)
        greedy_survival_seconds = (
            float(selected.survived_decisions) * float(env.decision_dt)
            if selected.collision
            else float("inf")
        )
        moderate_risk = selected.collision and greedy_survival_seconds <= 0.80
        use_sequence_search = bool(force_sequence_search or moderate_risk)
        nodes_expanded = 0
        used_beam_width = 0
        best_results = list(greedy_results)
        if use_sequence_search:
            strong = bool(
                force_sequence_search
                or all_greedy_collide
            )
            used_beam_width = max(
                action_count,
                int(strong_sequence_beam_width if strong else sequence_beam_width),
            )
            explicit_seconds = float(
                strong_sequence_search_seconds if strong else sequence_search_seconds
            )
            # The beam is a receding-horizon controller.  A weak greedy tail
            # can falsely reject every safe explicit sequence, so rank the
            # exact searched interval plus its terminal escape reserve.
            rollout_seconds = max(float(explicit_seconds), env.decision_dt)
            searched, nodes_expanded = _sequence_beam_rollouts(
                env,
                base,
                beam_width=used_beam_width,
                explicit_seconds=explicit_seconds,
                horizon_seconds=rollout_seconds,
                wall_margin=float(wall_margin),
                terminal_reserve_seconds=float(terminal_reserve_seconds),
            )
            for candidate in searched:
                root = int(candidate.root_action)
                if _rollout_rank(candidate) < _rollout_rank(best_results[root]):
                    best_results[root] = candidate
            for root, result in enumerate(best_results):
                if result.terminal_viable_action_count < 0:
                    best_results[root] = replace(
                        result,
                        terminal_viable_action_count=_terminal_viable_action_count(
                            env, result.snapshot, terminal_reserve_seconds
                        ),
                    )
            action_costs = np.asarray(
                [
                    _rollout_cost(
                        result,
                        total_decisions=max(
                            1,
                            int(np.ceil(rollout_seconds / env.decision_dt)),
                        ),
                        wall_margin=float(wall_margin),
                        wall_penalty_weight=float(wall_penalty_weight),
                    )
                    + _action_change_cost(env, base, result.root_action)
                    for result in best_results
                ],
                dtype=np.float32,
            )
        else:
            selected_terminal_viable = (
                0
                if selected.collision
                else _terminal_viable_action_count(
                    env, selected.snapshot, terminal_reserve_seconds
                )
            )
            best_results[greedy_action] = replace(
                selected,
                terminal_viable_action_count=int(selected_terminal_viable),
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
        sequence_viable=np.asarray(
            [not result.collision for result in best_results], dtype=np.bool_
        ),
        terminal_viable_action_count=np.asarray(
            [result.terminal_viable_action_count for result in best_results],
            dtype=np.int16,
        ),
        used_sequence_search=bool(use_sequence_search),
        sequence_nodes_expanded=int(nodes_expanded),
        sequence_beam_width=int(used_beam_width),
        greedy_survival_seconds=float(greedy_survival_seconds),
    )


def _sequence_beam_rollouts(
    env: BarrageVisionEnv,
    base: Any,
    *,
    beam_width: int,
    explicit_seconds: float,
    horizon_seconds: float,
    wall_margin: float,
    terminal_reserve_seconds: float,
) -> tuple[list[_RolloutResult], int]:
    """Search exact short action blocks, then close every survivor greedily."""
    action_count = len(env.ACTIONS)
    width = max(action_count, int(beam_width))
    total_decisions = max(1, int(np.ceil(float(horizon_seconds) / env.decision_dt)))
    explicit_decisions = min(
        total_decisions,
        max(1, int(np.ceil(float(explicit_seconds) / env.decision_dt))),
    )
    macro_decisions = max(1, int(round(0.30 / env.decision_dt)))
    nodes: list[_SequenceNode] = []
    terminal_results: list[_RolloutResult] = []
    expanded = 0
    for root in range(action_count):
        env.restore_state(base)
        terminated = _simulate_teacher_action(env, root)
        expanded += 1
        clearance = _minimum_center_clearance(env)
        if terminated:
            terminal_results.append(_RolloutResult(
                root_action=root,
                survived_decisions=0,
                collision=True,
                terminal_viable_action_count=0,
                minimum_clearance=float(clearance),
                terminal_wall_distance=float(_wall_distance(env)),
                danger_mean=float(np.exp(-max(clearance, 0.0) / 34.0)),
                action_changes=0,
                snapshot=env.capture_simulation_state(),
            ))
            continue
        nodes.append(_SequenceNode(
            root_action=root,
            last_action=root,
            snapshot=env.capture_simulation_state(),
            survived_decisions=1,
            minimum_clearance=float(clearance),
            danger_sum=float(np.exp(-max(clearance, 0.0) / 34.0)),
            action_changes=0,
        ))

    while nodes and nodes[0].survived_decisions < explicit_decisions:
        remaining = explicit_decisions - nodes[0].survived_decisions
        held_decisions = min(macro_decisions, remaining)
        candidates: list[_SequenceNode] = []
        for node in nodes:
            for action in range(action_count):
                env.restore_simulation_state(node.snapshot)
                minimum_clearance = float(node.minimum_clearance)
                danger_sum = float(node.danger_sum)
                survived = int(node.survived_decisions)
                terminated = False
                for _ in range(held_decisions):
                    terminated = _simulate_teacher_action(env, action)
                    expanded += 1
                    clearance = _minimum_center_clearance(env)
                    minimum_clearance = min(minimum_clearance, clearance)
                    danger_sum += float(np.exp(-max(clearance, 0.0) / 34.0))
                    if terminated:
                        break
                    survived += 1
                if terminated:
                    terminal_results.append(_RolloutResult(
                        root_action=int(node.root_action),
                        survived_decisions=int(survived),
                        collision=True,
                        terminal_viable_action_count=0,
                        minimum_clearance=float(minimum_clearance),
                        terminal_wall_distance=float(_wall_distance(env)),
                        danger_mean=float(danger_sum / max(survived + 1, 1)),
                        action_changes=(
                            int(node.action_changes)
                            + int(int(action) != int(node.last_action))
                        ),
                        snapshot=env.capture_simulation_state(),
                    ))
                    continue
                candidates.append(_SequenceNode(
                    root_action=int(node.root_action),
                    last_action=int(action),
                    snapshot=env.capture_simulation_state(),
                    survived_decisions=int(survived),
                    minimum_clearance=float(minimum_clearance),
                    danger_sum=float(danger_sum),
                    action_changes=(
                        int(node.action_changes)
                        + int(int(action) != int(node.last_action))
                    ),
                ))
        nodes = _prune_sequence_nodes(candidates, width, env)

    survivor_results: list[_RolloutResult] = []
    for node in nodes:
        env.restore_simulation_state(node.snapshot)
        survived = int(node.survived_decisions)
        minimum_clearance = float(node.minimum_clearance)
        danger_sum = float(node.danger_sum)
        action_changes = int(node.action_changes)
        previous_action = int(node.last_action)
        collision = False
        while survived < total_decisions:
            action = _reactive_recovery_action(env, wall_margin)
            action_changes += int(int(action) != previous_action)
            previous_action = int(action)
            collision = _simulate_teacher_action(env, action)
            expanded += 1
            clearance = _minimum_center_clearance(env)
            minimum_clearance = min(minimum_clearance, clearance)
            danger_sum += float(np.exp(-max(clearance, 0.0) / 34.0))
            if collision:
                break
            survived += 1
        terminal = env.capture_simulation_state()
        survivor_results.append(_RolloutResult(
            root_action=int(node.root_action),
            survived_decisions=int(survived),
            collision=bool(collision),
            terminal_viable_action_count=0 if collision else -1,
            minimum_clearance=float(minimum_clearance),
            terminal_wall_distance=float(_wall_distance(env)),
            danger_mean=float(danger_sum / max(total_decisions, 1)),
            action_changes=int(action_changes),
            snapshot=terminal,
        ))

    # Reserve scoring is expensive, so evaluate only a small safety-ranked
    # shortlist per root after the beam has collapsed thousands of branches.
    terminal_results.extend(survivor_results)
    best_by_root: dict[int, _RolloutResult] = {}
    for root in range(action_count):
        root_candidates = [
            result for result in terminal_results if result.root_action == root
        ]
        shortlist = sorted(root_candidates, key=_rollout_rank)[:3]
        checked: list[_RolloutResult] = []
        for result in shortlist:
            viable = (
                0
                if result.collision
                else _terminal_viable_action_count(
                    env, result.snapshot, terminal_reserve_seconds
                )
            )
            checked.append(replace(
                result,
                terminal_viable_action_count=int(viable),
            ))
        if checked:
            best_by_root[root] = min(checked, key=_rollout_rank)
    return [best_by_root[root] for root in sorted(best_by_root)], expanded


def _prune_sequence_nodes(
    nodes: list[_SequenceNode], width: int, env: BarrageVisionEnv
) -> list[_SequenceNode]:
    if len(nodes) <= width:
        return sorted(nodes, key=lambda node: _sequence_node_rank(node, env))
    diverse: dict[tuple[int, int, int, int], _SequenceNode] = {}
    for node in nodes:
        position = np.asarray(node.snapshot.plane_position, dtype=np.float32)
        key = (
            int(node.root_action),
            int(round(float(position[0]) / 12.0)),
            int(round(float(position[1]) / 12.0)),
            int(node.last_action),
        )
        current = diverse.get(key)
        if current is None or _sequence_node_rank(node, env) < _sequence_node_rank(
            current, env
        ):
            diverse[key] = node
    ordered = sorted(diverse.values(), key=lambda node: _sequence_node_rank(node, env))
    retained: list[_SequenceNode] = []
    roots: set[int] = set()
    for node in ordered:
        if node.root_action not in roots:
            retained.append(node)
            roots.add(int(node.root_action))
            if len(retained) >= width:
                return retained
    retained_ids = {id(node) for node in retained}
    retained.extend(
        node for node in ordered if id(node) not in retained_ids
    )
    return retained[:width]


def _sequence_node_rank(node: _SequenceNode, env: BarrageVisionEnv) -> tuple[float, ...]:
    position = np.asarray(node.snapshot.plane_position, dtype=np.float32)
    half_plane = env.plane_size / 2.0
    wall_distance = min(
        float(position[0] - half_plane[0]),
        float(env.screen_width - half_plane[0] - position[0]),
        float(position[1] - half_plane[1]),
        float(env.screen_height - half_plane[1] - position[1]),
    )
    return (
        -float(node.minimum_clearance),
        -float(wall_distance),
        float(node.danger_sum / max(node.survived_decisions, 1)),
        float(node.action_changes),
        float(node.root_action),
        float(node.last_action),
    )


def _rollout_rank(result: _RolloutResult) -> tuple[float, ...]:
    return (
        float(result.collision),
        -float(result.survived_decisions),
        -float(result.terminal_viable_action_count),
        -float(result.minimum_clearance),
        -float(result.terminal_wall_distance),
        float(result.danger_mean),
        float(result.action_changes),
        float(result.root_action),
    )


def _rollout_cost(
    result: _RolloutResult,
    *,
    total_decisions: int,
    wall_margin: float,
    wall_penalty_weight: float,
) -> float:
    if result.collision:
        return 1000.0 + 100.0 * (
            max(total_decisions - int(result.survived_decisions), 0)
            / max(total_decisions, 1)
        )
    wall_fraction = max(float(wall_margin) - result.terminal_wall_distance, 0.0) / max(
        float(wall_margin), 1.0
    )
    return (
        10.0 * (len(BarrageVisionEnv.ACTIONS) - result.terminal_viable_action_count)
        + float(result.danger_mean)
        + float(wall_penalty_weight) * wall_fraction * wall_fraction
        + 0.001 * float(result.action_changes)
    )


def _terminal_viable_action_count(
    env: BarrageVisionEnv, snapshot: Any, reserve_seconds: float
) -> int:
    decisions = max(1, int(np.ceil(float(reserve_seconds) / env.decision_dt)))
    count = 0
    try:
        for action in range(len(env.ACTIONS)):
            env.restore_simulation_state(snapshot)
            survived = True
            for _ in range(decisions):
                if _simulate_teacher_action(env, action):
                    survived = False
                    break
            count += int(survived)
    finally:
        env.restore_simulation_state(snapshot)
    return int(count)


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
    """Small observable inertia prior that suppresses fixed-decision teacher jitter."""
    if int(base.opening_spawned_batches) < OPENING_BATCH_COUNT:
        direction = env.ACTIONS[int(action)]
        return 0.03 * float(np.dot(direction, direction))
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


def _wall_distance(env: BarrageVisionEnv) -> float:
    half_plane = env.plane_size / 2.0
    return min(
        float(env.plane_position[0] - half_plane[0]),
        float(env.screen_width - half_plane[0] - env.plane_position[0]),
        float(env.plane_position[1] - half_plane[1]),
        float(env.screen_height - half_plane[1] - env.plane_position[1]),
    )
