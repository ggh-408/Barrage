"""Process-parallel environment rollouts with centralized policy inference."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import threading
import traceback
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from queue import Empty
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from .causal_control import FixedActionDelay
from .baselines import privileged_planner_supervision
from .env import BarrageVisionEnv
from .tracked_policy import TrackedFeatureExtractor, TrackedPolicySpec


@dataclass
class ParallelRolloutResult:
    survival_times: np.ndarray
    termination_reasons: List[str]
    bullet_sizes: np.ndarray
    bullet_speeds: np.ndarray
    reset_modes: List[str]
    minimum_wall_distances: np.ndarray
    wall_steps: int
    model_steps: int
    action_histogram: np.ndarray
    failure_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    policy_events: List[Dict[str, Any]] = field(default_factory=list)
    minimum_bullet_clearances: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


@dataclass
class ParallelRolloutProgress:
    """Completed episode data available at a rollout progress boundary."""

    completed_indices: np.ndarray
    survival_times: np.ndarray
    termination_reasons: List[str]
    minimum_wall_distances: np.ndarray


@dataclass
class ParallelRolloutInitialState:
    """Image-policy state for a deterministic counterfactual branch.

    The simulator snapshot is used only to fork the world.  Policy inference
    after the forced diagnostic intervention continues to consume the same
    rendered-RGB tracker features as production evaluation.
    """

    env_snapshot: Any
    extractor: Any
    semanticizer_state: Dict[str, Any]
    objects: np.ndarray
    mask: np.ndarray
    globals_: np.ndarray
    forced_action: int
    forced_decisions: int
    controller_plan: Dict[str, Any] | None = None
    continuation_seconds: float | None = None


def _shared_view(raw: Any, dtype: Any, shape: Sequence[int]) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _instantaneous_bullet_clearance(env: BarrageVisionEnv) -> float:
    """Conservative center-distance clearance used for branch labels only."""

    if len(env.bullet_positions) == 0:
        return float("inf")
    collision_radius = 0.5 * (
        max(float(env.plane_size[0]), float(env.plane_size[1]))
        + max(float(value) for value in env.bullet_surface.get_size())
    )
    distances = np.linalg.norm(
        env.bullet_positions - env.plane_position[None, :], axis=1
    )
    return float(np.min(distances - collision_radius))


def _plane_spatial_diagnostic(env: BarrageVisionEnv) -> Dict[str, Any]:
    """Return plane-center coordinates and edge-to-wall clearances in pixels."""

    center_x = float(env.plane_position[0])
    center_y = float(env.plane_position[1])
    half_width = 0.5 * float(env.plane_size[0])
    half_height = 0.5 * float(env.plane_size[1])
    return {
        "plane_center": [center_x, center_y],
        "wall_clearance_pixels": {
            "left": center_x - half_width,
            "right": float(env.screen_width) - half_width - center_x,
            "top": center_y - half_height,
            "bottom": float(env.screen_height) - half_height - center_y,
        },
    }


def _failure_attribution(
    env: BarrageVisionEnv,
    extractor: TrackedFeatureExtractor,
    *,
    episode_index: int,
    episode_seed: int,
    physics_substeps: int,
    raw_policy_action: int,
    proposed_action: int,
    executed_action: int,
    raw_immediate_risk: float,
    selected_immediate_risk: float,
    action_was_filtered: bool,
    all_actions_unsafe: bool,
    safety_threshold: float,
) -> Dict[str, Any]:
    """Match exact collision bullets to the last image-derived tracks."""
    collision_indices = env.colliding_bullet_indices()
    tracker = extractor.tracker
    tracks = tracker.tracks
    elapsed = max(1, int(physics_substeps)) * float(env.delta_time)
    if tracks:
        predicted_positions = np.stack([
            track.position
            + (track.velocity * elapsed if track.velocity_known else 0.0)
            for track in tracks
        ]).astype(np.float32)
    else:
        predicted_positions = np.empty((0, 2), dtype=np.float32)

    bullets: List[Dict[str, Any]] = []
    for bullet_index in collision_indices:
        index = int(bullet_index)
        position = env.bullet_positions[index].astype(np.float32)
        nearest_index: int | None = None
        match_distance = float("inf")
        matched = False
        if len(predicted_positions):
            distances = np.linalg.norm(predicted_positions - position[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            match_distance = float(distances[nearest_index])
            nearest = tracks[nearest_index]
            gate = 15.0 if nearest.velocity_known else 24.0
            matched = match_distance <= gate
        nearest = tracks[nearest_index] if nearest_index is not None else None
        matched_track = nearest if matched else None
        bullets.append({
            "bullet_index": index,
            "targeted": bool(env.bullet_is_targeted[index]),
            "position": [float(value) for value in position],
            "velocity": [
                float(value) for value in env.bullet_velocities[index]
            ],
            "matched_track": bool(matched),
            "match_distance_pixels": match_distance if np.isfinite(match_distance) else None,
            "track_id": (
                int(matched_track.track_id) if matched_track is not None else None
            ),
            "track_age_decisions": (
                int(matched_track.age) if matched_track is not None else None
            ),
            "track_age_seconds": (
                float(max(matched_track.age - 1, 0) * tracker.decision_dt)
                if matched_track is not None
                else None
            ),
            "track_history_length": (
                len(matched_track.history) if matched_track is not None else None
            ),
            "track_velocity_known": (
                bool(matched_track.velocity_known)
                if matched_track is not None
                else None
            ),
            "track_missed": (
                int(matched_track.missed) if matched_track is not None else None
            ),
            "track_confidence": (
                float(matched_track.confidence) if matched_track is not None else None
            ),
            "track_position_uncertainty": (
                float(matched_track.position_uncertainty)
                if matched_track is not None
                else None
            ),
            "track_occluded_steps": (
                int(matched_track.occluded_steps)
                if matched_track is not None
                else None
            ),
            "track_association_group_size": (
                int(matched_track.association_group_size)
                if matched_track is not None
                else None
            ),
            "nearest_track_id": int(nearest.track_id) if nearest is not None else None,
            "nearest_track_age_decisions": (
                int(nearest.age) if nearest is not None else None
            ),
            "nearest_track_velocity_known": (
                bool(nearest.velocity_known) if nearest is not None else None
            ),
        })

    return {
        "episode_index": int(episode_index),
        "seed": int(episode_seed),
        "survival_seconds": float(env.physics_steps / env.physics_fps),
        **_plane_spatial_diagnostic(env),
        "physics_substeps_in_terminal_action": int(physics_substeps),
        "collision_bullet_count": len(bullets),
        "collision_targeted_any": any(item["targeted"] for item in bullets),
        "collision_targeted_all": bool(bullets) and all(
            item["targeted"] for item in bullets
        ),
        "collision_bullets": bullets,
        "tracker_track_count": len(tracks),
        "tracker_detection_count": int(tracker.last_detection_count),
        "tracker_known_velocity_fraction": float(tracker.known_velocity_fraction),
        "tracker_ambiguous_track_count": int(tracker.last_ambiguous_track_count),
        "raw_policy_action": int(raw_policy_action),
        "proposed_action": int(proposed_action),
        "executed_action": int(executed_action),
        "raw_immediate_collision_risk": float(raw_immediate_risk),
        "selected_immediate_collision_risk": float(selected_immediate_risk),
        "action_was_filtered": bool(action_was_filtered),
        "all_actions_unsafe": bool(all_actions_unsafe),
        "safety_threshold": float(safety_threshold),
    }


def _rollout_worker(
    episode_indices: Sequence[int],
    episodes: int,
    seed: int,
    episode_seeds: Optional[Sequence[int]],
    env_kwargs: Mapping[str, Any],
    spec_data: Mapping[str, Any],
    wall_threshold: float,
    rendered_rgb: bool,
    causal_action_delay_steps: int,
    objects_raw: Any,
    masks_raw: Any,
    globals_raw: Any,
    actions_raw: Any,
    active_raw: Any,
    survival_raw: Any,
    decision_indices_raw: Any,
    ready_barrier: Any,
    action_barrier: Any,
    stop_event: Any,
    result_queue: Any,
    error_queue: Any,
    collect_failure_diagnostics: bool,
    failure_lookback_decisions: int,
    safety_threshold: float,
    raw_policy_actions_raw: Any,
    learned_filter_actions_raw: Any,
    learned_filter_max_risk_raw: Any,
    selected_max_risk_raw: Any,
    learned_filter_teacher_cost_raw: Any,
    selected_teacher_cost_raw: Any,
    learned_analytic_clearance_raw: Any,
    selected_analytic_clearance_raw: Any,
    raw_immediate_risk_raw: Any,
    selected_immediate_risk_raw: Any,
    action_was_filtered_raw: Any,
    all_actions_unsafe_raw: Any,
    initial_states: Optional[Sequence[Optional[ParallelRolloutInitialState]]],
    tracker_implementation: str = "reference",
) -> None:
    envs: List[tuple[int, int, BarrageVisionEnv, Any, Any]] = []
    try:
        spec = TrackedPolicySpec(**dict(spec_data))
        extractor_options = {}
        if tracker_implementation == "window":
            from .window_tracker import WindowImageTracker
            extractor_options["tracker_class"] = WindowImageTracker
        objects = _shared_view(
            objects_raw,
            np.float32,
            (episodes, spec.max_objects, spec.object_features),
        )
        masks = _shared_view(masks_raw, np.uint8, (episodes, spec.max_objects))
        globals_ = _shared_view(
            globals_raw, np.float32, (episodes, spec.global_features)
        )
        actions = _shared_view(actions_raw, np.int64, (episodes,))
        active = _shared_view(active_raw, np.uint8, (episodes,))
        survival = _shared_view(survival_raw, np.float64, (episodes,))
        decision_indices = _shared_view(
            decision_indices_raw, np.int64, (episodes,)
        )
        raw_policy_actions = _shared_view(
            raw_policy_actions_raw, np.int64, (episodes,)
        )
        learned_filter_actions = _shared_view(
            learned_filter_actions_raw, np.int64, (episodes,)
        )
        learned_filter_max_risk = _shared_view(
            learned_filter_max_risk_raw, np.float32, (episodes,)
        )
        selected_max_risk = _shared_view(
            selected_max_risk_raw, np.float32, (episodes,)
        )
        learned_filter_teacher_cost = _shared_view(
            learned_filter_teacher_cost_raw, np.float32, (episodes,)
        )
        selected_teacher_cost = _shared_view(
            selected_teacher_cost_raw, np.float32, (episodes,)
        )
        learned_analytic_clearance = _shared_view(
            learned_analytic_clearance_raw, np.float32, (episodes,)
        )
        selected_analytic_clearance = _shared_view(
            selected_analytic_clearance_raw, np.float32, (episodes,)
        )
        raw_immediate_risk = _shared_view(
            raw_immediate_risk_raw, np.float32, (episodes,)
        )
        selected_immediate_risk = _shared_view(
            selected_immediate_risk_raw, np.float32, (episodes,)
        )
        action_was_filtered = _shared_view(
            action_was_filtered_raw, np.uint8, (episodes,)
        )
        all_actions_unsafe = _shared_view(
            all_actions_unsafe_raw, np.uint8, (episodes,)
        )

        minimum_wall_distances: Dict[int, float] = {}
        minimum_bullet_clearances: Dict[int, float] = {}
        records: Dict[int, Dict[str, Any]] = {}
        wall_steps = 0
        model_steps = 0
        histogram = np.zeros(len(BarrageVisionEnv.ACTIONS), dtype=np.int64)
        action_delays: Dict[int, FixedActionDelay] = {}
        decision_histories: Dict[int, Any] = {}
        policy_events: Dict[int, List[Dict[str, Any]]] = {}
        forced_actions: Dict[int, int] = {}
        forced_remaining: Dict[int, int] = {}

        for episode_index in episode_indices:
            env = BarrageVisionEnv(**dict(env_kwargs))
            episode_seed = (
                int(episode_seeds[episode_index])
                if episode_seeds is not None
                else seed + episode_index
            )
            observation, _ = env.reset(seed=episode_seed)
            episode_extractor = TrackedFeatureExtractor(
                spec, decision_dt=env.decision_dt, **extractor_options
            )
            rendered = None
            if rendered_rgb:
                from .tracked_collection import _RenderedRGBObservation

                rendered = _RenderedRGBObservation(
                    env, int(env.observation_size)
                )
                detected = rendered.detections(True)
                episode_objects, episode_mask, episode_globals = (
                    episode_extractor.reset_detections(
                        detected.bullet_positions, detected.plane_position
                    )
                )
            else:
                episode_objects, episode_mask, episode_globals = (
                    episode_extractor.reset(observation)
                )
            initial_state = (
                initial_states[episode_index]
                if initial_states is not None
                else None
            )
            if initial_state is not None:
                if rendered is None:
                    raise ValueError(
                        "counterfactual initial states require rendered_rgb=True"
                    )
                env.restore_state(initial_state.env_snapshot)
                if initial_state.continuation_seconds is not None:
                    if initial_state.continuation_seconds <= 0:
                        raise ValueError("continuation_seconds must be positive")
                    env.max_episode_physics_steps = env.physics_steps + int(
                        round(initial_state.continuation_seconds * env.physics_fps))
                episode_extractor = deepcopy(initial_state.extractor)
                for name, value in initial_state.semanticizer_state.items():
                    setattr(rendered.semanticizer, name, deepcopy(value))
                episode_objects = np.asarray(
                    initial_state.objects, dtype=np.float32
                ).copy()
                episode_mask = np.asarray(
                    initial_state.mask, dtype=np.bool_
                ).copy()
                episode_globals = np.asarray(
                    initial_state.globals_, dtype=np.float32
                ).copy()
                forced_action = int(initial_state.forced_action)
                if not 0 <= forced_action < len(BarrageVisionEnv.ACTIONS):
                    raise ValueError("forced_action is out of range")
                forced_decisions = int(initial_state.forced_decisions)
                if forced_decisions < 0:
                    raise ValueError("forced_decisions must be non-negative")
                forced_actions[episode_index] = forced_action
                forced_remaining[episode_index] = forced_decisions
            objects[episode_index] = episode_objects
            masks[episode_index] = episode_mask
            globals_[episode_index] = episode_globals
            minimum_wall_distances[episode_index] = float("inf")
            minimum_bullet_clearances[episode_index] = (
                _instantaneous_bullet_clearance(env)
            )
            active[episode_index] = 1
            action_delays[episode_index] = FixedActionDelay(
                causal_action_delay_steps, initial_action=0
            )
            if failure_lookback_decisions > 0:
                decision_histories[episode_index] = deque(
                    maxlen=int(failure_lookback_decisions)
                )
            policy_events[episode_index] = []
            decision_indices[episode_index] = int(env.episode_steps)
            envs.append(
                (episode_index, episode_seed, env, episode_extractor, rendered)
            )

        ready_barrier.wait()
        while True:
            action_barrier.wait()
            if stop_event.is_set():
                break
            for episode_index, episode_seed, env, episode_extractor, rendered in envs:
                if not active[episode_index]:
                    continue
                proposed_action = int(actions[episode_index])
                if forced_remaining.get(episode_index, 0) > 0:
                    proposed_action = forced_actions[episode_index]
                    forced_remaining[episode_index] -= 1
                action = action_delays[episode_index].push(proposed_action)
                if collect_failure_diagnostics and (
                    all_actions_unsafe[episode_index]
                    or proposed_action
                    != int(learned_filter_actions[episode_index])
                ):
                    learned_action = int(
                        learned_filter_actions[episode_index]
                    )
                    # Attribute current-task decisions with the same reaction
                    # time as tracked training, independent of legacy defaults.
                    diagnostic_teacher = privileged_planner_supervision(
                        env, reaction_seconds=0.10
                    )
                    diagnostic_safety = np.asarray(
                        diagnostic_teacher.safety_targets
                    )
                    policy_events[episode_index].append({
                        "decision_index": int(env.episode_steps),
                        "survival_seconds": float(
                            env.physics_steps * env.delta_time
                        ),
                        "raw_policy_action": int(
                            raw_policy_actions[episode_index]
                        ),
                        "learned_filter_action": learned_action,
                        "executed_action": action,
                        "exact_teacher_action": int(
                            diagnostic_teacher.action
                        ),
                        "exact_learned_filter_action_regret": float(
                            diagnostic_teacher.regrets[learned_action]
                        ),
                        "exact_selected_action_regret": float(
                            diagnostic_teacher.regrets[proposed_action]
                        ),
                        "exact_teacher_urgent": bool(
                            diagnostic_teacher.urgent
                        ),
                        "exact_safe_action_count_0p10": int(
                            np.count_nonzero(diagnostic_safety[0] < 0.5)
                        ),
                        "exact_safe_action_count_0p30": int(
                            np.count_nonzero(diagnostic_safety[1] < 0.5)
                        ),
                        "exact_safe_action_count_1p20": int(
                            np.count_nonzero(diagnostic_safety[3] < 0.5)
                        ),
                        "learned_filter_max_risk": float(
                            learned_filter_max_risk[episode_index]
                        ),
                        "selected_max_risk": float(
                            selected_max_risk[episode_index]
                        ),
                        "learned_filter_teacher_cost": float(
                            learned_filter_teacher_cost[episode_index]
                        ),
                        "selected_teacher_cost": float(
                            selected_teacher_cost[episode_index]
                        ),
                        "learned_analytic_clearance": float(
                            learned_analytic_clearance[episode_index]
                        ),
                        "selected_analytic_clearance": float(
                            selected_analytic_clearance[episode_index]
                        ),
                        "known_velocity_fraction": float(
                            globals_[episode_index, 8]
                        ),
                        "tracked_count_ratio": float(
                            globals_[episode_index, 9]
                        ),
                        "detection_count_ratio": float(
                            globals_[episode_index, 10]
                        ),
                        "count_deficit_fraction": float(
                            globals_[episode_index, 11]
                        ),
                        "occluded_fraction": float(
                            globals_[episode_index, 12]
                        ),
                        "ambiguous_fraction": float(
                            globals_[episode_index, 13]
                        ),
                        "near_threat_fraction": float(
                            globals_[episode_index, 14]
                        ),
                        "mean_position_uncertainty": float(
                            globals_[episode_index, 15]
                        ),
                    })
                if failure_lookback_decisions > 0:
                    valid = masks[episode_index].astype(np.bool_, copy=False)
                    episode_globals = globals_[episode_index]
                    decision_histories[episode_index].append({
                        "snapshot": env.capture_state(),
                        "decision_index": int(env.episode_steps),
                        "survival_seconds": float(
                            env.physics_steps * env.delta_time
                        ),
                        **_plane_spatial_diagnostic(env),
                        "raw_policy_action": int(
                            raw_policy_actions[episode_index]
                        ),
                        "learned_filter_action": int(
                            learned_filter_actions[episode_index]
                        ),
                        "proposed_action": proposed_action,
                        "executed_action": action,
                        "raw_immediate_risk": float(
                            raw_immediate_risk[episode_index]
                        ),
                        "selected_immediate_risk": float(
                            selected_immediate_risk[episode_index]
                        ),
                        "learned_filter_max_risk": float(
                            learned_filter_max_risk[episode_index]
                        ),
                        "selected_max_risk": float(
                            selected_max_risk[episode_index]
                        ),
                        "learned_filter_teacher_cost": float(
                            learned_filter_teacher_cost[episode_index]
                        ),
                        "selected_teacher_cost": float(
                            selected_teacher_cost[episode_index]
                        ),
                        "learned_analytic_clearance": float(
                            learned_analytic_clearance[episode_index]
                        ),
                        "selected_analytic_clearance": float(
                            selected_analytic_clearance[episode_index]
                        ),
                        "action_was_filtered": bool(
                            action_was_filtered[episode_index]
                        ),
                        "all_actions_unsafe": bool(
                            all_actions_unsafe[episode_index]
                        ),
                        "tracked_object_count": int(valid.sum()),
                        "known_velocity_fraction": float(episode_globals[8]),
                        "tracked_count_ratio": float(episode_globals[9]),
                        "detection_count_ratio": float(episode_globals[10]),
                        "count_deficit_fraction": float(episode_globals[11]),
                        "occluded_fraction": float(episode_globals[12]),
                        "ambiguous_fraction": float(episode_globals[13]),
                        "near_threat_fraction": float(episode_globals[14]),
                        "mean_position_uncertainty": float(episode_globals[15]),
                        "minimum_wall_distance": float(
                            np.min(episode_globals[2:6]) * spec.source_size
                        ),
                    })
                half_plane = env.plane_size / 2.0
                wall_distance = min(
                    float(env.plane_position[0] - half_plane[0]),
                    float(
                        env.screen_width
                        - half_plane[0]
                        - env.plane_position[0]
                    ),
                    float(env.plane_position[1] - half_plane[1]),
                    float(
                        env.screen_height
                        - half_plane[1]
                        - env.plane_position[1]
                    ),
                )
                minimum_wall_distances[episode_index] = min(
                    minimum_wall_distances[episode_index], wall_distance
                )
                wall_steps += int(wall_distance < wall_threshold)
                model_steps += 1
                histogram[action] += 1

                physics_before = int(env.physics_steps)
                observation, _, terminated, truncated, info = env.step(action)
                minimum_bullet_clearances[episode_index] = min(
                    minimum_bullet_clearances[episode_index],
                    _instantaneous_bullet_clearance(env),
                )
                if terminated or truncated:
                    survival[episode_index] = float(info["survival_seconds"])
                    active[episode_index] = 0
                    records[episode_index] = {
                        "termination_reason": (
                            "collision" if terminated else "time_limit"
                        ),
                        "bullet_size": int(info["bullet_size"]),
                        "bullet_speed": float(info["bullet_speed"]),
                        "reset_mode": str(info["reset_mode"]),
                        "policy_events": policy_events[episode_index],
                    }
                    if terminated and collect_failure_diagnostics:
                        records[episode_index]["failure_diagnostic"] = (
                            _failure_attribution(
                                env,
                                episode_extractor,
                                episode_index=episode_index,
                                episode_seed=episode_seed,
                                physics_substeps=int(env.physics_steps) - physics_before,
                                raw_policy_action=int(raw_policy_actions[episode_index]),
                                proposed_action=proposed_action,
                                executed_action=action,
                                raw_immediate_risk=float(
                                    raw_immediate_risk[episode_index]
                                ),
                                selected_immediate_risk=float(
                                    selected_immediate_risk[episode_index]
                                ),
                                action_was_filtered=bool(
                                    action_was_filtered[episode_index]
                                ),
                                all_actions_unsafe=bool(
                                    all_actions_unsafe[episode_index]
                                ),
                                safety_threshold=safety_threshold,
                            )
                        )
                        if failure_lookback_decisions > 0:
                            history = decision_histories[episode_index]
                            backtrace = []
                            terminal_survival = float(info["survival_seconds"])
                            for history_item in history:
                                env.restore_state(history_item.pop("snapshot"))
                                teacher = privileged_planner_supervision(
                                    env, reaction_seconds=0.10
                                )
                                executed = int(history_item["executed_action"])
                                learned = int(
                                    history_item["learned_filter_action"]
                                )
                                safety = np.asarray(teacher.safety_targets)
                                backtrace.append({
                                    **history_item,
                                    "seconds_before_death": float(
                                        terminal_survival
                                        - history_item["survival_seconds"]
                                    ),
                                    "teacher_action": int(teacher.action),
                                    "executed_action_regret": float(
                                        teacher.regrets[executed]
                                    ),
                                    "learned_filter_action_regret": float(
                                        teacher.regrets[learned]
                                    ),
                                    "teacher_urgent": bool(teacher.urgent),
                                    "safe_action_count_0p10": int(
                                        np.count_nonzero(safety[0] < 0.5)
                                    ),
                                    "safe_action_count_0p30": int(
                                        np.count_nonzero(safety[1] < 0.5)
                                    ),
                                    "safe_action_count_0p60": int(
                                        np.count_nonzero(safety[2] < 0.5)
                                    ),
                                    "safe_action_count_1p20": int(
                                        np.count_nonzero(safety[3] < 0.5)
                                    ),
                                    "executed_action_collision_0p10": bool(
                                        safety[0, executed] >= 0.5
                                    ),
                                    "executed_action_collision_0p30": bool(
                                        safety[1, executed] >= 0.5
                                    ),
                                    "executed_action_collision_0p60": bool(
                                        safety[2, executed] >= 0.5
                                    ),
                                    "executed_action_collision_1p20": bool(
                                        safety[3, executed] >= 0.5
                                    ),
                                    "learned_action_collision_0p10": bool(
                                        safety[0, learned] >= 0.5
                                    ),
                                    "learned_action_collision_0p30": bool(
                                        safety[1, learned] >= 0.5
                                    ),
                                    "learned_action_collision_0p60": bool(
                                        safety[2, learned] >= 0.5
                                    ),
                                    "learned_action_collision_1p20": bool(
                                        safety[3, learned] >= 0.5
                                    ),
                                })
                            records[episode_index]["failure_diagnostic"][
                                "backtrace"
                            ] = backtrace
                else:
                    if rendered is not None:
                        detected = rendered.detections(
                            False, extractor=episode_extractor, decision_steps=1
                        )
                        episode_objects, episode_mask, episode_globals = (
                            episode_extractor.step_detections(
                                detected.bullet_positions,
                                detected.plane_position,
                            )
                        )
                    else:
                        episode_objects, episode_mask, episode_globals = (
                            episode_extractor.step(observation)
                        )
                    objects[episode_index] = episode_objects
                    masks[episode_index] = episode_mask
                    globals_[episode_index] = episode_globals
                decision_indices[episode_index] = int(env.episode_steps)
            ready_barrier.wait()

        result_queue.put(
            {
                "minimum_wall_distances": minimum_wall_distances,
                "minimum_bullet_clearances": minimum_bullet_clearances,
                "records": records,
                "wall_steps": wall_steps,
                "model_steps": model_steps,
                "action_histogram": histogram,
            }
        )
    except BaseException:
        error_queue.put(traceback.format_exc())
        try:
            ready_barrier.abort()
            action_barrier.abort()
        except BaseException:
            pass
    finally:
        for _, _, env, _, _ in envs:
            env.close()


def _queued_error(error_queue: Any) -> Optional[str]:
    try:
        return str(error_queue.get_nowait())
    except Empty:
        return None


def run_parallel_rollout(
    agent: Any,
    spec: Any,
    episodes: int,
    workers: int,
    seed: int,
    env_kwargs: Mapping[str, Any],
    wall_threshold: float,
    episode_seeds: Optional[Sequence[int]] = None,
    rendered_rgb: bool = False,
    causal_action_delay_steps: int = 1,
    collect_failure_diagnostics: bool = False,
    failure_lookback_decisions: int = 0,
    initial_states: Optional[
        Sequence[Optional[ParallelRolloutInitialState]]
    ] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tracker_implementation: str = "reference",
) -> ParallelRolloutResult:
    """Run deterministic model evaluation episodes in worker processes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if tracker_implementation not in ("reference", "window"):
        raise ValueError("Unknown evaluation tracker implementation")
    if tracker_implementation == "window" and not rendered_rgb:
        raise ValueError("The window evaluation tracker requires rendered RGB")
    if episode_seeds is not None and len(episode_seeds) != episodes:
        raise ValueError("episode_seeds must contain exactly one seed per episode")
    if initial_states is not None and len(initial_states) != episodes:
        raise ValueError("initial_states must contain exactly one entry per episode")
    if causal_action_delay_steps not in (0, 1):
        raise ValueError("causal_action_delay_steps must be zero or one")
    if failure_lookback_decisions < 0:
        raise ValueError("failure_lookback_decisions must be non-negative")
    if failure_lookback_decisions and not collect_failure_diagnostics:
        raise ValueError(
            "failure lookback requires collect_failure_diagnostics=True"
        )
    normalized_episode_seeds = (
        tuple(int(value) for value in episode_seeds)
        if episode_seeds is not None
        else None
    )
    worker_count = max(1, min(int(workers), episodes))
    context = mp.get_context("spawn")

    objects_raw = context.RawArray(
        ctypes.c_float, episodes * spec.max_objects * spec.object_features
    )
    masks_raw = context.RawArray(ctypes.c_ubyte, episodes * spec.max_objects)
    globals_raw = context.RawArray(ctypes.c_float, episodes * spec.global_features)
    actions_raw = context.RawArray(ctypes.c_int64, episodes)
    active_raw = context.RawArray(ctypes.c_ubyte, episodes)
    survival_raw = context.RawArray(ctypes.c_double, episodes)
    decision_indices_raw = context.RawArray(ctypes.c_int64, episodes)
    raw_policy_actions_raw = context.RawArray(ctypes.c_int64, episodes)
    learned_filter_actions_raw = context.RawArray(ctypes.c_int64, episodes)
    learned_filter_max_risk_raw = context.RawArray(ctypes.c_float, episodes)
    selected_max_risk_raw = context.RawArray(ctypes.c_float, episodes)
    learned_filter_teacher_cost_raw = context.RawArray(
        ctypes.c_float, episodes
    )
    selected_teacher_cost_raw = context.RawArray(ctypes.c_float, episodes)
    learned_analytic_clearance_raw = context.RawArray(ctypes.c_float, episodes)
    selected_analytic_clearance_raw = context.RawArray(ctypes.c_float, episodes)
    raw_immediate_risk_raw = context.RawArray(ctypes.c_float, episodes)
    selected_immediate_risk_raw = context.RawArray(ctypes.c_float, episodes)
    action_was_filtered_raw = context.RawArray(ctypes.c_ubyte, episodes)
    all_actions_unsafe_raw = context.RawArray(ctypes.c_ubyte, episodes)
    ready_barrier = context.Barrier(worker_count + 1)
    action_barrier = context.Barrier(worker_count + 1)
    stop_event = context.Event()
    result_queue = context.Queue()
    error_queue = context.Queue()

    assignments = [
        list(range(index, episodes, worker_count)) for index in range(worker_count)
    ]
    processes = [
        context.Process(
            target=_rollout_worker,
            args=(
                assignment,
                episodes,
                seed,
                normalized_episode_seeds,
                dict(env_kwargs),
                asdict(spec),
                wall_threshold,
                bool(rendered_rgb),
                int(causal_action_delay_steps),
                objects_raw,
                masks_raw,
                globals_raw,
                actions_raw,
                active_raw,
                survival_raw,
                decision_indices_raw,
                ready_barrier,
                action_barrier,
                stop_event,
                result_queue,
                error_queue,
                bool(collect_failure_diagnostics),
                int(failure_lookback_decisions),
                float(getattr(agent, "safety_threshold", 0.5)),
                raw_policy_actions_raw,
                learned_filter_actions_raw,
                learned_filter_max_risk_raw,
                selected_max_risk_raw,
                learned_filter_teacher_cost_raw,
                selected_teacher_cost_raw,
                learned_analytic_clearance_raw,
                selected_analytic_clearance_raw,
                raw_immediate_risk_raw,
                selected_immediate_risk_raw,
                action_was_filtered_raw,
                all_actions_unsafe_raw,
                initial_states,
                tracker_implementation,
            ),
        )
        for assignment in assignments
    ]

    objects = _shared_view(
        objects_raw,
        np.float32,
        (episodes, spec.max_objects, spec.object_features),
    )
    masks = _shared_view(masks_raw, np.uint8, (episodes, spec.max_objects))
    globals_ = _shared_view(globals_raw, np.float32, (episodes, spec.global_features))
    actions = _shared_view(actions_raw, np.int64, (episodes,))
    active = _shared_view(active_raw, np.uint8, (episodes,))
    survival = _shared_view(survival_raw, np.float64, (episodes,))
    decision_indices = _shared_view(
        decision_indices_raw, np.int64, (episodes,)
    )
    raw_policy_actions = _shared_view(
        raw_policy_actions_raw, np.int64, (episodes,)
    )
    learned_filter_actions = _shared_view(
        learned_filter_actions_raw, np.int64, (episodes,)
    )
    learned_filter_max_risk = _shared_view(
        learned_filter_max_risk_raw, np.float32, (episodes,)
    )
    selected_max_risk = _shared_view(
        selected_max_risk_raw, np.float32, (episodes,)
    )
    learned_filter_teacher_cost = _shared_view(
        learned_filter_teacher_cost_raw, np.float32, (episodes,)
    )
    selected_teacher_cost = _shared_view(
        selected_teacher_cost_raw, np.float32, (episodes,)
    )
    learned_analytic_clearance = _shared_view(
        learned_analytic_clearance_raw, np.float32, (episodes,)
    )
    selected_analytic_clearance = _shared_view(
        selected_analytic_clearance_raw, np.float32, (episodes,)
    )
    raw_immediate_risk = _shared_view(
        raw_immediate_risk_raw, np.float32, (episodes,)
    )
    selected_immediate_risk = _shared_view(
        selected_immediate_risk_raw, np.float32, (episodes,)
    )
    action_was_filtered = _shared_view(
        action_was_filtered_raw, np.uint8, (episodes,)
    )
    all_actions_unsafe = _shared_view(
        all_actions_unsafe_raw, np.uint8, (episodes,)
    )

    for process in processes:
        process.start()
    if initial_states is not None and hasattr(agent, '_receding_pixel_guard'):
        for index, state in enumerate(initial_states):
            if state is not None and state.controller_plan is not None:
                agent._receding_pixel_guard._plans[index] = deepcopy(state.controller_plan)
    worker_payloads: List[Dict[str, Any]] = []
    last_completed = 0
    try:
        ready_barrier.wait()
        while True:
            active_indices = np.flatnonzero(active)
            completed = episodes - len(active_indices)
            if progress_callback is not None and completed > last_completed:
                progress_callback(completed, episodes)
                last_completed = completed
            if not len(active_indices):
                stop_event.set()
                action_barrier.wait()
                break
            if collect_failure_diagnostics:
                active_actions, diagnostics = agent.act_features_with_diagnostics(
                    objects[active_indices],
                    masks[active_indices].astype(np.bool_),
                    globals_[active_indices],
                    episode_indices=active_indices,
                )
                raw_policy_actions[active_indices] = diagnostics[
                    "raw_policy_actions"
                ]
                learned_filter_actions[active_indices] = diagnostics[
                    "learned_filter_actions"
                ]
                learned_filter_max_risk[active_indices] = diagnostics[
                    "learned_filter_max_risk"
                ]
                selected_max_risk[active_indices] = diagnostics[
                    "selected_max_risk"
                ]
                learned_filter_teacher_cost[active_indices] = diagnostics[
                    "learned_filter_teacher_cost"
                ]
                selected_teacher_cost[active_indices] = diagnostics[
                    "selected_teacher_cost"
                ]
                learned_analytic_clearance[active_indices] = diagnostics.get(
                    "learned_analytic_clearance", np.nan
                )
                selected_analytic_clearance[active_indices] = diagnostics.get(
                    "selected_analytic_clearance", np.nan
                )
                raw_immediate_risk[active_indices] = diagnostics[
                    "raw_immediate_risk"
                ]
                selected_immediate_risk[active_indices] = diagnostics[
                    "selected_immediate_risk"
                ]
                action_was_filtered[active_indices] = diagnostics[
                    "action_was_filtered"
                ]
                all_actions_unsafe[active_indices] = diagnostics[
                    "all_actions_unsafe"
                ]
            else:
                active_actions = agent.act_features(
                    objects[active_indices],
                    masks[active_indices].astype(np.bool_),
                    globals_[active_indices],
                    deterministic=True,
                    episode_indices=active_indices,
                    decision_indices=decision_indices[active_indices],
                )
            if initial_states is not None:
                for row, index in enumerate(active_indices):
                    initial = initial_states[int(index)]
                    if (initial is not None and int(decision_indices[index]) <
                            initial.env_snapshot.episode_steps + initial.forced_decisions and
                            int(active_actions[row]) != initial.forced_action and hasattr(agent,'reset_state')):
                        # A forced root must not retain a plan produced for a
                        # different, unexecuted root, even against a wall.
                        agent.reset_state(np.asarray([index]))
            actions[active_indices] = active_actions
            action_barrier.wait()
            ready_barrier.wait()
        # Drain the multiprocessing queue before joining workers.  On Windows,
        # a worker's Queue feeder thread can block process shutdown once the
        # pipe buffer fills.  Joining first then times out and incorrectly
        # reports otherwise-successful workers as failed on larger evaluations.
        for _ in processes:
            worker_payloads.append(result_queue.get(timeout=30.0))
    except threading.BrokenBarrierError as error:
        message = _queued_error(error_queue) or str(error)
        raise RuntimeError(f"evaluation worker failed:\n{message}") from error
    except Empty as error:
        message = _queued_error(error_queue)
        raise RuntimeError(
            message or "evaluation worker result timed out"
        ) from error
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join()

    worker_error = _queued_error(error_queue)
    failed_processes = [process.pid for process in processes if process.exitcode]
    if worker_error or failed_processes:
        raise RuntimeError(
            worker_error or f"evaluation workers failed: {failed_processes}"
        )

    termination_reasons = [""] * episodes
    bullet_sizes = np.zeros(episodes, dtype=np.int64)
    bullet_speeds = np.zeros(episodes, dtype=np.float64)
    reset_modes = [""] * episodes
    minimum_wall_distances = np.full(episodes, np.inf, dtype=np.float64)
    minimum_bullet_clearances = np.full(episodes, np.inf, dtype=np.float64)
    wall_steps = 0
    model_steps = 0
    histogram = np.zeros(len(BarrageVisionEnv.ACTIONS), dtype=np.int64)
    failure_diagnostics: List[Dict[str, Any]] = []
    policy_events: List[Dict[str, Any]] = []
    for payload in worker_payloads:
        wall_steps += int(payload["wall_steps"])
        model_steps += int(payload["model_steps"])
        histogram += np.asarray(payload["action_histogram"], dtype=np.int64)
        for episode_index, value in payload["minimum_wall_distances"].items():
            minimum_wall_distances[int(episode_index)] = float(value)
        for episode_index, value in payload["minimum_bullet_clearances"].items():
            minimum_bullet_clearances[int(episode_index)] = float(value)
        for episode_index, record in payload["records"].items():
            index = int(episode_index)
            termination_reasons[index] = record["termination_reason"]
            bullet_sizes[index] = int(record["bullet_size"])
            bullet_speeds[index] = float(record["bullet_speed"])
            reset_modes[index] = record["reset_mode"]
            if "failure_diagnostic" in record:
                failure_diagnostics.append(record["failure_diagnostic"])
            for event in record.get("policy_events", []):
                policy_events.append({
                    "episode_index": index,
                    "seed": (
                        int(normalized_episode_seeds[index])
                        if normalized_episode_seeds is not None
                        else int(seed + index)
                    ),
                    **event,
                })

    return ParallelRolloutResult(
        survival_times=survival.copy(),
        termination_reasons=termination_reasons,
        bullet_sizes=bullet_sizes,
        bullet_speeds=bullet_speeds,
        reset_modes=reset_modes,
        minimum_wall_distances=minimum_wall_distances,
        wall_steps=wall_steps,
        model_steps=model_steps,
        action_histogram=histogram,
        failure_diagnostics=sorted(
            failure_diagnostics, key=lambda item: int(item["episode_index"])
        ),
        policy_events=sorted(
            policy_events,
            key=lambda item: (int(item["episode_index"]), int(item["decision_index"])),
        ),
        minimum_bullet_clearances=minimum_bullet_clearances,
    )
