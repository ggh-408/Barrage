"""Benchmark deterministic tracked DAgger collection worker counts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from barrage_rl.artifacts import atomic_write_json
from barrage_rl.task_spec import TARGET_TASK, tracking_capacity_for
from barrage_rl.tracked_collection import ParallelTrackedDaggerEnv
from barrage_rl.tracked_policy import TrackedPolicySpec


def _check_arrays(expected, actual, step):
    if len(expected) != len(actual):
        raise AssertionError(f'Collection output count differs at step {step}')
    for index, (left, right) in enumerate(zip(expected, actual)):
        if (left.dtype != right.dtype or left.shape != right.shape
                or left.tobytes() != right.tobytes()):
            raise AssertionError(f'Collection output {index} differs at step {step}')


def benchmark(
    workers: int,
    env_count: int,
    decisions: int,
    teacher_kind: str,
    observation_size: int,
    deployment_rgb_observation: bool,
    action_repeat_choices: tuple[int, ...],
    bullet_count: int,
    causal_action_delay_steps: int,
    initial_episode_seeds: tuple[int, ...] = (),
    repeat_initial_episode_seeds: bool = False,
    tracking_capacity_override: int | None = None,
    reference_trajectory: list[tuple[np.ndarray, ...]] | None = None,
    teacher_reaction_seconds: float = 0.10,
    feature_expected_bullet_count: int | None = None,
    episode_limit_seconds: float = 120.0,
) -> dict[str, object]:
    capture_reference = reference_trajectory is not None and not reference_trajectory
    if reference_trajectory and len(reference_trajectory) != decisions + 1:
        raise ValueError('Reference trajectory must contain the initial state and each step')
    validation_seconds = 0.0
    env_kwargs = {
        "bullet_count": int(bullet_count),
        "bullet_size_min": 5,
        "bullet_size_max": 5,
        "bullet_speed_min": 240.0,
        "bullet_speed_max": 240.0,
        "targeted_bullet_probability": 0.10,
        "observation_size": int(observation_size),
        "max_episode_seconds": float(episode_limit_seconds),
        "randomize_initial_phase": False,
    }
    started = time.perf_counter()
    tracking_capacity = (
        tracking_capacity_for(bullet_count)
        if tracking_capacity_override is None
        else int(tracking_capacity_override)
    )
    with ParallelTrackedDaggerEnv(
        env_count=env_count,
        workers=workers,
        seed=2_100_000,
        initial_episode_seeds=initial_episode_seeds,
        repeat_initial_episode_seeds=repeat_initial_episode_seeds,
        env_kwargs=env_kwargs,
        spec=TrackedPolicySpec(
            max_objects=tracking_capacity,
            tracker_capacity=tracking_capacity,
            expected_bullet_count=(bullet_count if feature_expected_bullet_count is None
                                   else int(feature_expected_bullet_count)),
        ),
        teacher_kind=teacher_kind,
        teacher_reaction_seconds=teacher_reaction_seconds,
        deployment_rgb_observation=deployment_rgb_observation,
        causal_action_delay_steps=causal_action_delay_steps,
        action_repeat_choices=action_repeat_choices,
    ) as pipeline:
        startup = time.perf_counter() - started
        stepped = time.perf_counter()
        active_track_counts: list[int] = []
        def check_frame(index):
            nonlocal validation_seconds
            if reference_trajectory is None:
                return
            start = time.perf_counter()
            arrays = (pipeline.objects, pipeline.masks, pipeline.globals,
                      pipeline.regrets, pipeline.collisions, pipeline.teacher_actions)
            if capture_reference:
                reference_trajectory.append(tuple(array.copy() for array in arrays))
            else:
                _check_arrays(reference_trajectory[index], arrays, index)
            validation_seconds += time.perf_counter() - start
        check_frame(0)
        for step in range(decisions):
            active_track_counts.extend(
                np.count_nonzero(pipeline.masks, axis=1).astype(int).tolist()
            )
            actions = pipeline.teacher_actions.copy()
            pipeline.step(actions)
            check_frame(step + 1)
        wall_elapsed = time.perf_counter() - stepped
        elapsed = wall_elapsed - validation_seconds
    active_tracks = np.asarray(active_track_counts, dtype=np.int32)
    return {
        "workers": workers,
        "env_count": env_count,
        "decisions": decisions,
        "teacher_kind": teacher_kind,
        "teacher_reaction_seconds": float(teacher_reaction_seconds),
        "feature_expected_bullet_count": (bullet_count if feature_expected_bullet_count is None
                                          else int(feature_expected_bullet_count)),
        "episode_limit_seconds": float(episode_limit_seconds),
        "observation_size": int(observation_size),
        "bullets": int(bullet_count),
        "tracked_objects": int(tracking_capacity),
        "deployment_rgb_observation": bool(deployment_rgb_observation),
        "causal_action_delay_steps": int(causal_action_delay_steps),
        "action_repeat_choices": list(action_repeat_choices),
        "initial_episode_seeds": list(initial_episode_seeds),
        "repeat_initial_episode_seeds": bool(repeat_initial_episode_seeds),
        "startup_seconds": startup,
        "step_seconds": elapsed,
        "step_wall_seconds": wall_elapsed,
        "direct_content_validation_seconds": validation_seconds,
        "states_per_second": env_count * decisions / elapsed,
        "active_tracks_mean": float(active_tracks.mean()),
        "active_tracks_p95": float(np.percentile(active_tracks, 95.0)),
        "active_tracks_max": int(active_tracks.max()),
        "capacity_hit_fraction": float(
            np.mean(active_tracks >= tracking_capacity)
        ),
        "trajectory_validation": ('reference_captured_in_memory' if capture_reference
            else 'all_arrays_bitwise_equal' if reference_trajectory is not None else 'not_requested'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=(4, 6, 8, 9, 10))
    parser.add_argument("--env-count", type=int, default=36)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument(
        "--teacher-kind", choices=("exact", "recovery"), default="recovery"
    )
    parser.add_argument("--observation-size", type=int, default=192)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument("--tracking-capacity", type=int, default=None)
    parser.add_argument("--teacher-reaction-seconds", type=float, default=0.10)
    parser.add_argument("--feature-expected-bullets", type=int, default=None,
                        help="Match the measured checkpoint's feature normalization")
    parser.add_argument("--episode-limit-seconds", type=float, default=120.0)
    parser.add_argument(
        "--causal-action-delay-steps", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--deployment-rgb-observation", action="store_true")
    parser.add_argument("--action-repeat-choices", default="")
    parser.add_argument("--initial-episode-seeds", default="")
    parser.add_argument("--repeat-initial-episode-seeds", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if min(args.env_count, args.decisions, args.bullets, *args.workers) <= 0:
        parser.error('Counts must be positive')
    action_repeat_choices = tuple(
        int(value.strip())
        for value in args.action_repeat_choices.split(",")
        if value.strip()
    )
    initial_episode_seeds = tuple(
        int(value.strip())
        for value in args.initial_episode_seeds.split(",")
        if value.strip()
    )
    reference_trajectory = []
    rows = [
        benchmark(
            workers,
            args.env_count,
            args.decisions,
            args.teacher_kind,
            args.observation_size,
            args.deployment_rgb_observation,
            action_repeat_choices,
            args.bullets,
            args.causal_action_delay_steps,
            initial_episode_seeds,
            args.repeat_initial_episode_seeds,
            args.tracking_capacity,
            reference_trajectory,
            args.teacher_reaction_seconds,
            args.feature_expected_bullets,
            args.episode_limit_seconds,
        )
        for workers in args.workers
    ]
    if args.output:
        atomic_write_json(Path(args.output), rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
