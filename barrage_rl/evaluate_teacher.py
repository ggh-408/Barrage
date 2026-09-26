"""Evaluate the stage-one DAgger teacher on fixed held-out episodes."""

from __future__ import annotations

from .timing import DECISION_DT, PHYSICS_FPS

import argparse
import csv
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .artifacts import atomic_write_json, atomic_write_text, prepare_new_output
from .baselines import fast_planner_supervision, privileged_planner_supervision
from .env import BarrageVisionEnv
from .recovery_planner import vectorized_recovery_supervision
from .metrics import (
    bootstrap_confidence_intervals,
    interquartile_mean,
    lower_tail_mean,
    wilson_lower_bound,
)
from .task_spec import BarrageTaskSpec, TARGET_TASK


def _wall_distance(env: BarrageVisionEnv) -> float:
    half_plane = env.plane_size / 2.0
    return min(
        float(env.plane_position[0] - half_plane[0]),
        float(env.screen_width - half_plane[0] - env.plane_position[0]),
        float(env.plane_position[1] - half_plane[1]),
        float(env.screen_height - half_plane[1] - env.plane_position[1]),
    )


def _evaluate_episode(task: Dict[str, Any]) -> Dict[str, Any]:
    episode = int(task["episode"])
    seed = int(task["seed"]) + episode
    wall_threshold = float(task["wall_threshold"])
    env = BarrageVisionEnv(**task["env_kwargs"])
    minimum_wall_distance = float("inf")
    wall_steps = 0
    steps = 0
    sequence_search_calls = 0
    sequence_nodes_expanded = 0
    maximum_sequence_beam_width = 0
    action_histogram = np.zeros(len(env.ACTIONS), dtype=np.int64)
    try:
        env.reset(seed=seed)
        while True:
            distance = _wall_distance(env)
            minimum_wall_distance = min(minimum_wall_distance, distance)
            wall_steps += int(distance < wall_threshold)
            planner = {
                "fast": fast_planner_supervision,
                "recovery": vectorized_recovery_supervision,
                "exact": privileged_planner_supervision,
            }[task["teacher_kind"]]
            supervision = planner(
                env,
                horizon_seconds=float(task["teacher_horizon_seconds"]),
                reaction_seconds=float(task["teacher_reaction_seconds"]),
                wall_margin=float(task["teacher_wall_margin"]),
                wall_penalty_weight=float(task["teacher_wall_penalty_weight"]),
                safety_horizons=tuple(task["safety_horizons"]),
            )
            sequence_search_calls += int(supervision.used_sequence_search)
            sequence_nodes_expanded += int(supervision.sequence_nodes_expanded)
            maximum_sequence_beam_width = max(
                maximum_sequence_beam_width,
                int(supervision.sequence_beam_width),
            )
            action = int(supervision.action)
            action_histogram[action] += 1
            steps += 1
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return {
                    "episode": episode,
                    "seed": seed,
                    "teacher_survival_seconds": float(info["survival_seconds"]),
                    "termination_reason": "collision" if terminated else "time_limit",
                    "minimum_wall_distance": minimum_wall_distance,
                    "wall_steps": wall_steps,
                    "steps": steps,
                    "action_histogram": action_histogram.tolist(),
                    "sequence_search_calls": sequence_search_calls,
                    "sequence_nodes_expanded": sequence_nodes_expanded,
                    "maximum_sequence_beam_width": maximum_sequence_beam_width,
                }
    finally:
        env.close()


def _episode_csv(rows: list[Dict[str, Any]]) -> str:
    episode_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        episode_buffer,
        fieldnames=(
            "episode", "seed", "teacher_survival_seconds",
            "termination_reason", "minimum_wall_distance", "wall_steps", "steps",
        ),
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: int(row["episode"])))
    return episode_buffer.getvalue()


def evaluate_teacher(
    *,
    episodes: int = 200,
    workers: int = 8,
    seed: int = 1_600_000,
    output_dir: str = "",
    smoke_test: bool = False,
    capacity_test: bool = False,
    teacher_kind: str = "fast",
    teacher_horizon_seconds: float | None = None,
    episode_limit_seconds: float = 120.0,
    bullet_count: int = TARGET_TASK.bullet_count,
    targeted_bullet_probability: float = TARGET_TASK.targeted_bullet_probability,
    resume: bool = False,
) -> Dict[str, float]:
    if smoke_test and capacity_test:
        raise ValueError("smoke_test and capacity_test are mutually exclusive")
    if capacity_test and episodes != 100:
        raise ValueError("capacity teacher evaluation requires exactly 100 episodes")
    if not capacity_test and episodes != 200 and not smoke_test:
        raise ValueError("teacher evaluation requires exactly 200 episodes")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episode_limit_seconds <= 0:
        raise ValueError("episode_limit_seconds must be positive")
    if (capacity_test or not smoke_test) and episode_limit_seconds != 120.0:
        raise ValueError("production teacher evaluation is locked to 120 seconds")
    if teacher_kind not in ("fast", "recovery", "exact"):
        raise ValueError("teacher_kind must be 'fast', 'recovery', or 'exact'")
    if teacher_horizon_seconds is None:
        teacher_horizon_seconds = 0.60 if teacher_kind == "fast" else 1.50
    if teacher_horizon_seconds <= 0:
        raise ValueError("teacher_horizon_seconds must be positive")

    output = Path(output_dir) if output_dir else None

    settings: Dict[str, Any] = {
        "episodes": episodes,
        "evaluation_mode": "capacity" if capacity_test else (
            "smoke" if smoke_test else "production"
        ),
        "workers": max(1, min(int(workers), episodes)),
        "seed": seed,
        "teacher_kind": teacher_kind,
        "wall_threshold": 40.0,
        "teacher_horizon_seconds": float(teacher_horizon_seconds),
        "teacher_reaction_seconds": DECISION_DT,
        "teacher_wall_margin": 80.0,
        "teacher_wall_penalty_weight": 0.35,
        "safety_horizons": (0.10,),
        "env_kwargs": BarrageTaskSpec(
            bullet_count=int(bullet_count),
            targeted_bullet_probability=float(targeted_bullet_probability),
            episode_limit_seconds=float(episode_limit_seconds),
        ).env_kwargs(),
    }
    rows: list[Dict[str, Any]] = []
    if output is not None:
        if resume:
            config_path = output / "config.json"
            partial_path = output / "teacher_episodes.partial.csv"
            if not config_path.exists() or not partial_path.exists():
                raise FileNotFoundError(
                    "teacher resume requires config.json and teacher_episodes.partial.csv"
                )
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            for name in (
                "episodes", "evaluation_mode", "seed", "teacher_kind",
                "teacher_horizon_seconds",
                "teacher_reaction_seconds", "env_kwargs",
            ):
                if saved[name] != settings[name]:
                    raise ValueError(
                        f"teacher resume configuration mismatch for {name}"
                    )
            with partial_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            for row in rows:
                row["action_histogram"] = [0] * len(BarrageVisionEnv.ACTIONS)
        else:
            prepare_new_output(output)
            atomic_write_json(output / "config.json", settings)
    completed_episodes = {int(row["episode"]) for row in rows}
    tasks = [
        {**settings, "episode": episode}
        for episode in range(episodes)
        if episode not in completed_episodes
    ]
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=settings["workers"]) as executor:
            futures = [executor.submit(_evaluate_episode, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
                completed = len(rows)
                if completed % 10 == 0 or completed == episodes:
                    partial_times = np.asarray(
                        [row["teacher_survival_seconds"] for row in rows],
                        dtype=np.float32,
                    )
                    print(
                        "teacher_progress "
                        f"kind={teacher_kind} completed={completed}/{episodes} "
                        f"mean={float(partial_times.mean()):.2f}s "
            f"iqm={interquartile_mean(partial_times):.2f}s "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                    if output is not None:
                        atomic_write_text(
                            output / "teacher_episodes.partial.csv", _episode_csv(rows)
                        )
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: int(row["episode"]))

    times = np.asarray(
        [row["teacher_survival_seconds"] for row in rows], dtype=np.float32
    )
    minimum_wall_distances = np.asarray(
        [row["minimum_wall_distance"] for row in rows], dtype=np.float64
    )
    episode_limit = float(episode_limit_seconds)
    success_count = int(np.count_nonzero(times >= episode_limit))
    bootstrap = bootstrap_confidence_intervals(times, seed=seed + 9_000_000)
    result = {
        "episodes": float(episodes),
        "seed": float(seed),
        "teacher_kind": teacher_kind,
        "teacher_mean": float(times.mean()),
        "teacher_median": float(np.median(times)),
        "teacher_iqm": interquartile_mean(times),
        "teacher_p10": float(np.percentile(times, 10.0)),
        "teacher_p1": float(np.percentile(times, 1.0)),
        "teacher_p5": float(np.percentile(times, 5.0)),
        "teacher_cvar1": lower_tail_mean(times, 0.01),
        "teacher_cvar5": lower_tail_mean(times, 0.05),
        "teacher_cvar90": lower_tail_mean(times, 0.90),
        "teacher_rmst": float(np.minimum(times, episode_limit).mean()),
        "episode_limit_seconds": episode_limit,
        "physics_fps": PHYSICS_FPS,
        "bullet_size_min": 5.0,
        "bullet_size_max": 5.0,
        "bullet_speed_min": 240.0,
        "bullet_speed_max": 240.0,
        "success_at_limit": float(success_count / episodes),
        "success_at_limit_ci95_low": wilson_lower_bound(success_count, episodes),
        "failure_before_1s": float(np.mean(times < 1.0)),
        "failure_before_10s": float(np.mean(times < 10.0)),
        "bullets": float(bullet_count),
        "targeted_bullet_probability": float(targeted_bullet_probability),
        "wall_threshold": 40.0,
        "wall_step_fraction": float(
            sum(int(row["wall_steps"]) for row in rows)
            / max(1, sum(int(row["steps"]) for row in rows))
        ),
        "wall_episode_fraction": float(np.mean(minimum_wall_distances < 40.0)),
        "median_min_wall_distance": float(np.median(minimum_wall_distances)),
        "teacher_mean_ci95_low": bootstrap["model_mean_ci95_low"],
        "teacher_mean_ci95_high": bootstrap["model_mean_ci95_high"],
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed,
        "sequence_search_calls": float(sum(
            int(row.get("sequence_search_calls", 0)) for row in rows
        )),
        "sequence_search_fraction": float(
            sum(int(row.get("sequence_search_calls", 0)) for row in rows)
            / max(1, sum(int(row["steps"]) for row in rows))
        ),
        "sequence_nodes_expanded": float(sum(
            int(row.get("sequence_nodes_expanded", 0)) for row in rows
        )),
        "maximum_sequence_beam_width": float(max(
            (int(row.get("maximum_sequence_beam_width", 0)) for row in rows),
            default=0,
        )),
    }
    print(
        "teacher_evaluation "
        f"kind={teacher_kind} "
        f"episodes={episodes} mean={result['teacher_mean']:.2f}s "
        f"median={result['teacher_median']:.2f}s IQM={result['teacher_iqm']:.2f}s "
        f"P10={result['teacher_p10']:.2f}s "
        f"pass={100.0 * result['success_at_limit']:.2f}% "
        f"fail<10s={100.0 * result['failure_before_10s']:.2f}% "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )

    if output is not None:
        atomic_write_json(output / "teacher_summary.json", result)
        atomic_write_text(output / "teacher_episodes.csv", _episode_csv(rows))
        histogram = np.sum(
            np.asarray([row["action_histogram"] for row in rows], dtype=np.int64), axis=0
        )
        atomic_write_json(output / "action_histogram.json", histogram.tolist())
    return result


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the DAgger teacher")
    parser.add_argument("--episodes", type=int, default=200, choices=(100, 200))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1_600_000)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--teacher-kind", choices=("fast", "recovery", "exact"), default="fast"
    )
    parser.add_argument("--teacher-horizon-seconds", type=float, default=None)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument(
        "--targeted-probability",
        type=float,
        default=TARGET_TASK.targeted_bullet_probability,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument(
        "--capacity-test",
        action="store_true",
        help="diagnostic mode locked to 100 episodes and 120 seconds",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _cli_episode_count(args: argparse.Namespace) -> int:
    if args.smoke_test:
        return 2
    if args.capacity_test:
        if int(args.episodes) != 100:
            raise ValueError("--capacity-test requires --episodes 100")
        return 100
    if int(args.episodes) != 200:
        raise ValueError("production teacher evaluation requires --episodes 200")
    return 200


def _cli_episode_limit_seconds(args: argparse.Namespace) -> float:
    return 5.0 if args.smoke_test else 120.0


def main() -> None:
    args = _build_cli_parser().parse_args()
    evaluate_teacher(
        episodes=_cli_episode_count(args),
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        capacity_test=args.capacity_test,
        teacher_kind=args.teacher_kind,
        teacher_horizon_seconds=args.teacher_horizon_seconds,
        episode_limit_seconds=_cli_episode_limit_seconds(args),
        bullet_count=args.bullets,
        targeted_bullet_probability=args.targeted_probability,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
