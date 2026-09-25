"""Evaluate the strict image-input model-based oracle on fixed episodes."""

from __future__ import annotations

from .timing import PHYSICS_FPS

import argparse
import csv
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, atomic_write_text, prepare_new_output
from .env import BarrageVisionEnv
from .metrics import (
    bootstrap_confidence_intervals,
    interquartile_mean,
    lower_tail_mean,
    wilson_lower_bound,
)
from .task_spec import BarrageTaskSpec, TARGET_TASK
from .image_oracle import ImageOnlyPlannerAgent


def _wall_distance(env: BarrageVisionEnv) -> float:
    half_plane = env.plane_size / 2.0
    return min(
        float(env.plane_position[0] - half_plane[0]),
        float(env.screen_width - half_plane[0] - env.plane_position[0]),
        float(env.plane_position[1] - half_plane[1]),
        float(env.screen_height - half_plane[1] - env.plane_position[1]),
    )


def _episode(task: dict[str, Any]) -> dict[str, Any]:
    episode = int(task["episode"])
    seed = int(task["seed"]) + episode
    env = BarrageVisionEnv(**task["env_kwargs"])
    agent = ImageOnlyPlannerAgent(
        observation_size=int(task["observation_size"]),
        bullet_count=int(task["env_kwargs"]["bullet_count"]),
        bullet_size=int(task["env_kwargs"]["bullet_size_min"]),
        bullet_speed=float(task["env_kwargs"]["bullet_speed_min"]),
        horizon_seconds=float(task["horizon_seconds"]),
        planner_kind=str(task["planner_kind"]),
    )
    minimum_wall_distance = float("inf")
    wall_steps = 0
    steps = 0
    known_velocity_sum = 0.0
    try:
        observation, _ = env.reset(seed=seed)
        agent.reset()
        while True:
            distance = _wall_distance(env)
            minimum_wall_distance = min(minimum_wall_distance, distance)
            wall_steps += int(distance < 40.0)
            action = agent.act(observation)
            known_velocity_sum += agent.tracker.known_velocity_fraction
            steps += 1
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return {
                    "episode": episode,
                    "seed": seed,
                    "survival_seconds": float(info["survival_seconds"]),
                    "termination_reason": "collision" if terminated else "time_limit",
                    "minimum_wall_distance": minimum_wall_distance,
                    "wall_steps": wall_steps,
                    "steps": steps,
                    "mean_known_velocity_fraction": known_velocity_sum / max(steps, 1),
                }
    finally:
        agent.close()
        env.close()


def _rows_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "episode", "seed", "survival_seconds", "termination_reason",
            "minimum_wall_distance", "wall_steps", "steps",
            "mean_known_velocity_fraction",
        ),
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: int(row["episode"])))
    return buffer.getvalue()


def evaluate_image_oracle(
    *,
    episodes: int = 200,
    workers: int = 8,
    seed: int = 1_900_000,
    output_dir: str = "",
    observation_size: int = 192,
    horizon_seconds: float = 1.5,
    planner_kind: str = "recovery",
    episode_limit_seconds: float = 120.0,
    bullet_count: int = TARGET_TASK.bullet_count,
    targeted_bullet_probability: float = TARGET_TASK.targeted_bullet_probability,
    smoke_test: bool = False,
    resume: bool = False,
) -> dict[str, float]:
    if episodes != 200 and not smoke_test:
        raise ValueError("image-oracle evaluation requires exactly 200 episodes")
    if not smoke_test and episode_limit_seconds != 120.0:
        raise ValueError("production image-oracle evaluation is locked to 120 seconds")
    if observation_size < 96:
        raise ValueError("observation_size must be at least 96")
    if planner_kind not in ("recovery", "exact"):
        raise ValueError("planner_kind must be 'recovery' or 'exact'")
    output = Path(output_dir) if output_dir else None
    if resume and output is None:
        raise ValueError("image-oracle resume requires output_dir")
    settings: dict[str, Any] = {
        "episodes": episodes,
        "workers": max(1, min(int(workers), episodes)),
        "seed": seed,
        "observation_size": int(observation_size),
        "horizon_seconds": float(horizon_seconds),
        "planner_kind": planner_kind,
        "episode_limit_seconds": float(episode_limit_seconds),
        "physics_fps": PHYSICS_FPS,
        "env_kwargs": BarrageTaskSpec(
            bullet_count=int(bullet_count),
            targeted_bullet_probability=float(targeted_bullet_probability),
            observation_size=int(observation_size),
            episode_limit_seconds=float(episode_limit_seconds),
        ).env_kwargs(),
    }
    rows: list[dict[str, Any]] = []
    if output is not None:
        if resume:
            config_path = output / "config.json"
            partial_path = output / "episodes.partial.csv"
            if not config_path.exists() or not partial_path.exists():
                raise FileNotFoundError(
                    "image-oracle resume requires config.json and episodes.partial.csv"
                )
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            for name in (
                "episodes", "seed", "observation_size", "horizon_seconds",
                "planner_kind", "episode_limit_seconds", "env_kwargs",
            ):
                if saved[name] != settings[name]:
                    raise ValueError(
                        f"image-oracle resume configuration mismatch for {name}"
                    )
            with partial_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
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
    with ProcessPoolExecutor(max_workers=settings["workers"]) as executor:
        futures = [executor.submit(_episode, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0 or len(rows) == episodes:
                partial = np.asarray(
                    [row["survival_seconds"] for row in rows], np.float32
                )
                print(
                    "image_oracle_progress "
                    f"planner={planner_kind} "
                    f"completed={len(rows)}/{episodes} "
                    f"mean={float(partial.mean()):.2f}s "
            f"iqm={interquartile_mean(partial):.2f}s "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                if output is not None:
                    atomic_write_text(output / "episodes.partial.csv", _rows_csv(rows))
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: int(row["episode"]))
    times = np.asarray([row["survival_seconds"] for row in rows], np.float32)
    minimum_walls = np.asarray(
        [row["minimum_wall_distance"] for row in rows], np.float64
    )
    success_count = int(np.count_nonzero(times >= episode_limit_seconds))
    bootstrap = bootstrap_confidence_intervals(times, seed=seed + 9_000_000)
    result = {
        "episodes": float(episodes),
        "seed": float(seed),
        "observation_size": float(observation_size),
        "planner_kind": planner_kind,
        "model_mean": float(times.mean()),
        "model_median": float(np.median(times)),
        "model_iqm": interquartile_mean(times),
        "model_p10": float(np.percentile(times, 10.0)),
        "model_p1": float(np.percentile(times, 1.0)),
        "model_p5": float(np.percentile(times, 5.0)),
        "model_cvar1": lower_tail_mean(times, 0.01),
        "model_cvar5": lower_tail_mean(times, 0.05),
        "model_cvar90": lower_tail_mean(times, 0.90),
        "model_rmst": float(np.minimum(times, episode_limit_seconds).mean()),
        "episode_limit_seconds": float(episode_limit_seconds),
        "physics_fps": PHYSICS_FPS,
        "success_at_limit": float(success_count / episodes),
        "success_at_limit_ci95_low": wilson_lower_bound(success_count, episodes),
        "failure_before_1s": float(np.mean(times < 1.0)),
        "failure_before_10s": float(np.mean(times < 10.0)),
        "bullets": float(bullet_count),
        "targeted_bullet_probability": float(targeted_bullet_probability),
        "wall_step_fraction": float(
            sum(int(row["wall_steps"]) for row in rows)
            / max(1, sum(int(row["steps"]) for row in rows))
        ),
        "wall_episode_fraction": float(np.mean(minimum_walls < 40.0)),
        "median_min_wall_distance": float(np.median(minimum_walls)),
        "mean_known_velocity_fraction": float(np.mean([
            float(row["mean_known_velocity_fraction"]) for row in rows
        ])),
        "model_mean_ci95_low": bootstrap["model_mean_ci95_low"],
        "model_mean_ci95_high": bootstrap["model_mean_ci95_high"],
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed,
    }
    print(
        "image_oracle_evaluation "
        f"planner={planner_kind} "
        f"episodes={episodes} mean={result['model_mean']:.2f}s "
        f"median={result['model_median']:.2f}s iqm={result['model_iqm']:.2f}s "
        f"success={100.0 * result['success_at_limit']:.1f}% "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    if output is not None:
        atomic_write_json(output / "summary.json", result)
        atomic_write_text(output / "episodes.csv", _rows_csv(rows))
    return result


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the image-only oracle")
    parser.add_argument("--episodes", type=int, default=200, choices=(200,))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1_900_000)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--observation-size", type=int, default=192)
    parser.add_argument("--horizon-seconds", type=float, default=1.5)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument(
        "--targeted-probability",
        type=float,
        default=TARGET_TASK.targeted_bullet_probability,
    )
    parser.add_argument(
        "--planner-kind", choices=("recovery", "exact"), default="recovery"
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    evaluate_image_oracle(
        episodes=2 if args.smoke_test else args.episodes,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        observation_size=args.observation_size,
        horizon_seconds=args.horizon_seconds,
        planner_kind=args.planner_kind,
        episode_limit_seconds=5.0 if args.smoke_test else 120.0,
        bullet_count=args.bullets,
        targeted_bullet_probability=args.targeted_probability,
        smoke_test=args.smoke_test,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
