"""Run exact-teacher random-seed tests at 300, then 350 bullets."""

from __future__ import annotations

import argparse
import csv
import io
import secrets
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from barrage_rl.artifacts import atomic_write_json, atomic_write_text, prepare_new_output
from barrage_rl.baselines import privileged_planner_supervision
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.evaluate_teacher import _wall_distance
from barrage_rl.metrics import wilson_lower_bound
from barrage_rl.task_spec import BarrageTaskSpec
from barrage_rl.timing import DECISION_DT


BULLET_COUNTS = (300, 350)
TARGETED_BULLET_PROBABILITY = 0.10
CSV_FIELDS = (
    "stage",
    "bullet_count",
    "global_episode",
    "episode",
    "seed",
    "passed",
    "teacher_survival_seconds",
    "termination_reason",
    "steps",
    "minimum_wall_distance",
    "death_plane_center_x",
    "death_plane_center_y",
    "death_action",
    "collision_bullet_count",
    "collision_bullet_indices",
    "collision_bullet_index",
    "collision_bullet_x",
    "collision_bullet_y",
    "collision_bullet_vx",
    "collision_bullet_vy",
    "collision_bullet_targeted",
)


def generate_unique_seeds(count: int, master_seed: int) -> list[int]:
    """Generate deterministic, randomly ordered, unique signed-32-bit seeds."""
    if count <= 0:
        raise ValueError("episodes must be positive")
    if count > np.iinfo(np.int32).max:
        raise ValueError("episode count exceeds the available seed range")
    rng = np.random.default_rng(int(master_seed))
    seeds: set[int] = set()
    while len(seeds) < count:
        needed = count - len(seeds)
        draws = rng.integers(
            0, np.iinfo(np.int32).max, size=max(needed, 16), dtype=np.int64
        )
        seeds.update(int(value) for value in draws)
    return list(seeds)[:count]


def _evaluate_episode(task: dict[str, Any]) -> dict[str, Any]:
    env = BarrageVisionEnv(**task["env_kwargs"])
    episode = int(task["episode"])
    seed = int(task["seed"])
    minimum_wall_distance = float("inf")
    steps = 0
    last_action = 0
    try:
        env.reset(seed=seed)
        while True:
            minimum_wall_distance = min(minimum_wall_distance, _wall_distance(env))
            supervision = privileged_planner_supervision(
                env,
                horizon_seconds=float(task["teacher_horizon_seconds"]),
                reaction_seconds=DECISION_DT,
                wall_margin=80.0,
                wall_penalty_weight=0.35,
                safety_horizons=(0.10,),
            )
            last_action = int(supervision.action)
            steps += 1
            _, _, terminated, truncated, info = env.step(last_action)
            if not (terminated or truncated):
                continue

            row: dict[str, Any] = {
                "stage": int(task["stage"]),
                "bullet_count": int(task["bullet_count"]),
                "global_episode": int(task["global_episode"]),
                "episode": episode,
                "seed": seed,
                "passed": int(bool(truncated)),
                "teacher_survival_seconds": float(info["survival_seconds"]),
                "termination_reason": "collision" if terminated else "time_limit",
                "steps": steps,
                "minimum_wall_distance": minimum_wall_distance,
                "death_plane_center_x": "",
                "death_plane_center_y": "",
                "death_action": "",
                "collision_bullet_count": 0,
                "collision_bullet_indices": "",
                "collision_bullet_index": "",
                "collision_bullet_x": "",
                "collision_bullet_y": "",
                "collision_bullet_vx": "",
                "collision_bullet_vy": "",
                "collision_bullet_targeted": "",
            }
            if terminated:
                colliding = env.colliding_bullet_indices().astype(np.int64)
                row.update({
                    "death_plane_center_x": float(env.plane_position[0]),
                    "death_plane_center_y": float(env.plane_position[1]),
                    "death_action": last_action,
                    "collision_bullet_count": int(len(colliding)),
                    "collision_bullet_indices": ";".join(
                        str(int(index)) for index in colliding
                    ),
                })
                if len(colliding):
                    offsets = env.bullet_positions[colliding] - env.plane_position
                    index = int(colliding[np.argmin(np.linalg.norm(offsets, axis=1))])
                    row.update({
                        "collision_bullet_index": index,
                        "collision_bullet_x": float(env.bullet_positions[index, 0]),
                        "collision_bullet_y": float(env.bullet_positions[index, 1]),
                        "collision_bullet_vx": float(env.bullet_velocities[index, 0]),
                        "collision_bullet_vy": float(env.bullet_velocities[index, 1]),
                        "collision_bullet_targeted": int(
                            env.bullet_is_targeted[index]
                        ),
                    })
            return row
    finally:
        env.close()


def episode_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: int(row["global_episode"])))
    return buffer.getvalue()


def run_batch(
    *,
    episodes_per_stage: int,
    workers: int,
    output_dir: Path,
    master_seed: int | None = None,
    episode_limit_seconds: float = 120.0,
    smoke_test: bool = False,
    progress_every: int = 1,
) -> dict[str, Any]:
    if episodes_per_stage <= 0 or workers <= 0:
        raise ValueError("episodes_per_stage and workers must be positive")
    if episode_limit_seconds <= 0.0:
        raise ValueError("episode_limit_seconds must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if not smoke_test and episode_limit_seconds != 120.0:
        raise ValueError("full batch tests are locked to 120 seconds")

    resolved_master_seed = (
        secrets.randbits(63) if master_seed is None else int(master_seed)
    )
    total_episodes = episodes_per_stage * len(BULLET_COUNTS)
    # Generate one held-out set and reuse it in every bullet-count stage so
    # per-seed outcomes remain directly comparable across task difficulty.
    seeds = generate_unique_seeds(episodes_per_stage, resolved_master_seed)
    worker_count = min(int(workers), episodes_per_stage)
    settings = {
        "episodes_per_stage": int(episodes_per_stage),
        "total_episodes": int(total_episodes),
        "workers": worker_count,
        "master_seed": resolved_master_seed,
        "shared_episode_seeds": seeds,
        "reuse_seeds_across_stages": True,
        "teacher_kind": "exact",
        "teacher_horizon_seconds": 1.5,
        "episode_limit_seconds": float(episode_limit_seconds),
        "bullet_counts": list(BULLET_COUNTS),
        "targeted_bullet_probability": TARGETED_BULLET_PROBABILITY,
        "smoke_test": bool(smoke_test),
        "progress_every": int(progress_every),
    }
    prepare_new_output(output_dir)
    atomic_write_json(output_dir / "config.json", settings)
    rows: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for stage_index, bullet_count in enumerate(BULLET_COUNTS, start=1):
        global_episode_start = (stage_index - 1) * episodes_per_stage
        env_kwargs = BarrageTaskSpec(
            bullet_count=bullet_count,
            targeted_bullet_probability=TARGETED_BULLET_PROBABILITY,
            episode_limit_seconds=float(episode_limit_seconds),
        ).env_kwargs()
        tasks = [
            {
                "stage": stage_index,
                "bullet_count": bullet_count,
                "global_episode": global_episode_start + episode,
                "episode": episode,
                "seed": seed,
                "env_kwargs": env_kwargs,
                "teacher_horizon_seconds": 1.5,
            }
            for episode, seed in enumerate(seeds)
        ]
        stage_rows: list[dict[str, Any]] = []
        stage_started = time.perf_counter()
        print(
            f"exact_teacher_stage_start stage={stage_index} "
            f"bullets={bullet_count} episodes={episodes_per_stage} "
            f"workers={worker_count}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_evaluate_episode, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                stage_rows.append(row)
                rows.append(row)
                completed = len(stage_rows)
                if completed % progress_every == 0 or completed == episodes_per_stage:
                    atomic_write_text(
                        output_dir / "teacher_episodes.partial.csv",
                        episode_csv(rows),
                    )
                    successes = sum(int(item["passed"]) for item in stage_rows)
                    stage_elapsed = time.perf_counter() - stage_started
                    stage_rate = completed / max(stage_elapsed, 1e-9)
                    stage_eta = (episodes_per_stage - completed) / stage_rate
                    total_completed = global_episode_start + completed
                    print(
                        f"exact_teacher_stage_progress stage={stage_index} "
                        f"bullets={bullet_count} "
                        f"completed={completed}/{episodes_per_stage} "
                        f"stage_progress={100.0 * completed / episodes_per_stage:.1f}% "
                        f"overall={total_completed}/{total_episodes} "
                        f"overall_progress={100.0 * total_completed / total_episodes:.1f}% "
                        f"passed={successes}/{completed} "
                        f"elapsed={stage_elapsed:.1f}s "
                        f"stage_eta={stage_eta:.1f}s",
                        flush=True,
                    )
        stage_elapsed = time.perf_counter() - stage_started
        stage_successes = sum(int(row["passed"]) for row in stage_rows)
        stage_summary = {
            "stage": stage_index,
            "bullet_count": bullet_count,
            "episodes": episodes_per_stage,
            "passed": stage_successes,
            "failed": episodes_per_stage - stage_successes,
            "success_at_limit": stage_successes / episodes_per_stage,
            "success_at_limit_ci95_low": wilson_lower_bound(
                stage_successes, episodes_per_stage
            ),
            "mean_survival_seconds": float(np.mean([
                float(row["teacher_survival_seconds"]) for row in stage_rows
            ])),
            "elapsed_seconds": stage_elapsed,
            "episodes_per_second": episodes_per_stage / stage_elapsed,
        }
        stage_summaries.append(stage_summary)
        atomic_write_text(
            output_dir / f"teacher_episodes_{bullet_count}.csv",
            episode_csv(stage_rows),
        )
        atomic_write_json(
            output_dir / f"summary_{bullet_count}.json", stage_summary
        )

    elapsed = time.perf_counter() - started
    successes = sum(int(row["passed"]) for row in rows)
    summary = {
        **settings,
        "stages": stage_summaries,
        "passed": successes,
        "failed": total_episodes - successes,
        "success_at_limit": successes / total_episodes,
        "success_at_limit_ci95_low": wilson_lower_bound(
            successes, total_episodes
        ),
        "mean_survival_seconds": float(np.mean([
            float(row["teacher_survival_seconds"]) for row in rows
        ])),
        "elapsed_seconds": elapsed,
        "episodes_per_second": total_episodes / elapsed,
    }
    atomic_write_text(output_dir / "teacher_episodes.csv", episode_csv(rows))
    atomic_write_json(output_dir / "summary.json", summary)
    print(
        f"exact_teacher_300_350_result passed={successes}/{total_episodes} "
        f"rate={100.0 * summary['success_at_limit']:.2f}% "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-test the exact teacher on unique random seeds: "
            "300 bullets first, then 350 bullets"
        )
    )
    parser.add_argument("--episodes-per-stage", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "output directory; defaults to a unique timestamped directory "
            "under diagnostics"
        ),
    )
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--episode-limit-seconds", type=float, default=120.0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="print and save progress after this many completed episodes",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path(
            f"diagnostics/exact_teacher_300_350_{timestamp}"
        )
    print(f"exact_teacher_output_dir path={output_dir.resolve()}", flush=True)
    run_batch(
        episodes_per_stage=args.episodes_per_stage,
        workers=args.workers,
        output_dir=output_dir,
        master_seed=args.master_seed,
        episode_limit_seconds=args.episode_limit_seconds,
        smoke_test=args.smoke_test,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
