"""Replay recorded collision seeds and save collision positions and lookback paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from barrage_rl.evaluate_tracked_policy import (  # noqa: E402
    checkpoint_action_delay_steps,
    load_tracked_agent,
)
from barrage_rl.parallel_evaluation import run_parallel_rollout  # noqa: E402
from barrage_rl.task_spec import (  # noqa: E402
    PRODUCTION_ANALYTIC_SHIELD,
    PRODUCTION_ANALYTIC_SHIELD_GATE,
    BarrageTaskSpec,
    TARGET_TASK,
)
from barrage_rl.timing import DECISION_DT  # noqa: E402


DEFAULT_SOURCE = PROJECT_ROOT / "diagnostics" / "short_risk_episodes_1000seeds.csv"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "diagnostics"
    / "risk_threshold_0p10691_fixed200_old"
    / "threshold_0.10691.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "diagnostics" / "risk_threshold_0p10691_random22_position_replay_300"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
    temporary.replace(path)


def _wall_columns(item: dict[str, Any]) -> dict[str, float]:
    center = item["plane_center"]
    wall = item["wall_clearance_pixels"]
    return {
        "plane_center_x": float(center[0]),
        "plane_center_y": float(center[1]),
        "wall_left_clearance": float(wall["left"]),
        "wall_right_clearance": float(wall["right"]),
        "wall_top_clearance": float(wall["top"]),
        "wall_bottom_clearance": float(wall["bottom"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--lookback-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.source.open(newline="", encoding="utf-8-sig") as file:
        source_rows = list(csv.DictReader(file))
    collision_rows = [
        row for row in source_rows if row["termination_reason"] == "collision"
    ]
    seeds = [int(row["seed"]) for row in collision_rows]
    if len(seeds) != 22 or len(set(seeds)) != 22:
        raise ValueError(
            f"expected 22 unique collision seeds, found {len(seeds)} rows and "
            f"{len(set(seeds))} unique seeds"
        )

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    agent, spec, checkpoint = load_tracked_agent(
        str(args.checkpoint),
        device,
        analytic_shield=PRODUCTION_ANALYTIC_SHIELD,
        analytic_shield_gate=PRODUCTION_ANALYTIC_SHIELD_GATE,
    )
    expected_threshold = 0.10691
    if not np.isclose(
        agent.safety_threshold, expected_threshold, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"checkpoint safety threshold is {agent.safety_threshold}, "
            f"expected {expected_threshold}"
        )

    task = BarrageTaskSpec(
        bullet_count=TARGET_TASK.bullet_count,
        targeted_bullet_probability=TARGET_TASK.targeted_bullet_probability,
        observation_size=int(checkpoint.get("observation_size", 384)),
        episode_limit_seconds=120.0,
    )
    lookback_decisions = int(math.ceil(args.lookback_seconds / DECISION_DT))
    rollout = run_parallel_rollout(
        agent=agent,
        spec=spec,
        episodes=len(seeds),
        workers=min(args.workers, len(seeds)),
        seed=seeds[0],
        episode_seeds=seeds,
        env_kwargs=task.env_kwargs(),
        wall_threshold=40.0,
        rendered_rgb=True,
        causal_action_delay_steps=checkpoint_action_delay_steps(checkpoint),
        collect_failure_diagnostics=True,
        failure_lookback_decisions=lookback_decisions,
    )

    by_seed = {int(item["seed"]): item for item in rollout.failure_diagnostics}
    episode_rows: list[dict[str, Any]] = []
    for index, (source, seed, survival) in enumerate(
        zip(collision_rows, seeds, rollout.survival_times)
    ):
        episode_rows.append({
            "episode": index,
            "original_episode": int(source["episode"]),
            "seed": seed,
            "source_threshold": expected_threshold,
            "replay_threshold": expected_threshold,
            "source_survival_seconds": float(source["survival_seconds"]),
            "replay_survival_seconds": float(survival),
            "replay_termination_reason": (
                "collision" if seed in by_seed else "time_limit"
            ),
            "minimum_wall_clearance_during_episode": float(
                rollout.minimum_wall_distances[index]
            ),
        })

    collision_output_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for failure in rollout.failure_diagnostics:
        spatial = _wall_columns(failure)
        collision_output_rows.append({
            "seed": int(failure["seed"]),
            "survival_seconds": float(failure["survival_seconds"]),
            "safety_threshold": float(failure["safety_threshold"]),
            **spatial,
            "collision_bullet_count": int(failure["collision_bullet_count"]),
            "collision_targeted_any": bool(failure["collision_targeted_any"]),
        })
        for point in failure.get("backtrace", []):
            trajectory_rows.append({
                "seed": int(failure["seed"]),
                "decision_index": int(point["decision_index"]),
                "survival_seconds": float(point["survival_seconds"]),
                "seconds_before_death": float(point["seconds_before_death"]),
                **_wall_columns(point),
                "raw_policy_action": int(point["raw_policy_action"]),
                "learned_filter_action": int(point["learned_filter_action"]),
                "executed_action": int(point["executed_action"]),
                "raw_immediate_risk": float(point["raw_immediate_risk"]),
                "selected_immediate_risk": float(point["selected_immediate_risk"]),
                "action_was_filtered": bool(point["action_was_filtered"]),
                "all_actions_unsafe": bool(point["all_actions_unsafe"]),
                "tracked_count_ratio": float(point["tracked_count_ratio"]),
                "mean_position_uncertainty": float(
                    point["mean_position_uncertainty"]
                ),
            })

    _write_csv(
        args.output_dir / "episodes.csv",
        list(episode_rows[0]),
        episode_rows,
    )
    collision_fields = [
        "seed", "survival_seconds", "safety_threshold",
        "plane_center_x", "plane_center_y",
        "wall_left_clearance", "wall_right_clearance",
        "wall_top_clearance", "wall_bottom_clearance",
        "collision_bullet_count", "collision_targeted_any",
    ]
    _write_csv(
        args.output_dir / "collision_positions.csv",
        collision_fields,
        collision_output_rows,
    )
    trajectory_fields = [
        "seed", "decision_index", "survival_seconds", "seconds_before_death",
        "plane_center_x", "plane_center_y",
        "wall_left_clearance", "wall_right_clearance",
        "wall_top_clearance", "wall_bottom_clearance",
        "raw_policy_action", "learned_filter_action", "executed_action",
        "raw_immediate_risk", "selected_immediate_risk",
        "action_was_filtered", "all_actions_unsafe", "tracked_count_ratio",
        "mean_position_uncertainty",
    ]
    _write_csv(
        args.output_dir / "pre_collision_trajectory.csv",
        trajectory_fields,
        trajectory_rows,
    )

    report = {
        "measurement_scope": "New current-task rollout on recorded collision seeds; source survival columns retain original values",
        "source_environment_equivalence_verified": False,
        "source": str(args.source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "safety_threshold": expected_threshold,
        "use_safety_filter": bool(agent.use_safety_filter),
        "episode_limit_seconds": 120.0,
        "bullet_count": TARGET_TASK.bullet_count,
        "targeted_bullet_probability": TARGET_TASK.targeted_bullet_probability,
        "rendered_rgb": True,
        "requested_seed_count": len(seeds),
        "unique_seed_count": len(set(seeds)),
        "replay_collision_count": len(rollout.failure_diagnostics),
        "replay_success_count": len(seeds) - len(rollout.failure_diagnostics),
        "lookback_seconds": float(args.lookback_seconds),
        "lookback_decisions": lookback_decisions,
        "failures": rollout.failure_diagnostics,
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps({key: report[key] for key in (
        "safety_threshold", "requested_seed_count", "unique_seed_count",
        "replay_collision_count", "replay_success_count", "lookback_seconds",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
