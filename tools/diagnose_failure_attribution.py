"""Re-run recorded failures and attribute collisions without changing policy input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from barrage_rl.timing import DECISION_DT
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.parallel_evaluation import run_parallel_rollout
from barrage_rl.task_spec import BarrageTaskSpec


ATTRIBUTION_CSV_FIELDNAMES = (
    "original_episode_index",
    "seed",
    "survival_seconds",
    "targeted",
    "matched_track",
    "track_age_decisions",
    "track_age_seconds",
    "track_velocity_known",
    "track_history_length",
    "track_missed",
    "track_confidence",
    "track_position_uncertainty",
    "track_occluded_steps",
    "track_association_group_size",
    "match_distance_pixels",
    "raw_policy_action",
    "proposed_action",
    "executed_action",
    "action_was_filtered",
    "raw_immediate_collision_risk",
    "selected_immediate_collision_risk",
    "safety_threshold",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _write_attribution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ATTRIBUTION_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def diagnose(
    evaluation_dir: Path,
    output: Path,
    *,
    workers: int,
    device_name: str,
    lookback_seconds: float = 0.0,
    force_consensus_shield: bool = False,
    failure_seeds: tuple[int, ...] | None = None,
    decision_trace_seconds: float = 0.0,
) -> dict[str, Any]:
    config = json.loads(
        (evaluation_dir / "evaluation_config.json").read_text(encoding="utf-8")
    )
    episode_limit = float(config["episode_limit_seconds"])
    episode_csv = evaluation_dir / "evaluation_episodes.csv"
    if not episode_csv.exists():
        episode_csv = evaluation_dir / "evaluation_episodes.partial.csv"
    with episode_csv.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    recorded_failures = [
        row for row in rows
        if float(row["model_survival_seconds"]) < episode_limit
    ]
    if failure_seeds is not None:
        requested = tuple(int(seed) for seed in failure_seeds)
        available = {int(row["seed"]) for row in rows}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                f"requested seeds are not recorded episodes: {missing}"
            )
        requested_set = set(requested)
        selected_rows = [
            row for row in rows if int(row["seed"]) in requested_set
        ]
    else:
        if not recorded_failures:
            raise ValueError("recorded evaluation contains no failures to diagnose")
        selected_rows = recorded_failures
    targeted_probability = float(config["targeted_bullet_probability"])
    if targeted_probability != 0.10:
        raise ValueError("failure attribution is locked to targeted probability 0.10")
    if not bool(config.get("rendered_rgb", False)):
        raise ValueError("failure attribution requires the deployed rendered-RGB path")

    checkpoint = Path(config["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    actual_sha = _sha256(checkpoint)
    expected_sha = str(config.get("checkpoint_sha256", ""))
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("checkpoint SHA256 does not match the recorded evaluation")

    device = torch.device(
        device_name
        if device_name == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    agent, spec, checkpoint_data = load_tracked_agent(
        str(checkpoint),
        device,
        analytic_shield=(
            True if force_consensus_shield
            else bool(config.get("analytic_shield", False))
        ),
        analytic_guard_horizon_seconds=float(
            config.get("analytic_guard_horizon_seconds", 0.10)
        ),
        analytic_clearance_margin=float(
            config.get("analytic_clearance_margin", 0.0)
        ),
        analytic_shield_gate=(
            "learned_all_unsafe" if force_consensus_shield
            else str(config.get("analytic_shield_gate", "always"))
        ),
        analytic_min_tracked_count_ratio=float(
            config.get("analytic_min_tracked_count_ratio", 0.90)
        ),
        analytic_max_model_risk_increase=float(
            config.get("analytic_max_model_risk_increase", 0.005)
        ),
        analytic_max_selected_violation=float(
            config.get("analytic_max_selected_violation", 0.50)
        ),
    )
    task = BarrageTaskSpec(
        bullet_count=int(config["bullet_count"]),
        targeted_bullet_probability=targeted_probability,
        observation_size=int(checkpoint_data.get("observation_size", 384)),
        episode_limit_seconds=episode_limit,
    )
    seeds = [int(row["seed"]) for row in selected_rows]
    decision_trace: list[dict[str, Any]] = []
    if decision_trace_seconds > 0.0:
        if len(seeds) != 1:
            raise ValueError("decision tracing requires exactly one selected seed")
        original_select = agent._action_selector.select
        trace_limit = int(np.ceil(decision_trace_seconds / DECISION_DT))

        def traced_select(*args: Any, **kwargs: Any) -> Any:
            selection = original_select(*args, **kwargs)
            if len(decision_trace) < trace_limit:
                policy, teacher_cost, _, _, globals_, geometry = args
                decision_trace.append({
                    "decision_index": len(decision_trace),
                    "policy": policy[0].detach().cpu().tolist(),
                    "teacher_cost": teacher_cost[0].detach().cpu().tolist(),
                    "immediate_risk": selection.immediate_risk[0].cpu().tolist(),
                    "constant_clearance": (
                        geometry.normalized_minimum_clearance[0] * 100.0
                    ).cpu().tolist(),
                    "known_velocity_fraction": float(globals_[0, 8]),
                    "learned_action": int(selection.learned_actions[0]),
                    "selected_action": int(selection.actions[0]),
                    "counters": selection.counter_values.cpu().tolist(),
                })
            return selection

        agent._action_selector.select = traced_select
    rollout = run_parallel_rollout(
        agent=agent,
        spec=spec,
        episodes=len(seeds),
        workers=min(int(workers), len(seeds)),
        seed=seeds[0],
        episode_seeds=seeds,
        env_kwargs=task.env_kwargs(),
        wall_threshold=40.0,
        rendered_rgb=True,
        causal_action_delay_steps=int(config["causal_action_delay_steps"]),
        collect_failure_diagnostics=True,
        failure_lookback_decisions=int(np.ceil(max(lookback_seconds, 0.0) / DECISION_DT)),
    )

    reproduction = []
    for local_index, (row, actual) in enumerate(
        zip(selected_rows, rollout.survival_times)
    ):
        expected = float(row["model_survival_seconds"])
        reproduction.append({
            "local_episode_index": local_index,
            "original_episode_index": int(row["episode"]),
            "seed": int(row["seed"]),
            "expected_survival_seconds": expected,
            "actual_survival_seconds": float(actual),
            "absolute_error_seconds": abs(float(actual) - expected),
        })
    by_seed = {
        int(item["seed"]): item for item in rollout.failure_diagnostics
    }
    for item, source in zip(reproduction, selected_rows):
        diagnostic = by_seed.get(int(item["seed"]))
        if diagnostic is not None:
            diagnostic["original_episode_index"] = int(source["episode"])

    collision_bullets = [
        bullet
        for diagnostic in rollout.failure_diagnostics
        for bullet in diagnostic["collision_bullets"]
    ]
    matched = [item for item in collision_bullets if item["matched_track"]]
    targeted = [item for item in collision_bullets if item["targeted"]]
    untargeted = [item for item in collision_bullets if not item["targeted"]]
    targeted_ages = [
        float(item["track_age_decisions"])
        for item in targeted
        if item["track_age_decisions"] is not None
    ]
    untargeted_ages = [
        float(item["track_age_decisions"])
        for item in untargeted
        if item["track_age_decisions"] is not None
    ]
    report = {
        "kind": "recorded_episode_subset_attribution",
        "evaluation_dir": str(evaluation_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "bullet_count": int(config["bullet_count"]),
        "targeted_bullet_probability": targeted_probability,
        "rendered_rgb": True,
        "model_input": "current rendered RGB only",
        "diagnostic_privileged_fields_are_model_input": False,
        "recorded_evaluation_episodes": len(rows),
        "recorded_failure_count": sum(
            float(row["model_survival_seconds"]) < episode_limit
            for row in selected_rows
        ),
        "recorded_selected_episode_count": len(selected_rows),
        "rerun_episode_count": len(seeds),
        "workers": min(int(workers), len(seeds)),
        "device": str(device),
        "lookback_seconds": float(lookback_seconds),
        "decision_trace": decision_trace,
        "force_consensus_shield": bool(force_consensus_shield),
        "requested_failure_seeds": (
            list(failure_seeds) if failure_seeds is not None else None
        ),
        "policy_events": rollout.policy_events,
        "reproduction": reproduction,
        "reproduction_max_absolute_error_seconds": max(
            item["absolute_error_seconds"] for item in reproduction
        ),
        "rerun_collision_count": len(rollout.failure_diagnostics),
        "collision_bullet_count": len(collision_bullets),
        "targeted_collision_bullet_count": len(targeted),
        "untargeted_collision_bullet_count": len(untargeted),
        "targeted_collision_bullet_fraction": (
            len(targeted) / max(len(collision_bullets), 1)
        ),
        "matched_collision_bullet_count": len(matched),
        "matched_collision_velocity_known_count": sum(
            bool(item["track_velocity_known"]) for item in matched
        ),
        "targeted_track_age_decisions": _statistics(targeted_ages),
        "untargeted_track_age_decisions": _statistics(untargeted_ages),
        "filtered_terminal_decision_count": sum(
            bool(item["action_was_filtered"])
            for item in rollout.failure_diagnostics
        ),
        "all_actions_unsafe_terminal_decision_count": sum(
            bool(item["all_actions_unsafe"])
            for item in rollout.failure_diagnostics
        ),
        "failures": rollout.failure_diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)

    csv_output = output.with_suffix(".csv")
    csv_rows = []
    for diagnostic in rollout.failure_diagnostics:
        for bullet in diagnostic["collision_bullets"]:
            csv_rows.append({
                "original_episode_index": diagnostic.get("original_episode_index"),
                "seed": diagnostic["seed"],
                "survival_seconds": diagnostic["survival_seconds"],
                "targeted": bullet["targeted"],
                "matched_track": bullet["matched_track"],
                "track_age_decisions": bullet["track_age_decisions"],
                "track_age_seconds": bullet["track_age_seconds"],
                "track_velocity_known": bullet["track_velocity_known"],
                "track_history_length": bullet["track_history_length"],
                "track_missed": bullet["track_missed"],
                "track_confidence": bullet["track_confidence"],
                "track_position_uncertainty": bullet["track_position_uncertainty"],
                "track_occluded_steps": bullet["track_occluded_steps"],
                "track_association_group_size": bullet[
                    "track_association_group_size"
                ],
                "match_distance_pixels": bullet["match_distance_pixels"],
                "raw_policy_action": diagnostic["raw_policy_action"],
                "proposed_action": diagnostic["proposed_action"],
                "executed_action": diagnostic["executed_action"],
                "action_was_filtered": diagnostic["action_was_filtered"],
                "raw_immediate_collision_risk": diagnostic[
                    "raw_immediate_collision_risk"
                ],
                "selected_immediate_collision_risk": diagnostic[
                    "selected_immediate_collision_risk"
                ],
                "safety_threshold": diagnostic["safety_threshold"],
            })
    _write_attribution_csv(csv_output, csv_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run only recorded failures for privileged attribution"
    )
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--lookback-seconds", type=float, default=0.0)
    parser.add_argument("--decision-trace-seconds", type=float, default=0.0)
    parser.add_argument("--force-consensus-shield", action="store_true")
    parser.add_argument(
        "--seed",
        dest="failure_seeds",
        action="append",
        type=int,
        help="replay only this recorded episode seed; may be repeated",
    )
    args = parser.parse_args()
    report = diagnose(
        args.evaluation_dir,
        args.output,
        workers=args.workers,
        device_name=args.device,
        lookback_seconds=args.lookback_seconds,
        decision_trace_seconds=args.decision_trace_seconds,
        force_consensus_shield=args.force_consensus_shield,
        failure_seeds=(
            tuple(args.failure_seeds)
            if args.failure_seeds is not None
            else None
        ),
    )
    print(json.dumps({
        key: value
        for key, value in report.items()
        if key not in {"failures", "reproduction", "decision_trace", "policy_events"}
    }, indent=2))


if __name__ == "__main__":
    main()
