"""Benchmark one exact-teacher episode and report emergency-search load."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barrage_rl.baselines import privileged_planner_supervision
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.task_spec import TARGET_TASK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--strong-beam-width", type=int, default=9)
    parser.add_argument("--strong-search-seconds", type=float, default=1.20)
    args = parser.parse_args()

    env = BarrageVisionEnv(
        **TARGET_TASK.env_kwargs(episode_limit_seconds=float(args.seconds))
    )
    env.reset(seed=int(args.seed))
    sequence_calls = 0
    nodes = 0
    decisions = 0
    sequence_decisions: list[int] = []
    greedy_survival: list[float] = []
    started = time.perf_counter()
    try:
        while True:
            supervision = privileged_planner_supervision(
                env,
                safety_horizons=(0.10,),
                sequence_beam_width=int(args.beam_width),
                strong_sequence_beam_width=int(args.strong_beam_width),
                strong_sequence_search_seconds=float(args.strong_search_seconds),
            )
            sequence_calls += int(supervision.used_sequence_search)
            if supervision.used_sequence_search:
                sequence_decisions.append(decisions)
            greedy_survival.append(float(supervision.greedy_survival_seconds))
            nodes += int(supervision.sequence_nodes_expanded)
            decisions += 1
            _, _, terminated, truncated, info = env.step(supervision.action)
            if terminated or truncated:
                break
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "seed": int(args.seed),
        "survival_seconds": float(info["survival_seconds"]),
        "terminated": bool(terminated),
        "decisions": int(decisions),
        "sequence_search_calls": int(sequence_calls),
        "sequence_nodes_expanded": int(nodes),
        "first_sequence_decision": (
            int(sequence_decisions[0]) if sequence_decisions else None
        ),
        "last_sequence_decision": (
            int(sequence_decisions[-1]) if sequence_decisions else None
        ),
        "finite_greedy_survival_min": min(
            (value for value in greedy_survival if value != float("inf")),
            default=None,
        ),
        "finite_greedy_survival_max": max(
            (value for value in greedy_survival if value != float("inf")),
            default=None,
        ),
        "wall_seconds": float(elapsed),
        "decisions_per_second": float(decisions / max(elapsed, 1e-9)),
    }, indent=2))


if __name__ == "__main__":
    main()
