"""Supplemental 300-bullet test of the current image guard policy (3000 episodes).

Run in the barrage environment. Use --smoke-test for a two-episode check.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from barrage_rl import evaluate_tracked_policy as evaluation
from barrage_rl.artifacts import atomic_write_json, prepare_new_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "diagnostics/risk_removal_20260920/policy_teacher_cost.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--guard", choices=("receding",), default="receding")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--search-workers", type=int, default=9)
    parser.add_argument("--smoke-limit-seconds", type=float, default=5.0)
    parser.add_argument("--smoke-episodes", type=int, default=2)
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.smoke_episodes, args.episodes, args.search_workers) <= 0 or not 0 < args.smoke_limit_seconds <= 120:
        parser.error("counts must be positive; smoke limit must be in (0, 120]")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    episodes = args.smoke_episodes if args.smoke_test else args.episodes
    seeds = random.Random(args.seed).sample(range(2_000_000_000, 2_147_000_000), episodes)
    mode = "smoke" if args.smoke_test else str(episodes)
    output = args.output or ROOT / "diagnostics" / f"pixel_guard_refined_{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = output.resolve()
    prepare_new_output(output)
    torch.set_num_threads(1)
    limit = args.smoke_limit_seconds if args.smoke_test else 120.0
    sources = [*sorted((ROOT / "barrage_rl").glob("*.py")), Path(__file__),
               ROOT / "tools/pixel_guard_refined.py", ROOT / "tools/pixel_recovery_refined.py",
               ROOT / "tools/pixel_guard_candidate.py", ROOT / "tools/pixel_recovery_planner.py",
               ROOT / "tools/pixel_search_kernel.py", ROOT / "tools/pixel_guard_receding.py",
               ROOT / "tools/pixel_receding_kernel.py", ROOT / "tools/pixel_guard_continuation.py",
               ROOT / "image/plane(0).gif", ROOT / "image/bullet(5).gif"]
    sources = list(dict.fromkeys(sources))
    original_contents = {p: p.read_bytes() for p in sources}
    checkpoint_contents = checkpoint.read_bytes()
    atomic_write_json(output / "experiment_manifest.json", {
        "variant": args.guard, "checkpoint": str(checkpoint),
        "source_files": [str(p.relative_to(ROOT)) for p in sources],
        "policy_architecture": "policy_teacher_cost",
        "search_workers": args.search_workers, "episodes": episodes, "episode_seeds": seeds,
        "seed": args.seed, "smoke_test": args.smoke_test,
        "supplemental_test": not args.smoke_test, "checkpoint_selection_performed": False,
        "bullet_count": 300, "targeted_bullet_probability": 0.10,
        "evaluation_episode_limit_seconds": limit, "workers": args.workers,
        "evaluation_batch_size": args.batch_size, "device": args.device,
    })
    original_loader = evaluation.load_tracked_agent
    agents = []

    def loader(*a, **kw):
        agent, spec, metadata = original_loader(*a, **kw)
        if any("collision_head" in name for name in agent.model.state_dict()):
            raise RuntimeError("Loaded model still contains a learned risk head")
        if agent.use_safety_filter:
            raise RuntimeError("Learned risk filtering is still enabled")
        agents.append(agent)
        return agent, spec, metadata

    print(f"Starting {episodes} episodes, 300 bullets, {limit:g} seconds; output: {output}", flush=True)
    started = time.perf_counter()
    with patch.object(evaluation, "load_tracked_agent", loader):
        result = evaluation.evaluate_tracked_checkpoint(
            str(checkpoint), episodes=episodes, episode_seeds=seeds, seed=args.seed,
            supplemental_test=not args.smoke_test, smoke_test=args.smoke_test,
            workers=args.workers, evaluation_batch_size=args.batch_size,
            output_dir=str(output / "evaluation"), device_name=args.device,
            episode_limit_seconds=limit, bullet_count=300, targeted_bullet_probability=0.10,
            rendered_rgb=True, causal_action_delay_steps=0,
            analytic_shield=False, pixel_guard="receding", search_workers=args.search_workers,
        )
    with (output / "evaluation/evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    failures = [int(row["seed"]) for row in rows if float(row["model_survival_seconds"]) < limit]
    changed = [str(p.relative_to(ROOT)) for p, content in original_contents.items()
               if p.read_bytes() != content]
    checkpoint_unchanged = checkpoint.read_bytes() == checkpoint_contents
    atomic_write_json(output / "result.json", {
        "episodes": len(rows), "success_at_limit": result["success_at_limit"],
        "success_count": len(rows) - len(failures), "failure_seeds": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "smoke_test": args.smoke_test, "source_files_changed_during_run": changed,
        "controller": result["controller"],
        "risk_head_absent": True, "learned_risk_filter_enabled": False,
        "checkpoint_unchanged": checkpoint_unchanged,
        "evaluation_episode_limit_seconds": limit,
    })
    if changed or not checkpoint_unchanged:
        raise RuntimeError(f"Source files changed during evaluation: {changed}")
    if len(rows) != episodes or sorted(int(row["seed"]) for row in rows) != sorted(seeds) or not agents:
        raise RuntimeError("Incomplete evaluation or episode seeds do not match")
    print(f"Completed: {len(rows) - len(failures)}/{episodes}; results: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
