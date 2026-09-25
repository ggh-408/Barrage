"""Supplemental 300-bullet test of the search_jit policy (3000 episodes).

Run in the barrage environment. Use --smoke-test for a two-episode check.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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
from barrage_rl.artifacts import atomic_write_json, prepare_new_output, sha256_file
from tools.pixel_guard_candidate import PixelGuardConfig, install_guard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-episodes", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0 or args.smoke_episodes <= 0:
        parser.error("workers, batch-size and smoke-episodes must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    episodes = args.smoke_episodes if args.smoke_test else 3000
    seeds = random.Random(args.seed).sample(range(2_000_000_000, 2_147_000_000), episodes)
    mode = "smoke" if args.smoke_test else "3000"
    output = args.output or ROOT / "diagnostics" / f"pixel_guard_{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = output.resolve()
    prepare_new_output(output)
    torch.set_num_threads(1)
    config = PixelGuardConfig(allow_imminent_escape=True, recovery_search=True, compiled_search=True)
    sources = [*sorted((ROOT / "barrage_rl").glob("*.py")), Path(__file__),
               ROOT / "tools/pixel_guard_candidate.py", ROOT / "tools/pixel_recovery_planner.py",
               ROOT / "tools/pixel_search_kernel.py", ROOT / "image/plane(0).gif", ROOT / "image/bullet(5).gif"]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
    atomic_write_json(output / "experiment_manifest.json", {
        "variant": "search_jit", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "source_hashes": hashes,
        "guard_config": asdict(config), "episodes": episodes, "episode_seeds": seeds,
        "seed": args.seed, "smoke_test": args.smoke_test,
        "supplemental_test": not args.smoke_test, "checkpoint_selection_performed": False,
        "bullet_count": 300, "targeted_bullet_probability": 0.10,
        "evaluation_episode_limit_seconds": 120, "workers": args.workers,
        "evaluation_batch_size": args.batch_size, "device": args.device,
    })
    original_loader = evaluation.load_tracked_agent
    guards = []

    def loader(*a, **kw):
        agent, spec, metadata = original_loader(*a, **kw)
        guards.append(install_guard(agent, config))
        return agent, spec, metadata

    print(f"Starting {episodes} episodes, 300 bullets, 120 seconds; output: {output}", flush=True)
    started = time.perf_counter()
    with patch.object(evaluation, "load_tracked_agent", loader):
        result = evaluation.evaluate_tracked_checkpoint(
            str(checkpoint), episodes=episodes, episode_seeds=seeds, seed=args.seed,
            supplemental_test=not args.smoke_test, smoke_test=args.smoke_test,
            workers=args.workers, evaluation_batch_size=args.batch_size,
            output_dir=str(output / "evaluation"), device_name=args.device,
            episode_limit_seconds=120, bullet_count=300, targeted_bullet_probability=0.10,
            rendered_rgb=True, causal_action_delay_steps=0,
            analytic_shield=True, analytic_shield_gate="learned_all_unsafe",
        )
    with (output / "evaluation/evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    failures = [int(row["seed"]) for row in rows if float(row["model_survival_seconds"]) < 120]
    changed = [p for p, digest in hashes.items() if sha256_file(ROOT / p) != digest]
    atomic_write_json(output / "result.json", {
        "episodes": len(rows), "success_at_limit": result["success_at_limit"],
        "success_count": len(rows) - len(failures), "failure_seeds": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "smoke_test": args.smoke_test, "source_files_changed_during_run": changed,
        "guards": [guard.manifest() for guard in guards],
    })
    if changed:
        raise RuntimeError(f"Source files changed during evaluation: {changed}")
    if len(rows) != episodes or not guards:
        raise RuntimeError("Incomplete evaluation or guard was not installed")
    print(f"Completed: {len(rows) - len(failures)}/{episodes}; results: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
