"""Bounded experiments using the project's existing checkpoint evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from barrage_rl import evaluate_tracked_policy as evaluation
from barrage_rl.artifacts import atomic_write_json, sha256_file, prepare_new_output
from tools.pixel_guard_candidate import PixelGuardConfig, install_guard

CHECKPOINT = ROOT / "diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["baseline", "shape", "interval", "escape", "recovery", "search", "search_jit", "long_interval"], required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=553110052)
    parser.add_argument("--seed-file", type=Path, help="explicit JSON seed list for smoke diagnostics")
    args = parser.parse_args()
    if args.episodes != 200 and not args.smoke_test:
        parser.error("formal evaluation requires exactly 200 episodes")
    output = Path(args.output)
    prepare_new_output(output)
    torch.set_num_threads(1)
    config = PixelGuardConfig(physics_steps=12 if args.variant == "long_interval" else 4,
                             use_intervals=args.variant != "shape",
                             allow_imminent_escape=args.variant in ("escape", "search", "search_jit"),
                             resolve_interval_conflicts=args.variant == "recovery",
                             recovery_search=args.variant in ("search", "search_jit"),
                             compiled_search=args.variant == "search_jit")
    seeds = None
    hard = []
    if args.smoke_test:
        seeds = (json.loads(args.seed_file.read_text(encoding="utf-8"))
                 if args.seed_file else list(range(args.seed, args.seed + args.episodes)))
        if len(seeds) != args.episodes or len(set(seeds)) != len(seeds):
            parser.error("smoke seeds must contain one unique seed per episode")
    elif args.seed_file:
        parser.error("--seed-file requires --smoke-test")
    original = evaluation.load_tracked_agent
    guards = []
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in [
        *sorted((ROOT / "barrage_rl").glob("*.py")),
        ROOT / "tools/pixel_guard_candidate.py", ROOT / "tools/validate_pixel_guard.py",
        ROOT / "tools/pixel_recovery_planner.py",
        ROOT / "tools/pixel_search_kernel.py",
        ROOT / "image/plane(0).gif", ROOT / "image/bullet(5).gif",
    ]}

    def loader(*a, **kw):
        agent, spec, metadata = original(*a, **kw)
        if args.variant != "baseline":
            guards.append(install_guard(agent, config))
        return agent, spec, metadata

    manifest = {"variant": args.variant, "guard_config": config.__dict__,
                "bullet_count": 300, "targeted_bullet_probability": 0.10,
                "seed_source": str(args.seed_file) if args.seed_file else "configured seed sequence",
                "checkpoint_sha256": sha256_file(CHECKPOINT), "source_hashes": hashes,
                "smoke_test": args.smoke_test, "episodes": args.episodes,
                "evaluation_batch_size": args.batch_size,
                "hard_seeds": hard, "episode_seeds": seeds,
                "safety_threshold": 0.18, "training_performed": False,
                "policy_dynamic_inputs": "current RGB-derived tracked features only"}
    atomic_write_json(output / "experiment_manifest.json", manifest)
    started = time.perf_counter()
    with patch.object(evaluation, "load_tracked_agent", loader):
        result = evaluation.evaluate_tracked_checkpoint(
            str(CHECKPOINT), episodes=args.episodes, workers=10, seed=args.seed,
            output_dir=str(output / "evaluation"), device_name="cuda",
            episode_limit_seconds=120, bullet_count=300, targeted_bullet_probability=0.10,
            rendered_rgb=True, smoke_test=args.smoke_test, episode_seeds=seeds,
            evaluation_batch_size=args.batch_size, causal_action_delay_steps=0,
            analytic_shield=True, analytic_shield_gate="learned_all_unsafe",
        )
    changed = [p for p,h in hashes.items() if sha256_file(ROOT / p) != h]
    with (output / "evaluation/evaluation_episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    passed = lambda row: float(row["model_survival_seconds"]) >= 120
    report = {"variant": args.variant, "success_at_limit": result["success_at_limit"],
              "bullet_count": 300, "historical_hard_control_classification_used": False,
              "episodes": len(rows), "failures": [int(r["seed"]) for r in rows if not passed(r)],
              "hard_successes": sum(passed(r) for r in rows if int(r["seed"]) in hard),
              "control_failures": [int(r["seed"]) for r in rows if int(r["seed"]) not in hard and not passed(r)],
              "elapsed_seconds": time.perf_counter()-started,
              "source_files_changed_during_run": changed,
              "guards": [guard.manifest() for guard in guards]}
    atomic_write_json(output / "result.json", report)
    print(json.dumps(report), flush=True)
    if changed:
        raise RuntimeError(f"experiment source changed during run: {changed}")


if __name__ == "__main__":
    main()
