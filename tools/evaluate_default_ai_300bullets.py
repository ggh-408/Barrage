"""Parallel supplemental evaluation of the default window AI with 300 bullets."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main():
    import torch
    from Barrage import DEFAULT_AI_CHECKPOINT
    from barrage_rl import evaluate_tracked_policy as evaluation
    from barrage_rl.artifacts import atomic_write_json, prepare_new_output, sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        parser.error("episodes and workers must be positive")
    output = args.output or ROOT / "diagnostics" / f"default_ai_300bullets_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = output.resolve()
    prepare_new_output(output)
    seeds = random.Random(args.seed).sample(range(2_000_000_000, 2_147_000_000), args.episodes)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this parallel GPU benchmark")
    sources = [ROOT / "Barrage.py", Path(__file__), *sorted((ROOT / "barrage_rl").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
    checkpoint_hash = sha256_file(DEFAULT_AI_CHECKPOINT)
    atomic_write_json(output / "experiment_manifest.json", {
        "variant": "default_window_ai", "checkpoint": str(DEFAULT_AI_CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash, "source_hashes": hashes,
        "episodes": args.episodes, "episode_seeds": seeds, "seed": args.seed,
        "bullet_count": 300, "targeted_bullet_probability": 0.10,
        "evaluation_episode_limit_seconds": 120, "workers": args.workers,
        "evaluation_batch_size": args.workers, "device": "cuda",
        "supplemental_test": True, "checkpoint_selection_performed": False,
        "extra_pixel_guard": False, "rendered_rgb": True,
    })
    print(f"Starting {args.episodes} episodes; output: {output}", flush=True)
    started = time.perf_counter()
    # Match the existing supplemental 300-bullet entry's process-local task override.
    with patch.object(evaluation, "TARGET_TASK", replace(evaluation.TARGET_TASK, bullet_count=300)):
        result = evaluation.evaluate_tracked_checkpoint(
            str(DEFAULT_AI_CHECKPOINT), episodes=args.episodes, episode_seeds=seeds,
            seed=args.seed, supplemental_test=True, workers=args.workers,
            evaluation_batch_size=args.workers, output_dir=str(output / "evaluation"),
            device_name="cuda", episode_limit_seconds=120, bullet_count=300,
            targeted_bullet_probability=0.10, rendered_rgb=True,
            causal_action_delay_steps=0,
        )
    with (output / "evaluation/evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    failures = [{"seed": int(r["seed"]), "survival_seconds": float(r["model_survival_seconds"])}
                for r in rows if float(r["model_survival_seconds"]) < 120]
    changed = [p for p, digest in hashes.items() if sha256_file(ROOT / p) != digest]
    checkpoint_changed = sha256_file(DEFAULT_AI_CHECKPOINT) != checkpoint_hash
    atomic_write_json(output / "result.json", {
        "episodes": len(rows), "success_at_limit": result["success_at_limit"],
        "success_count": len(rows) - len(failures), "failures": failures,
        "mean_survival_seconds": sum(float(r["model_survival_seconds"]) for r in rows) / len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "source_files_changed_during_run": changed, "checkpoint_changed": checkpoint_changed,
    })
    if len(rows) != args.episodes or changed or checkpoint_changed:
        raise RuntimeError("Evaluation completeness or source integrity check failed")
    print(f"Completed: {len(rows) - len(failures)}/{len(rows)}; {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
