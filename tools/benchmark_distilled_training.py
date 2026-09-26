"""Benchmark recurrent distillation batch sizes on an existing replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from barrage_rl.artifacts import atomic_write_json, prepare_new_output  # noqa: E402
from tools.train_distilled_student import (  # noqa: E402
    DistillationConfig,
    DistillationReplay,
    train_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sequences", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    root = args.output.resolve()
    prepare_new_output(root)
    replay = DistillationReplay(args.replay.resolve(), args.rows, create=False)
    replay.position = args.rows
    results = []
    for batch_sequences in args.batch_sequences:
        output = root / f"batch_{batch_sequences}"
        output.mkdir()
        config = DistillationConfig(
            samples=args.rows,
            batch_sequences=batch_sequences,
            epochs=args.epochs,
            patience=args.epochs,
            learning_rate=3e-4 * batch_sequences / 48.0,
            device="cuda",
            smoke_test=True,
            evaluation_episodes=2,
        )
        torch.cuda.reset_peak_memory_stats()
        _, metrics = train_student(replay, config, output, torch.device("cuda"))
        results.append({
            "batch_sequences": batch_sequences,
            "learning_rate": config.learning_rate,
            "training_seconds": metrics["training_seconds"],
            "epochs_run": metrics["epochs_run"],
            "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "validation_loss": metrics["loss"],
        })
    summary = {"rows": args.rows, "results": results}
    atomic_write_json(root / "benchmark.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
