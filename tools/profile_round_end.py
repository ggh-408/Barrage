"""Profile every major operation performed after a DAgger training round."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, TypeVar

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from barrage_rl.artifacts import atomic_copy, atomic_write_json, atomic_write_text
from barrage_rl.evaluate_visual_set import evaluate
from barrage_rl.plot import save_round_summary_plot


T = TypeVar("T")


def timed(timings: Dict[str, float], name: str, operation: Callable[[], T]) -> T:
    started = time.perf_counter()
    result = operation()
    timings[name] = time.perf_counter() - started
    print(f"profile phase={name} elapsed={timings[name]:.3f}s", flush=True)
    return result


def write_one_row_summary(path: Path, result: Dict[str, object]) -> None:
    row = {"round": 1, "new_samples": 0, "student_probability": 1.0, **result}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure DAgger post-round checkpoint/evaluation/artifact timings"
    )
    parser.add_argument("checkpoint", help="roundN/validation_best.pt to profile")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--limit-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1_600_000)
    parser.add_argument("--bullets", type=int, default=50)
    parser.add_argument("--bullet-size", type=int, default=5)
    parser.add_argument("--bullet-speed", type=float, default=240.0)
    parser.add_argument("--targeted-probability", type=float, default=0.35)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output", default="runs/round_end_profile.json",
        help="JSON report path; temporary evaluation artifacts are removed",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    overall_started = time.perf_counter()

    print("Profiling a real post-round evaluation. This can take as long as training evaluation.")
    timed(
        timings,
        "round_end.reload_training_checkpoint",
        lambda: torch.load(checkpoint, map_location="cpu", weights_only=False),
    )

    with tempfile.TemporaryDirectory(prefix="barrage-round-profile-") as directory:
        temporary = Path(directory)
        evaluation_dir = temporary / "evaluation"
        result = evaluate(
            str(checkpoint), args.episodes, args.bullets, args.seed, args.device,
            str(evaluation_dir), args.targeted_probability, 40.0,
            args.limit_seconds, args.bullet_size, args.bullet_size,
            args.bullet_speed, args.bullet_speed, args.workers,
            timings=timings, progress=True,
        )

        summary_path = temporary / "round_summaries.csv"
        timed(
            timings, "round_end.write_summary_csv",
            lambda: write_one_row_summary(summary_path, result),
        )
        timed(
            timings, "round_end.copy_best_checkpoint",
            lambda: atomic_copy(checkpoint, temporary / "best.pt"),
        )

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = dict(checkpoint_data.get("config", {}))
        config["output_dir"] = "profile"
        config_path = temporary / "config.json"
        atomic_write_json(config_path, config)
        timed(
            timings, "round_end.render_results_png",
            lambda: save_round_summary_plot(
                summary_path, temporary / "results.png", config_path
            ),
        )
        timings["round_end.temporary_artifact_bytes"] = float(
            sum(path.stat().st_size for path in temporary.rglob("*") if path.is_file())
        )

    timings["round_end.total"] = time.perf_counter() - overall_started
    timed_values = {
        name: seconds for name, seconds in timings.items() if not name.endswith("bytes")
    }
    timed_total = sum(
        seconds for name, seconds in timed_values.items()
        if name not in {"round_end.total", "evaluation.total"}
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "episodes": args.episodes,
        "limit_seconds": args.limit_seconds,
        "workers": args.workers,
        "timings_seconds": timings,
        "share_of_measured_work_percent": {
            name: 100.0 * seconds / max(timed_total, 1e-9)
            for name, seconds in timed_values.items()
            if name not in {"round_end.total", "evaluation.total"}
        },
    }
    atomic_write_json(output, report)

    print("\nPost-round timing summary")
    print("-" * 72)
    for name, seconds in sorted(timed_values.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:48s} {seconds:10.3f}s")
    print(f"\nreport={output.resolve()}")


if __name__ == "__main__":
    main()
