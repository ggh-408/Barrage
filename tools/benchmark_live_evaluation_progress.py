"""Explicit smoke preflight: identical actions with reporting, plus clean early rejection."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "diagnostics/robust_mpc_live_frozen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot.resolve()))
    sys.path.insert(1, str(ROOT))
    from barrage_rl.artifacts import atomic_write_json, prepare_new_output
    from barrage_rl.evaluate_tracked_policy import evaluate_tracked_checkpoint
    from barrage_rl.parallel_evaluation import ParallelRolloutPruned
    from tools.search_robust_mpc_margin import evaluation_kwargs
    from tools.search_robust_mpc_live import progress_report

    prepare_new_output(args.output)
    source = ROOT / "diagnostics/robust_mpc_deva_teacher1000_fixed200/evaluation_config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    kwargs = evaluation_kwargs(config)
    seeds = list(range(968000001, 968000021))
    kwargs.update(episodes=20, episode_seeds=seeds, seed=seeds[0],
                  episode_limit_seconds=5., smoke_test=True,
                  evaluation_batch_size=20)
    rows = []
    for enabled in (False, True):
        label = "observed" if enabled else "baseline"
        output = args.output / label
        started = time.perf_counter()
        snapshots = []

        def observe(progress):
            report = progress_report(progress, seeds, 5., 0, time.perf_counter() - started)
            snapshots.append(report)
            atomic_write_json(args.output / "live_smoke_progress.json", report)
            return False

        result = evaluate_tracked_checkpoint(
            **dict(kwargs, output_dir=str(output)),
            live_progress_callback=observe if enabled else None,
            live_progress_interval_seconds=2.0,
        )
        wall = time.perf_counter() - started
        rows.append(dict(label=label, wall_seconds=wall,
                         decisions_per_second=result["safety_decisions"] / wall,
                         snapshots=len(snapshots)))
    baseline = args.output / "baseline"
    observed = args.output / "observed"
    for name in ("evaluation_episodes.csv", "action_histogram.json"):
        if (baseline / name).read_bytes() != (observed / name).read_bytes():
            raise AssertionError(f"reporting altered deterministic output: {name}")
    started = time.perf_counter()
    rejected = args.output / "rejection_probe"
    try:
        evaluate_tracked_checkpoint(
            **dict(kwargs, episodes=2, episode_seeds=seeds[:2], workers=2,
                   evaluation_batch_size=2, output_dir=str(rejected)),
            live_progress_callback=lambda progress: True,
            live_progress_interval_seconds=2.0,
        )
    except ParallelRolloutPruned as error:
        if len(error.progress.active_indices) != 2:
            raise AssertionError("unexpected rejection snapshot")
    else:
        raise AssertionError("early rejection did not terminate the rollout")
    if mp.active_children():
        raise AssertionError("worker processes survived early rejection")
    if (rejected / "evaluation_summary.json").exists():
        raise AssertionError("a partial rollout produced a complete metric summary")
    report = dict(kind="explicit_smoke_preflight", trials=rows,
                  deterministic_episode_and_action_outputs_equal=True,
                  early_rejection_wall_seconds=time.perf_counter() - started,
                  early_rejection_workers_clean=True,
                  production_interval_seconds=120., smoke_interval_seconds=2.)
    atomic_write_json(args.output / "preflight_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
