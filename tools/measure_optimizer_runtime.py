"""Run the existing synthetic optimizer benchmark against a frozen source tree."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(source), str(ROOT)]
    import torch
    import barrage_rl.tracked_policy as policy
    from tools.benchmark_tracked_policy_runtime import benchmark
    from tools.benchmark_core_latency import _resources, _write_json
    if Path(policy.__file__).resolve() != source / "barrage_rl/tracked_policy.py":
        raise RuntimeError("Unexpected training source")
    torch.set_num_threads(10)
    torch.manual_seed(20260831)
    before = _resources()
    started = time.perf_counter()
    metrics = benchmark(args.batch_size, args.iterations, "full", 300)
    result = {
        "kind": "existing_synthetic_optimizer_performance_validation",
        "source_root": str(source), "seed": 20260831,
        "torch_threads": 10, "targeted_bullet_probability": 0.10,
        "metrics": metrics, "wall_seconds": time.perf_counter() - started,
        "resources_before": before, "resources_after": _resources(),
        "checkpoint_written": False,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
