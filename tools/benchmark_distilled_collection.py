"""Benchmark the recurrent-policy distillation collector."""

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
from barrage_rl.task_spec import BarrageTaskSpec  # noqa: E402
from tools.train_distilled_student import (  # noqa: E402
    DEFAULT_BACKBONE,
    DistillationConfig,
    DistillationReplay,
    _load_backbone,
    collect,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--env-count", type=int, default=108)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    output = args.output.resolve()
    prepare_new_output(output)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    config = DistillationConfig(
        samples=args.samples,
        env_count=args.env_count,
        workers=args.workers,
        device=args.device,
        smoke_test=True,
        evaluation_episodes=2,
    )
    model, spec, checkpoint = _load_backbone(args.backbone.resolve(), device)
    replay = DistillationReplay(output / "replay", config.samples, create=True)
    metrics = collect(
        model=model,
        spec=spec,
        base_checkpoint=checkpoint,
        replay=replay,
        config=config,
        task=BarrageTaskSpec(episode_limit_seconds=5.0),
        device=device,
        output=output,
    )
    result = {
        "workers": args.workers,
        "env_count": args.env_count,
        "samples": args.samples,
        **metrics,
    }
    atomic_write_json(output / "benchmark.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
