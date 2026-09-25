"""Benchmark tracked-policy optimizer throughput for each trainable scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from barrage_rl.train_tracked_policy import (
    TrackedDAggerConfig,
    TrackedReplay,
    _batch,
    _configure_trainable_scope,
    _loss,
)


def benchmark_scope(
    checkpoint: dict[str, object],
    replay: TrackedReplay,
    scope: str,
    batch_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float | str]:
    spec = TrackedPolicySpec(**checkpoint["tracked_policy_spec"])
    model = ActionQueryPolicy(spec, **checkpoint["model_hparams"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    parameters = _configure_trainable_scope(model, scope)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-5)
    indices = np.arange(batch_size, dtype=np.int64) % replay.size
    data = _batch(replay, indices, device)
    config = TrackedDAggerConfig(batch_size=batch_size, trainable_scope=scope)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss(model, *data, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return {
        "scope": scope,
        "batch_size": float(batch_size),
        "iterations": float(iterations),
        "elapsed_seconds": elapsed,
        "batches_per_second": iterations / elapsed,
        "samples_per_second": iterations * batch_size / elapsed,
        "peak_vram_mib": peak_bytes / (1024.0 * 1024.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--batch-size", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = TrackedPolicySpec(**checkpoint["tracked_policy_spec"])
    with np.load(args.replay, allow_pickle=False) as source:
        replay_count = len(source["actions"])
    replay = TrackedReplay.load(
        args.replay,
        capacity=max(int(replay_count), int(args.batch_size)),
        spec=spec,
        horizon_count=len(checkpoint["model_hparams"]["safety_horizons"]),
    )
    if replay.size <= 0:
        raise ValueError("replay is empty")
    results: list[dict[str, float | str]] = []
    for scope in ("full", "backbone"):
        result = benchmark_scope(
            checkpoint,
            replay,
            scope,
            args.batch_size,
            args.warmup,
            args.iterations,
            device,
        )
        results.append(result)
        print(" ".join(f"{key}={value}" for key, value in result.items()))
    if args.output is not None:
        payload = {
            "checkpoint": str(args.checkpoint),
            "replay": str(args.replay),
            "replay_samples": int(replay.size),
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
