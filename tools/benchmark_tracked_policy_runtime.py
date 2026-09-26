"""Benchmark action-query training batches on the local CUDA device."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from barrage_rl.train_tracked_policy import _configure_trainable_scope
from barrage_rl.task_spec import TARGET_TASK, tracking_capacity_for


def benchmark(
    batch_size: int,
    iterations: int,
    trainable_scope: str,
    bullet_count: int,
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    # Keep each candidate independent. Without releasing the previous model and
    # allocator cache, a later large batch can look artificially close to OOM.
    gc.collect()
    torch.cuda.empty_cache()
    tracking_capacity = tracking_capacity_for(bullet_count)
    spec = TrackedPolicySpec(
        max_objects=tracking_capacity,
        tracker_capacity=tracking_capacity,
        expected_bullet_count=bullet_count,
    )
    model = ActionQueryPolicy(spec).to(device).train()
    parameters = _configure_trainable_scope(model, trainable_scope)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    objects = torch.randn(
        batch_size, spec.max_objects, spec.object_features, device=device
    )
    masks = torch.ones(batch_size, spec.max_objects, dtype=torch.bool, device=device)
    globals_ = torch.randn(batch_size, spec.global_features, device=device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        output = model(objects, masks, globals_)
        loss = sum(value.square().mean() for value in output[:2])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        output = model(objects, masks, globals_)
        loss = sum(value.square().mean() for value in output[:2])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Validate outside the timed section, so failures cannot be mistaken for
    # throughput wins and the checks do not serialize the optimizer loop.
    finite = torch.stack([
        torch.isfinite(loss),
        *(torch.isfinite(parameter).all() for parameter in model.parameters()),
        *(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
          if parameter.grad is not None),
    ]).all()
    if not bool(finite.item()):
        raise FloatingPointError("optimizer benchmark produced non-finite values")
    return {
        "finite_loss_parameters_and_gradients": True,
        "trainable_scope": trainable_scope,
        "bullets": float(bullet_count),
        "tracked_objects": float(tracking_capacity),
        "trainable_parameters": float(sum(parameter.numel() for parameter in parameters)),
        "batch_size": float(batch_size),
        "iterations": float(iterations),
        "batches_per_second": iterations / elapsed,
        "states_per_second": batch_size * iterations / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--bullets", type=int, nargs="+", default=(TARGET_TASK.bullet_count,)
    )
    parser.add_argument(
        "--trainable-scopes",
        nargs="+",
        choices=("full", "backbone"),
        default=("full", "backbone"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        benchmark(size, args.iterations, scope, bullets)
        for bullets in args.bullets
        for size in args.batch_sizes
        for scope in args.trainable_scopes
    ]
    payload = json.dumps(rows, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(payload)


if __name__ == "__main__":
    main()
