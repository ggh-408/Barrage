"""Create a single-runtime checkpoint by averaging compatible model weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_checkpoints(
    inputs: list[Path],
    output: Path,
    weights: list[float] | None = None,
    include_prefixes: tuple[str, ...] | None = None,
) -> None:
    if len(inputs) < 2:
        raise ValueError("at least two checkpoints are required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if weights is None:
        weights = [1.0 / len(inputs)] * len(inputs)
    if len(weights) != len(inputs) or any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative and match the inputs")
    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must have a positive sum")
    normalized_weights = [weight / weight_sum for weight in weights]
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in inputs
    ]
    reference = checkpoints[0]
    compatibility_keys = (
        "model_version",
        "model_hparams",
        "tracked_policy_spec",
        "observation_size",
        "use_safety_filter",
    )
    for checkpoint in checkpoints[1:]:
        for key in compatibility_keys:
            if checkpoint.get(key) != reference.get(key):
                raise ValueError(f"incompatible checkpoint field: {key}")
    state_keys = tuple(reference["model"])
    if any(tuple(checkpoint["model"]) != state_keys for checkpoint in checkpoints):
        raise ValueError("model state dictionaries have different keys")

    averaged = {}
    for key in state_keys:
        tensors = [checkpoint["model"][key] for checkpoint in checkpoints]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"incompatible tensor shape: {key}")
        selected = include_prefixes is None or key.startswith(include_prefixes)
        if tensors[0].is_floating_point() and selected:
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float64)
            for weight, tensor in zip(normalized_weights, tensors):
                accumulator.add_(tensor.to(torch.float64), alpha=weight)
            averaged[key] = accumulator.to(tensors[0].dtype)
        elif not tensors[0].is_floating_point():
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating state differs: {key}")
            averaged[key] = tensors[0].clone()
        else:
            averaged[key] = tensors[0].clone()

    result = dict(reference)
    result["model"] = averaged
    result.pop("optimizer", None)
    result["checkpoint_average"] = {
        "method": (
            "weighted_parameter_mean"
            if include_prefixes is None
            else "selective_weighted_parameter_mean"
        ),
        "weights": normalized_weights,
        "include_prefixes": list(include_prefixes or ()),
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in inputs
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--weights",
        help="comma-separated source weights; defaults to a uniform mean",
    )
    parser.add_argument(
        "--include-prefixes",
        help=(
            "comma-separated model-state prefixes to average; parameters outside "
            "the selected modules are copied from the first checkpoint"
        ),
    )
    args = parser.parse_args()
    weights = (
        [float(value) for value in args.weights.split(",")]
        if args.weights
        else None
    )
    include_prefixes = (
        tuple(value for value in args.include_prefixes.split(",") if value)
        if args.include_prefixes
        else None
    )
    average_checkpoints(args.inputs, args.output, weights, include_prefixes)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
