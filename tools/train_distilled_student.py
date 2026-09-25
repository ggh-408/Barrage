"""Distill the tracked policy into one recurrent deployment student."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from barrage_rl.timing import DECISION_DT, PHYSICS_FPS
from barrage_rl.artifacts import (  # noqa: E402
    atomic_torch_save,
    atomic_write_json,
    git_revision,
    prepare_new_output,
)
from barrage_rl.distilled_student import (  # noqa: E402
    DistilledStudentNetwork,
    DistilledStudentSpec,
    causal_frame_latent,
)
from barrage_rl.evaluate_tracked_policy import (  # noqa: E402
    evaluate_tracked_checkpoint,
)
from barrage_rl.task_spec import BarrageTaskSpec  # noqa: E402
from barrage_rl.tracked_collection import ParallelTrackedDaggerEnv  # noqa: E402
from barrage_rl.tracked_policy import (  # noqa: E402
    ActionQueryPolicy,
    TrackedPolicySpec,
)


DEFAULT_BACKBONE = ROOT / "runs" / "visual_set_v32" / "best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "visual_set_v41"


@dataclass
class DistillationConfig:
    samples: int = 400_000
    env_count: int = 108
    workers: int = 9
    collection_seed: int = 990_000_001
    evaluation_seed: int = 991_000_001
    model_seed: int = 992_000_001
    batch_sequences: int = 96
    sequence_decisions: int = 60
    epochs: int = 24
    patience: int = 4
    learning_rate: float = 6e-4
    weight_decay: float = 1e-5
    random_action_probability: float = 0.05
    evaluation_episodes: int = 200
    evaluation_workers: int = 10
    evaluation_batch_size: int = 10
    device: str = "cuda"
    smoke_test: bool = False

    def validate(self) -> None:
        if min(
            self.samples,
            self.env_count,
            self.workers,
            self.batch_sequences,
            self.sequence_decisions,
            self.epochs,
            self.patience,
        ) <= 0:
            raise ValueError("distillation counts must be positive")
        if not self.smoke_test and self.evaluation_episodes != 200:
            raise ValueError("production distillation evaluation requires 200 episodes")


REPLAY_LAYOUT: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "frame_latent": (np.dtype(np.float16), (192,)),
    "action_queries": (np.dtype(np.float16), (9, 192)),
    "action_features": (np.dtype(np.float16), (9, 6)),
    "target_probability": (np.dtype(np.float16), (9,)),
    "episode_id": (np.dtype(np.int64), ()),
    "episode_step": (np.dtype(np.int32), ()),
    "priority": (np.dtype(np.float16), ()),
}


class DistillationReplay:
    def __init__(self, directory: Path, rows: int, *, create: bool) -> None:
        self.directory = Path(directory)
        self.rows = int(rows)
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.arrays: dict[str, np.memmap[Any, Any]] = {}
        for name, (dtype, tail) in REPLAY_LAYOUT.items():
            path = self.directory / f"{name}.npy"
            self.arrays[name] = open_memmap(
                path,
                mode="w+" if create else "r+",
                dtype=dtype,
                shape=(self.rows, *tail),
            )
        self.position = 0

    def append(self, **values: np.ndarray) -> None:
        count = len(values["episode_id"])
        stop = self.position + count
        if stop > self.rows:
            raise ValueError("distillation replay capacity exceeded")
        if set(values) != set(self.arrays):
            raise ValueError("distillation replay fields differ")
        for name, value in values.items():
            self.arrays[name][self.position:stop] = value
        self.position = stop

    def flush(self) -> None:
        for value in self.arrays.values():
            value.flush()


def _load_backbone(
    path: Path, device: torch.device
) -> tuple[ActionQueryPolicy, TrackedPolicySpec, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    spec = TrackedPolicySpec(**checkpoint["tracked_policy_spec"])
    if spec.max_objects != 384 or spec.tracker_capacity != 384:
        raise ValueError("distillation backbone requires 384 tracked/model slots")
    hparams = dict(checkpoint.get("model_hparams", {}))
    if int(checkpoint.get("model_version", 10)) <= 10:
        hparams.setdefault("geometry_statistics", ("minimum",))
    model = ActionQueryPolicy(spec, **hparams).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, spec, checkpoint


@torch.inference_mode()
def collect(
    *,
    model: ActionQueryPolicy,
    spec: TrackedPolicySpec,
    base_checkpoint: dict[str, Any],
    replay: DistillationReplay,
    config: DistillationConfig,
    task: BarrageTaskSpec,
    device: torch.device,
    output: Path,
) -> dict[str, float]:
    rng = np.random.default_rng(config.collection_seed ^ 0x5A17)
    started = time.perf_counter()
    disagreements = 0
    failures = 0
    next_report = started + 30.0
    with ParallelTrackedDaggerEnv(
        env_count=config.env_count,
        workers=config.workers,
        seed=config.collection_seed,
        env_kwargs=task.env_kwargs(),
        spec=spec,
        safety_horizons=model.safety_horizons,
        teacher_kind="exact",
        teacher_horizon_seconds=1.5,
        teacher_reaction_seconds=DECISION_DT,
        deployment_rgb_observation=True,
        causal_action_delay_steps=0,
    ) as pipeline:
        while replay.position < config.samples:
            take = min(config.env_count, config.samples - replay.position)
            objects = torch.as_tensor(
                pipeline.objects[:take], device=device, dtype=torch.float32
            )
            masks = torch.as_tensor(
                pipeline.masks[:take], device=device, dtype=torch.bool
            )
            globals_ = torch.as_tensor(
                pipeline.globals[:take], device=device, dtype=torch.float32
            )
            policy, teacher_cost, collision_logits, _, latents = (
                model.forward_with_arbiter_latents(objects, masks, globals_)
            )
            learned = policy.argmax(dim=1)
            target = torch.softmax(policy, dim=1)
            teacher = torch.as_tensor(
                pipeline.teacher_actions[:take], device=device, dtype=torch.long
            )
            episode_ids = (
                np.arange(take, dtype=np.int64) * 10**7
                + pipeline.episode_serial[:take]
            )
            priority = (
                1.0
                + 4.0 * (learned != teacher).float()
            )
            frame = causal_frame_latent(
                latents.object_tokens, masks, latents.action_queries
            )
            features = DistilledStudentNetwork.action_features(
                policy, teacher_cost, collision_logits
            )
            replay.append(
                frame_latent=frame.cpu().numpy().astype(np.float16),
                action_queries=latents.action_queries.cpu().numpy().astype(np.float16),
                action_features=features.cpu().numpy().astype(np.float16),
                target_probability=target.cpu().numpy().astype(np.float16),
                episode_id=episode_ids,
                episode_step=pipeline.episode_steps[:take].copy(),
                priority=priority.cpu().numpy().astype(np.float16),
            )
            disagreements += int((learned != teacher).sum())
            behavior = learned.cpu().numpy()
            random_actions = rng.random(take) < config.random_action_probability
            if np.any(random_actions):
                behavior[random_actions] = rng.integers(
                    0, 9, size=int(random_actions.sum())
                )
            actions = np.zeros(config.env_count, np.int64)
            actions[:take] = behavior
            if take < config.env_count:
                actions[take:] = 0
            done, truncated, _ = pipeline.step(actions)
            failures += int(np.sum(done[:take] & ~truncated[:take]))
            now = time.perf_counter()
            if now >= next_report or replay.position == config.samples:
                elapsed = now - started
                print(
                    "distillation_collection "
                    f"rows={replay.position}/{config.samples} "
                    f"states_per_second={replay.position / max(elapsed, 1e-6):.1f} "
                    f"failures={failures}",
                    flush=True,
                )
                replay.flush()
                atomic_write_json(output / "collection_progress.json", {
                    "rows": replay.position,
                    "samples": config.samples,
                    "failures": failures,
                    "elapsed_seconds": elapsed,
                })
                next_report = now + 30.0
    replay.flush()
    elapsed = time.perf_counter() - started
    return {
        "rows": float(replay.position),
        "collection_seconds": elapsed,
        "states_per_second": replay.position / max(elapsed, 1e-6),
        "teacher_disagreement_fraction": disagreements / max(replay.position, 1),
        "failed_episodes": float(failures),
    }


def _episode_split(values: np.ndarray) -> np.ndarray:
    mixed = np.asarray(values, dtype=np.uint64).copy()
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return (mixed % np.uint64(10)).astype(np.int8)


def _sequence_chunks(replay: DistillationReplay, length: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    episode = np.asarray(replay.arrays["episode_id"][: replay.position])
    step = np.asarray(replay.arrays["episode_step"][: replay.position])
    order = np.lexsort((step, episode))
    sorted_episode = episode[order]
    boundaries = np.flatnonzero(np.diff(sorted_episode)) + 1
    groups = np.split(order, boundaries)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    for group in groups:
        if not len(group):
            continue
        target = validation if _episode_split(episode[group[:1]])[0] == 0 else train
        for start in range(0, len(group), length):
            chunk = group[start:start + length]
            if len(chunk) >= 4:
                target.append(chunk)
    if not train or not validation:
        raise RuntimeError("distillation replay split produced an empty partition")
    return train, validation


def _batch(
    replay: DistillationReplay,
    chunks: list[np.ndarray],
    selected: np.ndarray,
    length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = len(selected)
    indices = np.zeros((rows, length), np.int64)
    valid = np.zeros((rows, length), np.bool_)
    for row, chunk_index in enumerate(selected):
        chunk = chunks[int(chunk_index)]
        indices[row, : len(chunk)] = chunk
        valid[row, : len(chunk)] = True
    result: dict[str, torch.Tensor] = {
        "valid": torch.as_tensor(valid, device=device),
    }
    for name in (
        "frame_latent",
        "action_queries",
        "action_features",
        "target_probability",
        "priority",
    ):
        value = np.asarray(replay.arrays[name][indices])
        result[name] = torch.as_tensor(value, device=device, dtype=torch.float32)
    return result


def _sequence_loss(
    model: DistilledStudentNetwork,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = None
    policy_sum = torch.zeros((), device=batch["frame_latent"].device)
    policy_weight = torch.zeros_like(policy_sum)
    for step in range(batch["frame_latent"].shape[1]):
        prediction = model(
            batch["frame_latent"][:, step],
            batch["action_queries"][:, step],
            batch["action_features"][:, step],
            hidden,
        )
        hidden = prediction.hidden
        valid = batch["valid"][:, step]
        weight = batch["priority"][:, step] * valid.float()
        target = batch["target_probability"][:, step]
        student_logits = (
            batch["action_features"][:, step, :, 0]
            + prediction.policy_correction
        )
        per_policy = torch.sum(
            target
            * (
                torch.log(target.clamp_min(1e-7))
                - torch.log_softmax(student_logits, dim=1)
            ),
            dim=1,
        )
        policy_sum += torch.sum(per_policy * weight)
        policy_weight += weight.sum()
        if hidden is not None:
            hidden = torch.where(valid[:, None], hidden, hidden.detach())
    policy_loss = policy_sum / policy_weight.clamp_min(1.0)
    return policy_loss, {"policy": policy_loss}


@torch.inference_mode()
def _validation(
    model: DistilledStudentNetwork,
    replay: DistillationReplay,
    chunks: list[np.ndarray],
    config: DistillationConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {key: 0.0 for key in ("loss", "policy")}
    batches = 0
    for start in range(0, len(chunks), config.batch_sequences):
        chosen = np.arange(start, min(start + config.batch_sequences, len(chunks)))
        batch = _batch(replay, chunks, chosen, config.sequence_decisions, device)
        loss, pieces = _sequence_loss(model, batch)
        sums["loss"] += float(loss)
        for key, value in pieces.items():
            sums[key] += float(value)
        batches += 1
    return {key: value / max(batches, 1) for key, value in sums.items()}


def train_student(
    replay: DistillationReplay,
    config: DistillationConfig,
    output: Path,
    device: torch.device,
) -> tuple[DistilledStudentNetwork, dict[str, float]]:
    torch.manual_seed(config.model_seed)
    spec = DistilledStudentSpec(
        history_decisions=config.sequence_decisions,
    )
    model = DistilledStudentNetwork(spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_chunks, validation_chunks = _sequence_chunks(
        replay, config.sequence_decisions
    )
    rng = np.random.default_rng(config.model_seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_chunks))
        total = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_sequences):
            selected = order[start:start + config.batch_sequences]
            batch = _batch(
                replay, train_chunks, selected, config.sequence_decisions, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _sequence_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        validation = _validation(
            model, replay, validation_chunks, config, device
        )
        row = {
            "epoch": float(epoch),
            "train_loss": total / max(batches, 1),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        with (output / "training_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            "distillation_epoch "
            f"epoch={epoch} train_loss={row['train_loss']:.5f} "
            f"validation_loss={validation['loss']:.5f}",
            flush=True,
        )
        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = dict(validation)
            best_metrics["best_epoch"] = float(epoch)
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("distilled student training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    best_metrics.update({
        "training_seconds": time.perf_counter() - started,
        "train_sequence_count": float(len(train_chunks)),
        "validation_sequence_count": float(len(validation_chunks)),
        "epochs_run": float(len(history)),
    })
    return model, best_metrics


def run(
    *,
    backbone_path: Path,
    output: Path,
    config: DistillationConfig,
    reuse_replay: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    if reuse_replay is not None:
        source_manifest = reuse_replay.resolve().parent / "run_manifest.json"
        if not source_manifest.is_file():
            raise ValueError("reused replay requires its original run_manifest.json")
        source_task = json.loads(source_manifest.read_text(encoding="utf-8")).get("task", {})
        if (source_task.get("bullet_count") != BarrageTaskSpec().bullet_count
                or source_task.get("targeted_bullet_probability") != 0.10):
            raise ValueError("reused replay must have been collected with 300 bullets and 0.10 targeting")
    prepare_new_output(output)
    device = torch.device(
        config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, policy_spec, base_checkpoint = _load_backbone(backbone_path, device)
    task = BarrageTaskSpec(episode_limit_seconds=5.0 if config.smoke_test else 120.0)
    manifest = {
        "git_revision": git_revision(ROOT),
        "backbone": str(backbone_path.resolve()),
        "task": task.manifest(),
        "config": {**asdict(config), "physics_fps": PHYSICS_FPS},
        "architecture": "unified_distilled_student",
        "deployment_action_modules": ["student", "unified_action_selector"],
        "reused_replay": None if reuse_replay is None else str(reuse_replay.resolve()),
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    if reuse_replay is None:
        replay = DistillationReplay(output / "replay", config.samples, create=True)
        collection = collect(
            model=model,
            spec=policy_spec,
            base_checkpoint=base_checkpoint,
            replay=replay,
            config=config,
            task=task,
            device=device,
            output=output,
        )
    else:
        replay = DistillationReplay(reuse_replay.resolve(), config.samples, create=False)
        replay.position = config.samples
        source_summary = reuse_replay.resolve().parent / "summary.json"
        collection = {
            "rows": float(config.samples),
            "reused": True,
            "source": str(reuse_replay.resolve()),
        }
        if source_summary.exists():
            prior = json.loads(source_summary.read_text(encoding="utf-8"))
            collection.update(dict(prior.get("collection", {})))
    student, training = train_student(replay, config, output, device)
    checkpoint = {k: base_checkpoint[k] for k in ("tracked_policy_spec", "model_hparams",
        "observation_size", "inference_head") if k in base_checkpoint}
    checkpoint.pop("use_safety_filter", None)
    checkpoint.pop("safety_threshold", None)
    checkpoint["model"] = model.state_dict()
    checkpoint["model_version"] = model.model_version
    for key in (
        "option_arbiter",
        "option_arbiter_spec",
        "option_arbiter_schema_version",
        "sequence_gate",
        "sequence_gate_spec",
        "distilled_planner_fallback",
        "distilled_fallback_threshold",
    ):
        checkpoint.pop(key, None)
    checkpoint.update({
        "distilled_student_schema_version": student.schema_version,
        "distilled_student_spec": student.spec.to_dict(),
        "distilled_student": student.state_dict(),
        "distillation_manifest": manifest,
        "distillation_collection": collection,
        "distillation_training": training,
        "config": {
            **{k:v for k,v in base_checkpoint.get("config", {}).items()
               if k in {"bullet_count", "bullet_size", "observation_size", "action_repeat",
                        "physics_fps", "targeted_bullet_probability",
                        "collection_causal_action_delay_steps", "evaluation_causal_action_delay_steps"}},
            "bullet_count": 300,
            "evaluation_bullet_count": 300,
            "max_objects": 384,
            "tracker_capacity": 384,
            "targeted_bullet_probability": 0.10,
            "evaluation_causal_action_delay_steps": 0,
            "collection_causal_action_delay_steps": 0,
        },
    })
    candidate = output / "candidate.pt"
    atomic_torch_save(checkpoint, candidate)
    episodes = 2 if config.smoke_test else config.evaluation_episodes
    evaluation = evaluate_tracked_checkpoint(
        str(candidate),
        episodes=episodes,
        workers=min(config.evaluation_workers, episodes),
        seed=config.evaluation_seed,
        output_dir=str(output / "round1" / "evaluation"),
        device_name=config.device,
        episode_limit_seconds=5.0 if config.smoke_test else 120.0,
        bullet_count=300,
        targeted_bullet_probability=0.10,
        rendered_rgb=True,
        smoke_test=config.smoke_test,
        evaluation_batch_size=min(config.evaluation_batch_size, episodes),
        causal_action_delay_steps=0,
        analytic_shield=False,
    )
    checkpoint["distillation_evaluation"] = evaluation
    atomic_torch_save(checkpoint, output / "best.pt")
    summary = {
        "collection": collection,
        "training": training,
        "evaluation": evaluation,
        "checkpoint": str((output / "best.pt").resolve()),
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=400_000)
    parser.add_argument("--env-count", type=int, default=108)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--batch-sequences", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--reuse-replay", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    smoke = bool(args.smoke_test)
    config = DistillationConfig(
        samples=2_048 if smoke else args.samples,
        env_count=4 if smoke else args.env_count,
        workers=2 if smoke else args.workers,
        batch_sequences=4 if smoke else args.batch_sequences,
        learning_rate=args.learning_rate,
        epochs=1 if smoke else args.epochs,
        patience=1 if smoke else args.patience,
        evaluation_episodes=2 if smoke else 200,
        device=args.device,
        smoke_test=smoke,
    )
    summary = run(
        backbone_path=args.backbone.resolve(),
        output=args.output.resolve(),
        config=config,
        reuse_replay=(
            None if args.reuse_replay is None else args.reuse_replay.resolve()
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
