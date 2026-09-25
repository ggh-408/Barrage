"""DAgger training for the persistent-track action-query policy."""

from __future__ import annotations

from .timing import DECISION_DT, DECISION_HZ, PHYSICS_FPS

import argparse
import copy
import csv
import io
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from .artifacts import (
    atomic_copy,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    git_revision,
    prepare_new_output,
    contents_equal,
)
from .env import BarrageVisionEnv
from .evaluate_tracked_policy import evaluate_tracked_checkpoint
from .plot import save_round_summary_plot
from .task_spec import (
    BarrageTaskSpec,
    PRODUCTION_ANALYTIC_SHIELD,
    PRODUCTION_ANALYTIC_SHIELD_GATE,
    TARGET_TASK,
    TARGET_TRACKING_CAPACITY,
)
from .tracked_collection import ParallelTrackedDaggerEnv
from .tracked_policy import (
    ActionQueryPolicy,
    TrackedPolicyAgent,
    TrackedPolicySpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_SOURCE_FILES = (
    "Barrage.py",
    "barrage_rl/runtime_core.py",
    "barrage_rl/env.py",
    "barrage_rl/live_screen.py",
    "barrage_rl/image_oracle.py",
    "barrage_rl/tracked_policy.py",
    "barrage_rl/action_geometry.py",
    "barrage_rl/action_selector.py",
    "barrage_rl/tracked_collection.py",
    "barrage_rl/baselines.py",
    "barrage_rl/recovery_planner.py",
    "barrage_rl/causal_control.py",
    "barrage_rl/parallel_evaluation.py",
    "barrage_rl/evaluate_tracked_policy.py",
    "barrage_rl/train_tracked_policy.py",
    "barrage_rl/task_spec.py",
)
_COLLECTION_ROUND_SEED_STRIDE = 1_000_000
_AUTOMATIC_SEED_LIMIT = 2**31
_PROGRESS_REPORT_INTERVAL_SECONDS = 30.0
DEFAULT_TRAINING_BULLET_COUNT = 300
_COMPOSITE_DEPLOYMENT_KEY_PREFIXES = (
    "distilled_",
    "distillation_",
    "option_arbiter",
    "option_agent",
    "option_controller",
    "option_replay_",
    "sequence_gate",
    "viability_controller",
    "recovery_controller",
    "deployment_controller",
    "controller_stack",
    "composite_",
)
_COMPOSITE_DEPLOYMENT_KEYS = frozenset({
    "deployment_action_modules",
    "planner_fallback",
})


@dataclass
class TrackedDAggerConfig:
    output_dir: str = "runs/visual_set_v52"
    initial_checkpoint: str = "diagnostics/risk_removal_20260920/policy_teacher_cost.pt"
    # Start fresh: historical replays may use different bullet counts or
    # teacher reaction durations and therefore different supervision labels.
    initial_replay: str = ""
    evaluate_initial_checkpoint: bool = True
    rounds: int = 1
    samples_per_round: int = 400_000
    replay_capacity: int = 1_000_000
    epochs_per_round: int = 4
    early_stopping_patience: int = 1
    batch_size: int = 512
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    policy_weight: float = 1.0
    teacher_cost_weight: float = 0.5
    trainable_scope: str = "full"
    teacher_temperature: float = 0.08
    num_envs: int = 36
    cpu_workers: int = 9
    observation_size: int = 384
    bullet_count: int = DEFAULT_TRAINING_BULLET_COUNT
    targeted_bullet_probability: float = TARGET_TASK.targeted_bullet_probability
    deployment_rgb_observation: bool = True
    collection_episode_seconds: float = 120.0
    random_action_probability: float = 0.0
    pixel_guard: str = "receding"
    search_workers: int = 9
    record_branch_snapshots: bool = False
    priority_sample_fraction: float = 0.75
    priority_mode: str = "action_disagreement"
    disagreement_priority: float = 4.0
    regret_priority: float = 4.0
    unsafe_behavior_priority: float = 8.0
    priority_cap: float = 0.0
    early_state_priority: float = 3.0
    late_state_priority: float = 2.0
    failure_tail_priority: float = 12.0
    early_state_decisions: int = 90
    late_state_decisions: int = 900
    failure_tail_decisions: int = 36
    seed: int = 4_701
    collection_seed: int | None = None
    collection_seed_list: tuple[int, ...] = ()
    repeat_collection_seeds: bool = False
    evaluation_seed: int | None = None
    evaluation_episodes: int = 200
    evaluation_workers: int = 10
    evaluation_bullet_count: int = TARGET_TASK.bullet_count
    evaluation_batch_size: int = 10
    collection_causal_action_delay_steps: int = 0
    evaluation_causal_action_delay_steps: int = 0
    evaluation_episode_limit_seconds: float = 120.0
    selection_mode: str = "success_at_limit"
    teacher_kind: str = "exact"
    teacher_horizon_seconds: float = 1.5
    teacher_reaction_seconds: float = 0.10
    safety_horizons: tuple[float, ...] = (0.10, 0.30, 0.60, 1.20)
    max_objects: int = TARGET_TRACKING_CAPACITY
    object_features: int = 16
    global_features: int = 16
    tracker_capacity: int = TARGET_TRACKING_CAPACITY
    model_width: int = 192
    attention_layers: int = 2
    attention_heads: int = 4
    # DAgger refinement collects its own image-only state distribution instead
    # of reverting to teacher-only behavior cloning.
    bootstrap_with_teacher_behavior: bool = False
    smoke_test: bool = False
    resume: bool = False
    resume_replay: str = ""
    refine_replay_only: bool = False
    refinement_epochs: int = 20
    device: str = "cuda"


def _generated_collection_seed_footprint(
    config: TrackedDAggerConfig, seed: int
) -> set[int]:
    """Return newly allocated collection seeds, excluding deliberate replays."""
    values = {int(seed)}
    for round_index in range(1, int(config.rounds) + 1):
        first = int(seed) + round_index * _COLLECTION_ROUND_SEED_STRIDE
        values.update(range(first, first + int(config.num_envs)))
    return values


def _collection_seed_footprint(
    config: TrackedDAggerConfig, seed: int
) -> set[int]:
    """Return new collection seeds plus explicitly requested failure replays."""
    return _generated_collection_seed_footprint(config, seed) | {
        int(value) for value in config.collection_seed_list
    }


def _evaluation_seed_footprint(
    config: TrackedDAggerConfig, seed: int
) -> set[int]:
    """Return the fixed held-out episode seeds assigned to this run."""
    return set(range(int(seed), int(seed) + int(config.evaluation_episodes)))


def _seed_footprint_from_saved_config(data: dict[str, Any]) -> set[int]:
    """Read seed assignments from a historical run config."""
    values: set[int] = set()
    collection_seed = data.get("collection_seed")
    if collection_seed is not None:
        collection_seed = int(collection_seed)
        values.add(collection_seed)
        values.update(int(value) for value in data.get("collection_seed_list", ()))
        rounds = max(0, int(data.get("rounds", 0)))
        num_envs = max(0, int(data.get("num_envs", 0)))
        for round_index in range(1, rounds + 1):
            first = collection_seed + round_index * _COLLECTION_ROUND_SEED_STRIDE
            values.update(range(first, first + num_envs))
    evaluation_seed = data.get("evaluation_seed")
    if evaluation_seed is not None:
        evaluation_seed = int(evaluation_seed)
        episodes = max(1, int(data.get("evaluation_episodes", 1)))
        values.update(range(evaluation_seed, evaluation_seed + episodes))
    return values


def _historical_run_seed_footprint(output_dir: Path) -> set[int]:
    """Collect assigned seeds from prior primary runs, excluding this output."""
    values: set[int] = set()
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.is_dir():
        return values
    current_config = output_dir.resolve() / "config.json"
    for path in runs_dir.glob("visual_set_v*/config.json"):
        if path.resolve() == current_config:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(data, dict):
            values.update(_seed_footprint_from_saved_config(data))
    return values


def _allocate_seed(
    footprint: Callable[[int], set[int]],
    unavailable: set[int],
    explicit_seed: int | None,
    label: str,
) -> int:
    """Allocate one seed whose complete run footprint is not already assigned."""
    if explicit_seed is not None:
        candidate = int(explicit_seed)
        if candidate < 0:
            raise ValueError(f"{label} must be non-negative")
        overlap = footprint(candidate) & unavailable
        if overlap:
            example = min(overlap)
            raise ValueError(
                f"{label} reuses an assigned project seed ({example}); omit the "
                f"option to generate a fresh seed"
            )
        return candidate
    while True:
        candidate = secrets.randbelow(_AUTOMATIC_SEED_LIMIT)
        if not (footprint(candidate) & unavailable):
            return candidate


def _resolve_run_seeds(config: TrackedDAggerConfig) -> None:
    """Resolve automatic seeds for a new run or restore them for a resume."""
    output = Path(config.output_dir)
    if config.resume:
        config_path = output / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError("resume requires an existing tracked run config")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        for name in ("collection_seed", "evaluation_seed"):
            saved_value = saved.get(name)
            current_value = getattr(config, name)
            if saved_value is None:
                raise ValueError(f"resume config is missing {name}")
            if current_value is not None and int(current_value) != int(saved_value):
                raise ValueError(
                    f"resume must reuse {name}: saved={saved_value!r} "
                    f"current={current_value!r}"
                )
            setattr(config, name, int(saved_value))
        saved_seed_list = tuple(
            int(value) for value in saved.get("collection_seed_list", ())
        )
        if config.collection_seed_list and config.collection_seed_list != saved_seed_list:
            raise ValueError("resume must reuse collection_seed_list")
        config.collection_seed_list = saved_seed_list
        return

    unavailable = _historical_run_seed_footprint(output)
    replay_seeds = {int(value) for value in config.collection_seed_list}
    collection_seed = _allocate_seed(
        lambda seed: _generated_collection_seed_footprint(config, seed),
        unavailable | replay_seeds,
        config.collection_seed,
        "collection_seed",
    )
    collection_values = _collection_seed_footprint(config, collection_seed)
    expected_collection_values = len(
        _generated_collection_seed_footprint(config, collection_seed)
    ) + len(config.collection_seed_list)
    if len(collection_values) != expected_collection_values:
        raise ValueError("collection seeds overlap within the configured run")
    unavailable.update(collection_values)
    evaluation_seed = _allocate_seed(
        lambda seed: _evaluation_seed_footprint(config, seed),
        unavailable,
        config.evaluation_seed,
        "evaluation_seed",
    )
    config.collection_seed = collection_seed
    config.evaluation_seed = evaluation_seed


class TrackedReplay:
    def __init__(
        self, capacity: int, spec: TrackedPolicySpec, horizon_count: int
    ) -> None:
        self.capacity = int(capacity)
        self.spec = spec
        self.objects = np.empty(
            (capacity, spec.max_objects, spec.object_features), np.float16
        )
        self.masks = np.empty((capacity, spec.max_objects), np.bool_)
        self.globals = np.empty((capacity, spec.global_features), np.float16)
        self.actions = np.empty(capacity, np.int8)
        self.behavior_actions = np.empty(capacity, np.int8)
        self.exploration = np.empty(capacity, np.bool_)
        self.regrets = np.empty((capacity, len(BarrageVisionEnv.ACTIONS)), np.float16)
        self.collisions = np.empty(
            (capacity, horizon_count, len(BarrageVisionEnv.ACTIONS)), np.bool_
        )
        self.episode_ids = np.empty(capacity, np.int64)
        self.episode_steps = np.empty(capacity, np.int32)
        self.priorities = np.empty(capacity, np.float32)
        self.size = 0
        self.position = 0

    def add(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        actions: np.ndarray,
        regrets: np.ndarray,
        collisions: np.ndarray,
        episode_ids: np.ndarray,
        episode_steps: np.ndarray | None = None,
        priorities: np.ndarray | None = None,
        behavior_actions: np.ndarray | None = None,
        exploration: np.ndarray | None = None,
    ) -> None:
        count = len(objects)
        if count > self.capacity:
            start = count - self.capacity
            return self.add(
                objects[start:], masks[start:], globals_[start:], actions[start:],
                regrets[start:], collisions[start:], episode_ids[start:],
                None if episode_steps is None else episode_steps[start:],
                None if priorities is None else priorities[start:],
                None if behavior_actions is None else behavior_actions[start:],
                None if exploration is None else exploration[start:],
            )
        source_objects = np.asarray(objects)
        source_masks = np.asarray(masks, dtype=np.bool_)
        source_globals = np.asarray(globals_)
        if source_objects.ndim != 3 or source_masks.ndim != 2:
            raise ValueError("tracked replay objects/masks have invalid rank")
        if source_globals.ndim != 2:
            raise ValueError("tracked replay globals have invalid rank")
        object_count = min(source_objects.shape[1], self.spec.max_objects)
        feature_count = min(
            source_objects.shape[2], self.spec.object_features
        )
        global_count = min(source_globals.shape[1], self.spec.global_features)
        source_actions = np.asarray(actions)
        source_regrets = np.asarray(regrets)
        source_collisions = np.asarray(collisions)
        source_episode_ids = np.asarray(episode_ids)
        source_episode_steps = (
            None
            if episode_steps is None
            else np.asarray(episode_steps, dtype=np.int32)
        )
        source_priorities = (
            None
            if priorities is None
            else np.asarray(priorities, dtype=np.float32)
        )

        # The ring has at most two contiguous spans.  Slice assignment avoids
        # constructing a potentially million-element advanced index and lets
        # NumPy use bulk copies.  Exact-shape replays also avoid the old redundant
        # zero-fill before every copy; only migrated padding needs clearing.
        first_count = min(count, self.capacity - self.position)
        spans = (
            (self.position, 0, first_count),
            (0, first_count, count - first_count),
        )
        clear_objects = (
            object_count < self.spec.max_objects
            or feature_count < self.spec.object_features
        )
        clear_masks = object_count < self.spec.max_objects
        clear_globals = global_count < self.spec.global_features
        for destination_start, source_start, span_count in spans:
            if span_count == 0:
                continue
            destination = slice(destination_start, destination_start + span_count)
            source = slice(source_start, source_start + span_count)
            if clear_objects:
                self.objects[destination].fill(0.0)
            if clear_masks:
                self.masks[destination].fill(False)
            if clear_globals:
                self.globals[destination].fill(0.0)
            self.objects[
                destination, :object_count, :feature_count
            ] = source_objects[source, :object_count, :feature_count]
            self.masks[destination, :object_count] = source_masks[
                source, :object_count
            ]
            self.globals[destination, :global_count] = source_globals[
                source, :global_count
            ]
            self.actions[destination] = source_actions[source]
            self.behavior_actions[destination] = (source_actions if behavior_actions is None else behavior_actions)[source]
            self.exploration[destination] = False if exploration is None else exploration[source]
            self.regrets[destination] = source_regrets[source]
            self.collisions[destination] = source_collisions[source]
            self.episode_ids[destination] = source_episode_ids[source]
            if source_episode_steps is None:
                self.episode_steps[destination].fill(0)
            else:
                self.episode_steps[destination] = source_episode_steps[source]
            if source_priorities is None:
                self.priorities[destination].fill(1.0)
            else:
                self.priorities[destination] = source_priorities[source]
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def save(self, path: Path) -> None:
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            temporary,
            objects=self.objects[: self.size],
            masks=self.masks[: self.size],
            globals=self.globals[: self.size],
            actions=self.actions[: self.size],
            behavior_actions=self.behavior_actions[: self.size],
            exploration=self.exploration[: self.size],
            replay_schema=np.asarray(2, np.int64),
            regrets=self.regrets[: self.size],
            collisions=self.collisions[: self.size],
            episode_ids=self.episode_ids[: self.size],
            episode_steps=self.episode_steps[: self.size],
            priorities=self.priorities[: self.size],
            position=np.asarray(self.position, np.int64),
        )
        os.replace(temporary, path)

    @classmethod
    def load(
        cls,
        path: Path,
        capacity: int,
        spec: TrackedPolicySpec,
        horizon_count: int,
    ) -> "TrackedReplay":
        replay = cls(capacity, spec, horizon_count)
        with np.load(path, allow_pickle=False) as data:
            count = len(data["objects"])
            if count > capacity:
                raise ValueError("saved tracked replay exceeds configured capacity")
            replay.add(
                data["objects"],
                data["masks"],
                data["globals"],
                data["actions"],
                data["regrets"],
                data["collisions"],
                data["episode_ids"],
                data["episode_steps"] if "episode_steps" in data else None,
                data["priorities"] if "priorities" in data else None,
                data["behavior_actions"] if "behavior_actions" in data else None,
                data["exploration"] if "exploration" in data else None,
            )
            # A full source replay commonly has a wrapped position of zero.
            # When loading it into a larger buffer, append into the unused tail
            # instead of immediately overwriting the retained anchor samples.
            replay.position = count if count < capacity else int(data["position"])
        return replay


def _teacher_distribution(
    regrets: torch.Tensor, collisions: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Imitate teacher continuation costs; collision labels are replay diagnostics."""
    logits = -regrets / max(float(temperature), 1e-4)
    return nn.functional.softmax(logits, dim=1)


def _loss(
    model: ActionQueryPolicy,
    objects: torch.Tensor,
    masks: torch.Tensor,
    globals_: torch.Tensor,
    actions: torch.Tensor,
    regrets: torch.Tensor,
    collisions: torch.Tensor,
    config: TrackedDAggerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy, teacher_cost, _ = model(objects, masks, globals_)
    teacher_probability = _teacher_distribution(
        regrets, collisions, config.teacher_temperature
    )
    policy_loss = nn.functional.kl_div(
        nn.functional.log_softmax(policy, dim=1),
        teacher_probability,
        reduction="batchmean",
    )
    cost_loss = nn.functional.smooth_l1_loss(
        teacher_cost, torch.clamp(regrets / 20.0, 0.0, 1.0)
    )
    loss = (
        config.policy_weight * policy_loss
        + config.teacher_cost_weight * cost_loss
    )
    return loss, {
        "loss": float(loss.item()),
        "policy": float(policy_loss.item()),
        "cost": float(cost_loss.item()),
        "accuracy": float((policy.argmax(dim=1) == actions).float().mean().item()),
    }


def _configure_trainable_scope(
    model: ActionQueryPolicy, scope: str
) -> list[nn.Parameter]:
    """Train both output heads and backbone, or freeze both heads."""
    if scope not in {"full", "backbone"}:
        raise ValueError(f"unsupported trainable scope: {scope}")
    output_head_prefixes = (
        "policy_head.",
        "teacher_cost_head.",
    )
    selected: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            scope == "full"
            or (scope == "backbone" and not name.startswith(output_head_prefixes))
        )
        if parameter.requires_grad:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"trainable scope selected no parameters: {scope}")
    return selected


def _load_migrated_model_state(
    model: nn.Module, source_state: dict[str, torch.Tensor]
) -> dict[str, list[str]]:
    """Load a smaller tracked model by copying every overlapping tensor slice."""
    destination = model.state_dict()
    exact: list[str] = []
    expanded: list[str] = []
    skipped: list[str] = []
    for name, target in destination.items():
        source = source_state.get(name)
        if source is None or source.ndim != target.ndim:
            skipped.append(name)
            continue
        source = source.to(device=target.device, dtype=target.dtype)
        if source.shape == target.shape:
            destination[name] = source
            exact.append(name)
            continue
        if any(
            source.shape[index] > target.shape[index]
            for index in range(target.ndim)
        ):
            skipped.append(name)
            continue
        # Zero new rows/columns so migration initially preserves the source
        # behavior; newly introduced density/ambiguity inputs are learned by
        # DAgger instead of injecting random actions before collection.
        migrated = torch.zeros_like(target)
        overlap = tuple(slice(0, size) for size in source.shape)
        migrated[overlap] = source
        destination[name] = migrated
        expanded.append(name)
    model.load_state_dict(destination)
    return {"exact": exact, "expanded": expanded, "skipped": skipped}


def _validate_plain_dagger_warm_start(
    checkpoint: Mapping[str, Any], checkpoint_path: str | Path
) -> None:
    """Reject checkpoints whose deployed actions are not produced by ``model``.

    Plain tracked DAgger restores only the action-query backbone state.  A
    distilled student, option arbiter, sequence gate, or similar controller can
    override that backbone at deployment, so silently dropping it changes the
    policy used to produce the checkpoint's reported evaluation score.
    """

    markers = sorted(
        str(key)
        for key in checkpoint
        if str(key) in _COMPOSITE_DEPLOYMENT_KEYS
        or str(key).startswith(_COMPOSITE_DEPLOYMENT_KEY_PREFIXES)
        or str(key).endswith("_controller")
        or str(key).startswith("controller_")
    )
    inference_head = checkpoint.get("inference_head")
    if inference_head not in (None, "policy"):
        markers.append(f"inference_head={inference_head!r}")
    if not markers:
        return

    marker_text = ", ".join(markers)
    raise ValueError(
        "initial checkpoint cannot be used as a plain tracked DAgger warm-start "
        "because it contains composite deployment action semantics "
        f"({marker_text}): {checkpoint_path}. Plain DAgger restores only the "
        "ActionQueryPolicy backbone in checkpoint['model'], which would discard "
        "the deployed controller and silently change behavior. Use the evaluated "
        "pre-composition backbone checkpoint, or continue with the corresponding "
        "distillation, option, or controller training pipeline."
    )


def _resume_initial_checkpoint_path(output: Path, saved_config: Mapping[str, Any]) -> tuple[Path | None, bool]:
    recorded = []
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("initial_checkpoint"):
            recorded.append(str(manifest["initial_checkpoint"]))
    if saved_config.get("initial_checkpoint"):
        recorded.append(str(saved_config["initial_checkpoint"]))
    for value in recorded:
        raw = Path(value)
        for candidate in ((raw,) if raw.is_absolute() else (PROJECT_ROOT / raw, output / raw)):
            if candidate.is_file(): return candidate.resolve(), True
    return None, bool(recorded)


def _has_trusted_round0_baseline(
    output: Path,
    history: list[dict[str, Any]],
    saved_config: Mapping[str, Any],
) -> bool:
    """Verify that round 0 is a restorable, formally evaluated plain candidate."""
    row: dict[str, Any] | None = None
    for candidate_row in history:
        try:
            is_round0 = int(float(candidate_row.get("round", -1))) == 0
            no_collection = float(candidate_row["new_samples"]) == 0.0
            finite_selection = all(np.isfinite(float(candidate_row[name])) for name in (
                "model_iqm", "success_at_limit", "model_mean"
            ))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            is_round0
            and candidate_row.get("phase") == "initial_baseline"
            and no_collection
            and finite_selection
        ):
            row = candidate_row
            break
    if row is None:
        return False

    candidate_path = output / "round0" / "candidate.pt"
    evaluation_dir = output / "round0" / "evaluation"
    try:
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        evaluation_config = json.loads(
            (evaluation_dir / "evaluation_config.json").read_text(encoding="utf-8")
        )
        evaluation_summary = json.loads(
            (evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8")
        )
    except (EOFError, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False
    if not isinstance(candidate, Mapping) or not all(
        name in candidate for name in ("model", "optimizer")
    ):
        return False
    try:
        if int(candidate.get("round", -1)) != 0:
            return False
        _validate_plain_dagger_warm_start(candidate, candidate_path)
    except (TypeError, ValueError):
        return False
    if not isinstance(evaluation_config, Mapping) or not isinstance(
        evaluation_summary, Mapping
    ):
        return False

    reference_path = evaluation_dir / "evaluated_model.pt"
    if not reference_path.is_file():
        return False
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not contents_equal(candidate["model"], reference.get("model")):
        return False
    expected_evaluation = {
        "episodes": saved_config.get("evaluation_episodes"),
        "seed": saved_config.get("evaluation_seed"),
        "episode_limit_seconds": saved_config.get(
            "evaluation_episode_limit_seconds"
        ),
        "bullet_count": saved_config.get(
            "evaluation_bullet_count", saved_config.get("bullet_count")
        ),
        "targeted_bullet_probability": saved_config.get(
            "targeted_bullet_probability"
        ),
        "rendered_rgb": saved_config.get("deployment_rgb_observation"),
        "causal_action_delay_steps": saved_config.get(
            "evaluation_causal_action_delay_steps"
        ),
    }
    for name, expected in expected_evaluation.items():
        if expected is not None and evaluation_config.get(name) != expected:
            return False
    for name in ("model_iqm", "success_at_limit", "model_mean"):
        try:
            if not np.isclose(
                float(row[name]),
                float(evaluation_summary[name]),
                rtol=0.0,
                atol=1e-9,
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _validate_resume_plain_dagger_warm_start(
    output: Path,
    saved_config: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """Reject unsafe legacy resumes whose original deployed policy was composite."""
    checkpoint_path, was_recorded = _resume_initial_checkpoint_path(
        output, saved_config
    )
    if checkpoint_path is None:
        if was_recorded:
            print(
                "tracked_resume_warning original initial checkpoint is unavailable; "
                "retaining legacy resume compatibility",
                flush=True,
            )
        return
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (EOFError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(
            "resume initial checkpoint could not be inspected: "
            f"{checkpoint_path}"
        ) from error
    reference_path = output / "initial_model_reference.pt"
    if reference_path.is_file():
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        if not contents_equal(checkpoint.get("model"), reference.get("model")):
            raise ValueError("resume initial checkpoint content mismatch")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"resume initial checkpoint is not a checkpoint mapping: {checkpoint_path}"
        )
    try:
        _validate_plain_dagger_warm_start(checkpoint, checkpoint_path)
    except ValueError as error:
        if _has_trusted_round0_baseline(output, history, saved_config):
            return
        raise ValueError(
            "resume source uses composite deployment action semantics and the run "
            "has no trusted round0 initial_baseline. Continuing would silently "
            "restore a different plain backbone policy. Start a new main run at "
            "runs/visual_set_v46 from a pre-composition backbone checkpoint."
        ) from error


def _batch(
    replay: TrackedReplay, indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(replay.objects[indices], device=device, dtype=torch.float32),
        torch.as_tensor(replay.masks[indices], device=device, dtype=torch.bool),
        torch.as_tensor(replay.globals[indices], device=device, dtype=torch.float32),
        torch.as_tensor(replay.actions[indices], device=device, dtype=torch.long),
        torch.as_tensor(replay.regrets[indices], device=device, dtype=torch.float32),
        torch.as_tensor(replay.collisions[indices], device=device, dtype=torch.float32),
    )


def _episode_validation_mask(episode_ids: np.ndarray) -> np.ndarray:
    """Assign whole episodes to a stable, approximately 10% holdout.

    Episode ids encode round, environment index, and serial in decimal blocks.
    Taking ``id % 10`` therefore selected almost every serial-zero trajectory
    and badly skewed the old holdout toward the first episode in each worker.
    A seeded episode-local random draw keeps every state from one logical
    episode on the same side of the split.
    """
    unique, inverse = np.unique(episode_ids, return_inverse=True)
    validation = np.array([np.random.default_rng(int(value) % (2**64)).random() < .1
                           for value in unique], dtype=bool)
    return validation[inverse]


def _prioritized_epoch_indices(
    training: np.ndarray,
    priorities: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix uniform coverage with weighted rare-danger replay."""
    training = np.asarray(training, dtype=np.int64)
    if len(training) == 0:
        return training
    fraction = float(np.clip(fraction, 0.0, 1.0))
    prioritized_count = int(round(len(training) * fraction))
    uniform_count = len(training) - prioritized_count
    parts: list[np.ndarray] = []
    if uniform_count:
        parts.append(rng.choice(training, size=uniform_count, replace=False))
    if prioritized_count:
        weights = np.asarray(priorities[training], dtype=np.float64)
        weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
        total = float(weights.sum())
        probability = None if total <= 0.0 else weights / total
        parts.append(
            rng.choice(
                training,
                size=prioritized_count,
                replace=True,
                p=probability,
            )
        )
    result = np.concatenate(parts) if len(parts) > 1 else parts[0]
    rng.shuffle(result)
    return result


def _renamespace_initial_replay_episodes(replay: TrackedReplay) -> int:
    """Put imported episode groups in a namespace disjoint from new rounds."""
    if replay.size == 0:
        return 0
    _, inverse = np.unique(
        replay.episode_ids[: replay.size], return_inverse=True
    )
    replay.episode_ids[: replay.size] = -(
        inverse.astype(np.int64, copy=False) + 1
    )
    return int(inverse.max(initial=-1) + 1)


def _collection_priorities(
    behavior_actions: np.ndarray,
    teacher_actions: np.ndarray,
    regrets: np.ndarray,
    collisions: np.ndarray,
    episode_steps: np.ndarray,
    config: TrackedDAggerConfig,
) -> np.ndarray:
    """Score useful, recoverable mistakes without chasing every disagreement."""
    behavior_actions = np.asarray(behavior_actions, dtype=np.int64)
    rows = np.arange(len(behavior_actions), dtype=np.int64)
    collisions = np.asarray(collisions, dtype=np.bool_)
    behavior_unsafe = collisions[rows, 0, behavior_actions]
    if config.priority_mode == "action_disagreement":
        signal = (
            np.asarray(behavior_actions) != np.asarray(teacher_actions)
        ).astype(np.float32)
        priorities = 1.0 + config.disagreement_priority * signal
        priorities += (
            config.unsafe_behavior_priority
            * behavior_unsafe.astype(np.float32)
        )
    elif config.priority_mode == "behavior_regret":
        selected_regret = np.asarray(regrets, dtype=np.float32)[
            rows, behavior_actions
        ]
        regret_fraction = np.clip(selected_regret / 20.0, 0.0, 1.0)
        selected_collision_fraction = collisions[
            rows, :, behavior_actions
        ].mean(axis=1, dtype=np.float32)
        # A horizon contributes only when at least one alternative remains safe.
        # This focuses replay on errors the student can still correct instead of
        # spending priority mass on already-unavoidable terminal states.
        recoverable_fraction = (~collisions).any(axis=2).mean(
            axis=1, dtype=np.float32
        )
        severity = np.maximum(regret_fraction, selected_collision_fraction)
        avoidable_immediate = behavior_unsafe & (~collisions[:, 0]).any(axis=1)
        priorities = 1.0 + config.regret_priority * severity * recoverable_fraction
        priorities += (
            config.unsafe_behavior_priority
            * avoidable_immediate.astype(np.float32)
        )
    else:
        raise ValueError(f"unsupported priority mode: {config.priority_mode}")
    steps = np.asarray(episode_steps)
    priorities *= np.where(
        steps < config.early_state_decisions, config.early_state_priority, 1.0
    )
    priorities *= np.where(
        steps >= config.late_state_decisions, config.late_state_priority, 1.0
    )
    if config.priority_cap > 0.0:
        priorities = np.minimum(priorities, config.priority_cap)
    return np.asarray(priorities, dtype=np.float32)


def _boost_failed_episode_tail(
    replay: TrackedReplay,
    episode_id: int,
    terminal_step: int,
    tail_decisions: int,
    multiplier: float,
    priority_cap: float = 0.0,
) -> int:
    """Raise replay priority for the causal window preceding a real collision."""
    if tail_decisions < 1 or multiplier < 1.0:
        return 0
    lower_step = max(0, int(terminal_step) - int(tail_decisions) + 1)
    active = slice(0, replay.size)
    mask = (
        (replay.episode_ids[active] == int(episode_id))
        & (replay.episode_steps[active] >= lower_step)
        & (replay.episode_steps[active] <= int(terminal_step))
    )
    count = int(mask.sum())
    active_priorities = replay.priorities[: replay.size]
    active_priorities[mask] *= float(multiplier)
    if priority_cap > 0.0:
        active_priorities[mask] = np.minimum(
            active_priorities[mask], float(priority_cap)
        )
    return count


def _train_round(
    model: ActionQueryPolicy,
    optimizer: torch.optim.Optimizer,
    replay: TrackedReplay,
    config: TrackedDAggerConfig,
    device: torch.device,
    round_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    indices = np.arange(replay.size)
    validation_mask = _episode_validation_mask(
        replay.episode_ids[: replay.size]
    )
    validation = indices[validation_mask]
    training = indices[~validation_mask]
    if not len(validation) or not len(training):
        split = max(1, replay.size // 10)
        validation, training = indices[:split], indices[split:]
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_metrics: dict[str, float] = {}
    stale = 0
    rng = np.random.default_rng(config.seed + round_index * 1000)
    print(
        f"tracked_train_start round={round_index} "
        f"training_samples={len(training)} validation_samples={len(validation)} "
        f"epochs={config.epochs_per_round} batch_size={config.batch_size}",
        flush=True,
    )
    for epoch in range(1, config.epochs_per_round + 1):
        shuffled = _prioritized_epoch_indices(
            training,
            replay.priorities,
            config.priority_sample_fraction,
            rng,
        )
        model.train()
        train_metrics: list[dict[str, float]] = []
        total_batches = max(
            1, (len(shuffled) + config.batch_size - 1) // config.batch_size
        )
        epoch_started = time.perf_counter()
        next_progress_at = epoch_started + _PROGRESS_REPORT_INTERVAL_SECONDS
        for batch_index, start in enumerate(
            range(0, len(shuffled), config.batch_size), start=1
        ):
            data = _batch(replay, shuffled[start : start + config.batch_size], device)
            loss, metrics = _loss(model, *data, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            train_metrics.append(metrics)
            now = time.perf_counter()
            if now >= next_progress_at and batch_index < total_batches:
                elapsed = now - epoch_started
                eta = elapsed / batch_index * (total_batches - batch_index)
                print(
                    f"tracked_train_progress round={round_index} epoch={epoch} "
                    f"batches={batch_index}/{total_batches} "
                    f"loss={float(np.mean([item['loss'] for item in train_metrics])):.4f} "
                    f"elapsed_seconds={elapsed:.0f} eta_seconds={eta:.0f}",
                    flush=True,
                )
                next_progress_at = now + _PROGRESS_REPORT_INTERVAL_SECONDS
        model.eval()
        validation_metrics: list[dict[str, float]] = []
        with torch.inference_mode():
            for start in range(0, len(validation), config.batch_size):
                data = _batch(
                    replay, validation[start : start + config.batch_size], device
                )
                _, metrics = _loss(model, *data, config)
                validation_metrics.append(metrics)
        aggregate = {
            f"train_{key}": float(np.mean([item[key] for item in train_metrics]))
            for key in train_metrics[0]
        }
        aggregate.update({
            f"validation_{key}": float(
                np.mean([item[key] for item in validation_metrics])
            )
            for key in validation_metrics[0]
        })
        aggregate["epoch"] = float(epoch)
        print(
            f"tracked_train round={round_index} epoch={epoch} "
            f"loss={aggregate['train_loss']:.4f} "
            f"val={aggregate['validation_loss']:.4f} "
            f"acc={aggregate['train_accuracy']:.3f} "
            f"val_acc={aggregate['validation_accuracy']:.3f}",
            flush=True,
        )
        if aggregate["validation_loss"] < best_validation:
            best_validation = aggregate["validation_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = aggregate
            stale = 0
        else:
            stale += 1
            if stale > config.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best_state, best_metrics


def _collect_round(
    model: ActionQueryPolicy,
    replay: TrackedReplay,
    config: TrackedDAggerConfig,
    spec: TrackedPolicySpec,
    device: torch.device,
    round_index: int,
) -> dict[str, float]:
    task = BarrageTaskSpec(
        bullet_count=config.bullet_count,
        targeted_bullet_probability=config.targeted_bullet_probability,
        observation_size=config.observation_size,
        episode_limit_seconds=config.collection_episode_seconds,
    )
    env_kwargs = task.env_kwargs()
    agent = TrackedPolicyAgent(model, device)
    from .deployment import configure_image_controller
    configure_image_controller(agent, config.pixel_guard, search_workers=config.search_workers)
    rng = np.random.default_rng(config.seed + round_index * 10_000)
    collected = 0
    agreement = 0
    priority_sum = 0.0
    priority_max = 0.0
    early_states = 0
    late_states = 0
    unsafe_behavior_states = 0
    failed_episodes = 0
    failure_tail_states = 0
    started = time.perf_counter()
    next_progress_at = started + _PROGRESS_REPORT_INTERVAL_SECONDS
    last_progress_at = started
    last_progress_samples = 0
    print(
        f"tracked_collection_start round={round_index} "
        f"samples={config.samples_per_round} envs={config.num_envs} "
        f"workers={config.cpu_workers} bullets={config.bullet_count}",
        flush=True,
    )
    from .branch_records import DecisionRecorder
    branch_output = Path(config.output_dir) / f"round{round_index}" / "branch_records"
    recorder = DecisionRecorder(branch_output) if config.record_branch_snapshots else None
    with ParallelTrackedDaggerEnv(
        env_count=config.num_envs,
        workers=config.cpu_workers,
        seed=config.collection_seed + round_index * _COLLECTION_ROUND_SEED_STRIDE,
        initial_episode_seeds=config.collection_seed_list,
        repeat_initial_episode_seeds=config.repeat_collection_seeds,
        env_kwargs=env_kwargs,
        spec=spec,
        safety_horizons=config.safety_horizons,
        teacher_kind=config.teacher_kind,
        teacher_horizon_seconds=config.teacher_horizon_seconds,
        teacher_reaction_seconds=config.teacher_reaction_seconds,
        deployment_rgb_observation=config.deployment_rgb_observation,
        causal_action_delay_steps=config.collection_causal_action_delay_steps,
        branch_output_dir=str(branch_output) if recorder else "",
    ) as pipeline:
        while collected < config.samples_per_round:
            teacher = pipeline.teacher_actions.copy()
            explore = np.zeros(config.num_envs, dtype=bool)
            if round_index == 1 and config.bootstrap_with_teacher_behavior:
                actions = teacher.copy()
            else:
                environment_indices = np.arange(
                    config.num_envs, dtype=np.int64
                )
                episode_ids = (
                    int(round_index) * 10**12
                    + environment_indices * 10**6
                    + pipeline.episode_serial
                )
                if recorder:
                    prior_records = recorder.before(agent, pipeline, episode_ids)
                    actions, decision_diagnostics = agent.act_features_with_diagnostics(
                        pipeline.objects, pipeline.masks, pipeline.globals, episode_indices=episode_ids)
                else:
                    actions = agent.act_features(pipeline.objects, pipeline.masks, pipeline.globals, episode_indices=episode_ids)
                explore = rng.random(config.num_envs) < config.random_action_probability
                actions[explore] = rng.integers(
                    0, len(BarrageVisionEnv.ACTIONS), size=int(explore.sum())
                )
                if recorder:
                    recorder.after(agent, pipeline, episode_ids, prior_records, actions, explore, decision_diagnostics)
            take = min(config.num_envs, config.samples_per_round - collected)
            environment_indices = np.arange(take, dtype=np.int64)
            episode_ids = (
                int(round_index) * 10**12
                + environment_indices * 10**6
                + pipeline.episode_serial[:take]
            )
            episode_steps = pipeline.episode_steps[:take].copy()
            behavior_actions = actions[:take]
            behavior_unsafe = pipeline.collisions[
                np.arange(take, dtype=np.int64), 0, behavior_actions
            ].astype(np.bool_)
            priorities = _collection_priorities(
                behavior_actions,
                teacher[:take],
                pipeline.regrets[:take],
                pipeline.collisions[:take],
                episode_steps,
                config,
            )
            early = episode_steps < config.early_state_decisions
            late = episode_steps >= config.late_state_decisions
            replay.add(
                pipeline.objects[:take],
                pipeline.masks[:take],
                pipeline.globals[:take],
                teacher[:take],
                pipeline.regrets[:take],
                pipeline.collisions[:take],
                episode_ids,
                episode_steps,
                priorities,
                behavior_actions=behavior_actions,
                exploration=explore[:take],
            )
            agreement += int(np.sum(actions[:take] == teacher[:take]))
            collected += take
            priority_sum += float(priorities.sum())
            priority_max = max(priority_max, float(priorities.max(initial=0.0)))
            early_states += int(early.sum())
            late_states += int(late.sum())
            unsafe_behavior_states += int(behavior_unsafe.sum())
            done, truncated, _ = pipeline.step(actions)
            if done.any():
                agent.reset_state(episode_ids[np.flatnonzero(done[:take])])
            failed = done[:take] & ~truncated[:take]
            for env_index in np.flatnonzero(failed):
                failed_episodes += 1
                failure_tail_states += _boost_failed_episode_tail(
                    replay,
                    int(episode_ids[env_index]),
                    int(episode_steps[env_index]),
                    config.failure_tail_decisions,
                    config.failure_tail_priority,
                    config.priority_cap,
                )
            now = time.perf_counter()
            if now >= next_progress_at or collected >= config.samples_per_round:
                elapsed = now - started
                total_rate = collected / max(elapsed, 1e-6)
                interval_rate = (collected - last_progress_samples) / max(
                    now - last_progress_at, 1e-6
                )
                eta = (
                    (config.samples_per_round - collected) / total_rate
                    if total_rate > 0.0
                    else float("inf")
                )
                print(
                    f"tracked_collection round={round_index} "
                    f"samples={collected}/{config.samples_per_round} "
                    f"progress={100.0 * collected / config.samples_per_round:.1f}% "
                    f"states_per_second={total_rate:.1f} "
                    f"interval_states_per_second={interval_rate:.1f} "
                    f"failed_episodes={failed_episodes} "
                    f"elapsed_seconds={elapsed:.0f} eta_seconds={eta:.0f}",
                    flush=True,
                )
                last_progress_at = now
                last_progress_samples = collected
                next_progress_at = now + _PROGRESS_REPORT_INTERVAL_SECONDS
    if recorder: recorder.close()
    elapsed = time.perf_counter() - started
    return {
        "new_samples": float(collected),
        "collection_seconds": elapsed,
        "collection_states_per_second": collected / max(elapsed, 1e-6),
        "behavior_teacher_agreement": agreement / max(collected, 1),
        "behavior_controller_decisions": float(agent.decision_count),
        "replay_priority_mean": priority_sum / max(collected, 1),
        "replay_priority_max": priority_max,
        "replay_early_state_fraction": early_states / max(collected, 1),
        "replay_late_state_fraction": late_states / max(collected, 1),
        "replay_unsafe_behavior_fraction": unsafe_behavior_states / max(collected, 1),
        "failed_episodes_discovered": float(failed_episodes),
        "failure_tail_states_boosted": float(failure_tail_states),
    }


def _checkpoint(
    model: ActionQueryPolicy,
    optimizer: torch.optim.Optimizer,
    config: TrackedDAggerConfig,
    spec: TrackedPolicySpec,
    round_index: int,
    training_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {**asdict(config), "physics_fps": PHYSICS_FPS},
        "tracked_policy_spec": asdict(spec),
        "model_version": model.model_version,
        "model_hparams": {
            "action_count": model.action_count,
            "width": model.width,
            "attention_layers": model.attention_layers,
            "attention_heads": model.attention_heads,
            "safety_horizons": model.safety_horizons,
            "geometry_statistics": model.geometry_statistics,
            "continuation_horizons": model.continuation_horizons,
            "continuation_weight": model.continuation_weight,
        },
        "observation_size": config.observation_size,
        "inference_head": "policy",
        "policy_architecture": "policy_teacher_cost",
        "round": round_index,
        "training_metrics": training_metrics,
    }


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _checkpoint_selection_key(
    metrics: dict[str, Any], selection_mode: str, episode_limit_seconds: float
) -> tuple[float, ...]:
    if selection_mode not in {"success_at_limit", "iqm_then_success_then_mean"}:
        raise ValueError(f"unsupported checkpoint selection mode: {selection_mode}")
    success = float(metrics["success_at_limit"])
    if not np.isfinite(success):
        raise ValueError("checkpoint selection metrics must be finite")
    # A strict comparison at promotion sites keeps the earlier checkpoint on ties.
    return (success,)


def _typed_history_row(row: dict[str, Any]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(value, str):
            typed[key] = value
            continue
        try:
            typed[key] = float(value)
        except ValueError:
            typed[key] = value
    return typed


def _next_round_index(history: list[dict[str, Any]]) -> int:
    """Return the first unused positive DAgger round index."""
    return max((int(float(row["round"])) for row in history), default=0) + 1


def _restore_selected_checkpoint(
    output: Path,
    history: list[dict[str, Any]],
    selection_mode: str,
    iqm_floor_seconds: float,
) -> tuple[float, ...]:
    """Recompute the configured winner on resume from immutable candidates."""
    if not history:
        return (-float("inf"),)
    keys = [
        _checkpoint_selection_key(row, selection_mode, iqm_floor_seconds)
        for row in history
    ]
    best_index = max(range(len(history)), key=keys.__getitem__)
    best_row = history[best_index]
    running_key: tuple[float, ...] = (-float("inf"),)
    for row, key in zip(history, keys):
        promoted = key > running_key
        if promoted:
            running_key = key
        row["selection_mode"] = selection_mode
        row["checkpoint_promoted"] = float(promoted)
    _write_history(output / "round_summaries.csv", history)
    selected_round = int(float(best_row["round"]))
    candidate = output / f"round{selected_round}" / "candidate.pt"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"selected round checkpoint is missing during resume: {candidate}"
        )
    atomic_copy(candidate, output / "best.pt")
    atomic_copy(candidate, output / "latest.pt")
    typed = _typed_history_row(best_row)
    typed["selection_mode"] = selection_mode
    atomic_write_json(output / "best_summary.json", typed)
    return _checkpoint_selection_key(best_row, selection_mode, iqm_floor_seconds)


def _finalize_round(
    *,
    output: Path,
    model: ActionQueryPolicy,
    optimizer: torch.optim.Optimizer,
    replay: TrackedReplay,
    config: TrackedDAggerConfig,
    spec: TrackedPolicySpec,
    device: torch.device,
    round_index: int,
    phase: str,
    collection: dict[str, float],
    training_metrics: dict[str, float],
    history: list[dict[str, Any]],
    best_path: Path,
    best_selection_key: tuple[float, ...],
    save_replay: bool = True,
) -> tuple[float, ...]:
    round_dir = output / f"round{round_index}"
    round_dir.mkdir(parents=True, exist_ok=True)
    candidate = round_dir / "candidate.pt"
    atomic_torch_save(
        _checkpoint(model, optimizer, config, spec, round_index, training_metrics),
        candidate,
    )
    print(
        f"tracked_evaluation_start round={round_index} "
        f"bullets={config.evaluation_bullet_count} "
        f"episodes={config.evaluation_episodes} "
        f"workers={config.evaluation_workers} batch_size={config.evaluation_batch_size}",
        flush=True,
    )
    evaluation = evaluate_tracked_checkpoint(
        str(candidate),
        episodes=config.evaluation_episodes,
        workers=config.evaluation_workers,
        seed=config.evaluation_seed,
        output_dir=str(round_dir / "evaluation"),
        device_name=config.device,
        episode_limit_seconds=config.evaluation_episode_limit_seconds,
        bullet_count=config.evaluation_bullet_count,
        targeted_bullet_probability=config.targeted_bullet_probability,
        rendered_rgb=config.deployment_rgb_observation,
        smoke_test=config.smoke_test,
        evaluation_batch_size=config.evaluation_batch_size,
        causal_action_delay_steps=config.evaluation_causal_action_delay_steps,
        analytic_shield=PRODUCTION_ANALYTIC_SHIELD,
        analytic_shield_gate=PRODUCTION_ANALYTIC_SHIELD_GATE,
        pixel_guard=config.pixel_guard,
        search_workers=config.search_workers,
    )
    row = {
        "round": float(round_index),
        "phase": phase,
        "replay_samples": float(replay.size),
        **collection,
        **training_metrics,
        **evaluation,
    }
    candidate_key = _checkpoint_selection_key(
        row, config.selection_mode, config.evaluation_episode_limit_seconds
    )
    row["selection_mode"] = config.selection_mode
    row["checkpoint_promoted"] = float(candidate_key > best_selection_key)
    history.append(row)
    _write_history(output / "round_summaries.csv", history)
    if save_replay:
        replay.save(output / "replay_latest.npz")
    if candidate_key > best_selection_key:
        best_selection_key = candidate_key
        atomic_copy(candidate, best_path)
        atomic_write_json(output / "best_summary.json", row)
    else:
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        optimizer.load_state_dict(best_checkpoint["optimizer"])
    atomic_copy(best_path, output / "latest.pt")
    save_round_summary_plot(
        output / "round_summaries.csv",
        output / "results.png",
        output / "config.json",
    )
    print(
        f"tracked_round={round_index} phase={phase} "
        f"iqm={evaluation['model_iqm']:.3f}s "
        f"cvar5={evaluation['model_cvar5']:.3f}s "
        f"selection={config.selection_mode} "
        f"best_key={best_selection_key}",
        flush=True,
    )
    return best_selection_key


def train_tracked_policy(config: TrackedDAggerConfig) -> Path:
    torch.set_num_threads(1)
    if config.record_branch_snapshots and config.bootstrap_with_teacher_behavior:
        raise ValueError("branch recording requires deployed student behavior")
    if config.evaluation_bullet_count < 1:
        raise ValueError("evaluation_bullet_count must be positive")
    _resolve_run_seeds(config)
    assert config.collection_seed is not None
    assert config.evaluation_seed is not None
    if not config.smoke_test and config.evaluation_episodes != 200:
        raise ValueError("tracked DAgger evaluation is locked to 200 episodes")
    if (
        not config.smoke_test
        and config.collection_causal_action_delay_steps != 0
    ):
        raise ValueError(
            "production tracked DAgger collection requires synchronous zero-delay "
            "actions"
        )
    if (
        not config.smoke_test
        and config.evaluation_causal_action_delay_steps != 0
    ):
        raise ValueError(
            "production tracked DAgger evaluation requires synchronous zero-delay "
            "actions"
        )
    if (
        not config.smoke_test
        and config.collection_episode_seconds
        < config.evaluation_episode_limit_seconds
    ):
        raise ValueError(
            "production collection must cover the complete evaluation horizon"
        )
    decisions_per_episode = int(np.ceil(
        config.collection_episode_seconds * DECISION_HZ
    ))
    if (
        not config.smoke_test
        and config.samples_per_round < config.num_envs * decisions_per_episode
    ):
        raise ValueError(
            "samples_per_round must cover at least one complete episode per environment"
        )
    if config.selection_mode not in {"success_at_limit", "iqm_then_success_then_mean"}:
        raise ValueError(f"unsupported checkpoint selection mode: {config.selection_mode}")
    if config.teacher_reaction_seconds <= 0.0:
        raise ValueError("teacher_reaction_seconds must be positive")
    if not 0.0 <= config.priority_sample_fraction <= 1.0:
        raise ValueError("priority_sample_fraction must be within [0, 1]")
    if not 0.0 <= config.random_action_probability <= 1.0:
        raise ValueError("random_action_probability must be within [0, 1]")
    if config.priority_mode not in {"action_disagreement", "behavior_regret"}:
        raise ValueError(f"unsupported priority mode: {config.priority_mode}")
    for name in (
        "disagreement_priority",
        "regret_priority",
        "unsafe_behavior_priority",
        "priority_cap",
        "early_state_priority",
        "late_state_priority",
        "failure_tail_priority",
    ):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if (
        config.refine_replay_only
        and not config.resume
        and not (config.initial_checkpoint and config.initial_replay)
    ):
        raise ValueError(
            "refine_replay_only requires --resume or both an initial checkpoint "
            "and initial replay"
        )
    if config.trainable_scope not in {"full", "backbone"}:
        raise ValueError(f"unsupported trainable scope: {config.trainable_scope}")
    if len(config.collection_seed_list) > config.num_envs:
        raise ValueError("collection_seed_list cannot exceed num_envs")
    if len(set(config.collection_seed_list)) != len(config.collection_seed_list):
        raise ValueError("collection_seed_list must contain unique seeds")
    if any(int(value) < 0 for value in config.collection_seed_list):
        raise ValueError("collection_seed_list seeds must be non-negative")
    if config.repeat_collection_seeds and not config.collection_seed_list:
        raise ValueError(
            "repeat_collection_seeds requires collection_seed_list"
        )
    if not config.resume:
        for label, value in (
            ("initial checkpoint", config.initial_checkpoint),
            ("initial replay", config.initial_replay),
        ):
            if value and not Path(value).is_file():
                raise FileNotFoundError(f"{label} does not exist: {value}")
    initial_source: Mapping[str, Any] | None = None
    resume_history: list[dict[str, Any]] | None = None
    if not config.resume and config.initial_checkpoint:
        initial_source = torch.load(
            config.initial_checkpoint, map_location="cpu", weights_only=False
        )
        _validate_plain_dagger_warm_start(
            initial_source, config.initial_checkpoint
        )
    output = Path(config.output_dir)
    if config.resume:
        if not (output / "config.json").exists():
            raise FileNotFoundError("resume requires an existing tracked run config")
        saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if "use_safety_filter" in saved_config or "collision_weight" in saved_config:
            raise ValueError("Legacy risk-head runs require a fresh output directory and --initial-checkpoint; their optimizer and evaluation history cannot be resumed")
        history_path = output / "round_summaries.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as file:
                resume_history = list(csv.DictReader(file))
        else:
            resume_history = []
        _validate_resume_plain_dagger_warm_start(
            output, saved_config, resume_history
        )
        immutable = (
            "evaluation_bullet_count",
            "replay_capacity", "observation_size", "bullet_count",
            "targeted_bullet_probability", "max_objects", "tracker_capacity",
            "object_features", "global_features", "model_width",
            "attention_layers", "attention_heads", "safety_horizons",
            "teacher_kind", "teacher_horizon_seconds",
            "teacher_reaction_seconds",
            "evaluate_initial_checkpoint",
            "bootstrap_with_teacher_behavior", "selection_mode",
            "collection_episode_seconds", "evaluation_episode_limit_seconds",
            "priority_sample_fraction",
            "priority_mode", "regret_priority", "priority_cap",
            "random_action_probability",
            "repeat_collection_seeds",
            "deployment_rgb_observation", "pixel_guard",
            "disagreement_priority", "unsafe_behavior_priority",
            "early_state_priority", "late_state_priority",
            "early_state_decisions", "late_state_decisions",
            "failure_tail_priority", "failure_tail_decisions",
            "trainable_scope",
            "collection_causal_action_delay_steps",
            "evaluation_causal_action_delay_steps",
        )
        current = asdict(config)
        for name in immutable:
            current_value = current[name]
            saved_value = saved_config.get(name, "off" if name == "pixel_guard" else current_value)
            if name == "evaluation_bullet_count":
                # Older runs evaluated at their training bullet count. Never
                # silently mix a new evaluation task with saved selection keys.
                saved_value = saved_config.get(name, saved_config["bullet_count"])
            if name == "safety_horizons":
                saved_value = tuple(saved_value)
                current_value = tuple(current_value)
            if saved_value != current_value:
                raise ValueError(
                    f"resume configuration mismatch for {name}: "
                    f"saved={saved_value!r} current={current_value!r}"
                )
    else:
        prepare_new_output(output)
        atomic_write_json(output / "config.json", {**asdict(config), "physics_fps": PHYSICS_FPS})
        manifest: dict[str, Any] = {
            "git_branch": git_revision(PROJECT_ROOT),
            "training_seed": config.seed,
            "collection_seed": config.collection_seed,
            "collection_seed_list": list(config.collection_seed_list),
            "repeat_collection_seeds": bool(config.repeat_collection_seeds),
            "validation_seed": config.evaluation_seed,
            "selection_metric": "success_at_limit",
            "selection_mode": config.selection_mode,
            "control_timing": {
                "collection_causal_action_delay_steps": int(
                    config.collection_causal_action_delay_steps
                ),
                "evaluation_causal_action_delay_steps": int(
                    config.evaluation_causal_action_delay_steps
                ),
                "initial_action": 0,
                "teacher_labels_after_pending_action": True,
            },
            "task": BarrageTaskSpec(
                bullet_count=config.bullet_count,
                targeted_bullet_probability=config.targeted_bullet_probability,
                observation_size=config.observation_size,
                episode_limit_seconds=config.evaluation_episode_limit_seconds,
            ).manifest(),
            "evaluation_task": BarrageTaskSpec(
                bullet_count=config.evaluation_bullet_count,
                targeted_bullet_probability=config.targeted_bullet_probability,
                observation_size=config.observation_size,
                episode_limit_seconds=config.evaluation_episode_limit_seconds,
            ).manifest(),
            "source_files": list(_MANIFEST_SOURCE_FILES),
        }
        if config.initial_checkpoint:
            source_checkpoint = Path(config.initial_checkpoint)
            if not source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"initial checkpoint does not exist: {source_checkpoint}"
                )
            manifest["initial_checkpoint"] = str(source_checkpoint.resolve())
            atomic_torch_save({"model": initial_source["model"]}, output / "initial_model_reference.pt")
        if config.initial_replay:
            source_replay = Path(config.initial_replay)
            if not source_replay.is_file():
                raise FileNotFoundError(
                    f"initial replay does not exist: {source_replay}"
                )
            manifest["initial_replay"] = str(source_replay.resolve())
        atomic_write_json(output / "run_manifest.json", manifest)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    spec = TrackedPolicySpec(
        max_objects=config.max_objects,
        object_features=config.object_features,
        global_features=config.global_features,
        tracker_capacity=config.tracker_capacity,
        expected_bullet_count=config.bullet_count,
    )
    model = ActionQueryPolicy(
        spec,
        width=config.model_width,
        attention_layers=config.attention_layers,
        attention_heads=config.attention_heads,
        safety_horizons=config.safety_horizons,
    ).to(device)
    trainable_parameters = _configure_trainable_scope(
        model, config.trainable_scope
    )
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    print(
        f"tracked_model trainable_scope={config.trainable_scope} "
        f"trainable_parameters={trainable_parameter_count} "
        f"total_parameters={total_parameter_count}",
        flush=True,
    )
    if not config.resume:
        manifest["model_parameters"] = {
            "trainable_scope": config.trainable_scope,
            "trainable": trainable_parameter_count,
            "total": total_parameter_count,
        }
        atomic_write_json(output / "run_manifest.json", manifest)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = (
        [] if resume_history is None else resume_history
    )
    best_path = output / "best.pt"
    if config.resume:
        replay_source = (
            Path(config.resume_replay)
            if config.resume_replay
            else output / "replay_latest.npz"
        )
        if not replay_source.is_file():
            raise FileNotFoundError(f"resume replay does not exist: {replay_source}")
        replay = TrackedReplay.load(
            replay_source,
            config.replay_capacity,
            spec,
            len(config.safety_horizons),
        )
        best_selection_key = _restore_selected_checkpoint(
            output,
            history,
            config.selection_mode,
            config.evaluation_episode_limit_seconds,
        )
        latest = torch.load(output / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(latest["model"])
        optimizer.load_state_dict(latest["optimizer"])
        # Learning-rate changes are intentionally allowed between refinement
        # rounds. Optimizer state loading otherwise silently restores the old
        # value and defeats a conservative resume configuration.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(config.learning_rate)
            parameter_group["weight_decay"] = float(config.weight_decay)
        start_round = _next_round_index(history)
        if config.resume_replay:
            provenance_path = output / "resume_replay_inputs.json"
            provenance = (
                json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance_path.exists()
                else []
            )
            provenance.append({
                "round": start_round,
                "path": str(replay_source.resolve()),
            })
            atomic_write_json(provenance_path, provenance)
    else:
        if config.initial_checkpoint:
            if initial_source is None:
                raise RuntimeError("initial checkpoint preflight was not completed")
            source = initial_source
            source_spec_data = dict(source["tracked_policy_spec"])
            required_capacity_fields = {
                "tracker_capacity", "expected_bullet_count"
            }
            missing_capacity_fields = required_capacity_fields - source_spec_data.keys()
            if missing_capacity_fields:
                raise ValueError(
                    "initial checkpoint lacks the >100-bullet capacity metadata: "
                    + ", ".join(sorted(missing_capacity_fields))
                )
            source_spec = TrackedPolicySpec(**source_spec_data)
            if (
                source_spec.source_size != spec.source_size
                or source_spec.bullet_speed != spec.bullet_speed
                or source_spec.collision_radius != spec.collision_radius
                or source_spec.max_objects > spec.max_objects
                or source_spec.object_features > spec.object_features
                or source_spec.global_features > spec.global_features
            ):
                raise ValueError(
                    "initial checkpoint tracked policy spec cannot be expanded "
                    "into the configured run"
                )
            source_hparams = dict(source.get("model_hparams", {}))
            expected_hparams = {
                "action_count": model.action_count,
                "width": model.width,
                "attention_layers": model.attention_layers,
                "attention_heads": model.attention_heads,
                "safety_horizons": model.safety_horizons,
                "geometry_statistics": model.geometry_statistics,
            }
            for name, expected in expected_hparams.items():
                actual = source_hparams.get(name, expected)
                if name == "safety_horizons":
                    actual = tuple(actual)
                if name == "geometry_statistics":
                    actual = tuple(actual)
                if actual != expected:
                    raise ValueError(
                        f"initial checkpoint mismatch for {name}: "
                        f"source={actual!r} run={expected!r}"
                    )
            migration = _load_migrated_model_state(model, source["model"])
            atomic_write_json(
                output / "checkpoint_migration.json",
                {
                    "source_spec": asdict(source_spec),
                    "destination_spec": asdict(spec),
                    **migration,
                },
            )
        if config.initial_replay:
            replay = TrackedReplay.load(
                Path(config.initial_replay),
                config.replay_capacity,
                spec,
                len(config.safety_horizons),
            )
            _renamespace_initial_replay_episodes(replay)
        else:
            replay = TrackedReplay(
                config.replay_capacity, spec, len(config.safety_horizons)
            )
        start_round = 1
        best_selection_key = (-float("inf"),)
        if config.initial_checkpoint and config.evaluate_initial_checkpoint:
            best_selection_key = _finalize_round(
                output=output,
                model=model,
                optimizer=optimizer,
                replay=replay,
                config=config,
                spec=spec,
                device=device,
                round_index=0,
                phase="initial_baseline",
                collection={
                    "new_samples": 0.0,
                    "collection_seconds": 0.0,
                    "collection_states_per_second": 0.0,
                },
                training_metrics={},
                history=history,
                best_path=best_path,
                best_selection_key=best_selection_key,
            )
    if config.refine_replay_only:
        round_index = start_round
        config.epochs_per_round = int(config.refinement_epochs)
        _, training_metrics = _train_round(
            model, optimizer, replay, config, device, round_index
        )
        best_selection_key = _finalize_round(
            output=output,
            model=model,
            optimizer=optimizer,
            replay=replay,
            config=config,
            spec=spec,
            device=device,
            round_index=round_index,
            phase="replay_refinement",
            collection={
                "new_samples": 0.0,
                "collection_seconds": 0.0,
                "collection_states_per_second": 0.0,
            },
            training_metrics=training_metrics,
            history=history,
            best_path=best_path,
            best_selection_key=best_selection_key,
        )
        return best_path

    for round_index in range(start_round, config.rounds + 1):
        collection = _collect_round(
            model,
            replay,
            config,
            spec,
            device,
            round_index,
        )
        _, training_metrics = _train_round(
            model, optimizer, replay, config, device, round_index
        )
        best_selection_key = _finalize_round(
            output=output,
            model=model,
            optimizer=optimizer,
            replay=replay,
            config=config,
            spec=spec,
            device=device,
            round_index=round_index,
            phase="dagger_collection",
            collection=collection,
            training_metrics=training_metrics,
            history=history,
            best_path=best_path,
            best_selection_key=best_selection_key,
        )
    return best_path


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tracked action-query DAgger")
    parser.add_argument("--output-dir", default="runs/visual_set_v52")
    parser.add_argument(
        "--initial-checkpoint",
        default="diagnostics/risk_removal_20260920/policy_teacher_cost.pt",
    )
    parser.add_argument(
        "--initial-replay", default=""
    )
    parser.add_argument(
        "--skip-initial-evaluation",
        action="store_true",
        help=(
            "load the initial checkpoint directly into round 1 without a round0 "
            "evaluation"
        ),
    )
    parser.add_argument("--pixel-guard", choices=("off", "receding"), default="receding")
    parser.add_argument("--search-workers", type=int, default=9)
    parser.add_argument("--record-branch-snapshots", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--samples-per-round", type=int, default=400_000)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--trainable-scope",
        choices=("full", "backbone"),
        default="full",
    )
    parser.add_argument("--epochs-per-round", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=36)
    parser.add_argument("--cpu-workers", type=int, default=9)
    parser.add_argument("--evaluation-workers", type=int, default=10)
    parser.add_argument("--evaluation-bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument("--evaluation-batch-size", type=int, default=10)
    parser.add_argument(
        "--collection-causal-action-delay-steps",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--observation-size", type=int, default=384)
    parser.add_argument("--bullets", type=int, default=DEFAULT_TRAINING_BULLET_COUNT)
    parser.add_argument(
        "--max-objects", type=int, default=TARGET_TRACKING_CAPACITY
    )
    parser.add_argument(
        "--tracker-capacity", type=int, default=TARGET_TRACKING_CAPACITY
    )
    parser.add_argument(
        "--targeted-probability",
        type=float,
        default=TARGET_TASK.targeted_bullet_probability,
    )
    parser.add_argument("--collection-episode-seconds", type=float, default=120.0)
    parser.add_argument(
        "--collection-seed",
        type=int,
        default=None,
        help=(
            "explicit unique seed; by default a fresh project-unique "
            "seed is generated"
        ),
    )
    parser.add_argument(
        "--collection-seeds",
        default="",
        help=(
            "comma-separated one-shot hard seeds for the initial environments; "
            "remaining environments use --collection-seed"
        ),
    )
    parser.add_argument(
        "--repeat-collection-seeds",
        action="store_true",
        help=(
            "reset hard-seed environments to their assigned seed after every "
            "episode; exploration still perturbs the collected trajectories"
        ),
    )
    parser.add_argument("--priority-sample-fraction", type=float, default=0.75)
    parser.add_argument(
        "--priority-mode",
        choices=("action_disagreement", "behavior_regret"),
        default="action_disagreement",
    )
    parser.add_argument("--regret-priority", type=float, default=4.0)
    parser.add_argument(
        "--priority-cap",
        type=float,
        default=0.0,
        help="maximum replay priority; zero disables clipping",
    )
    parser.add_argument(
        "--random-action-probability",
        type=float,
        default=0.0,
        help="student-behavior exploration probability during collection",
    )
    parser.add_argument("--failure-tail-priority", type=float, default=12.0)
    parser.add_argument("--failure-tail-decisions", type=int, default=36)
    parser.add_argument("--ideal-semantic-collection", action="store_true")
    parser.add_argument(
        "--teacher-kind", choices=("exact", "recovery"), default="exact"
    )
    parser.add_argument(
        "--teacher-reaction-seconds",
        type=float,
        default=0.10,
        help=(
            "minimum duration for which each exact-teacher root action is held "
            "before reactive recovery"
        ),
    )
    parser.add_argument("--seed", type=int, default=4_701)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help=(
            "explicit unique fixed held-out seed; by default a fresh project-unique "
            "seed is generated"
        ),
    )
    parser.add_argument("--device", default="cuda")
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument(
        "--student-behavior-from-round1",
        dest="bootstrap_with_teacher_behavior",
        action="store_false",
    )
    behavior.add_argument(
        "--bootstrap-with-teacher-behavior",
        dest="bootstrap_with_teacher_behavior",
        action="store_true",
    )
    parser.set_defaults(bootstrap_with_teacher_behavior=False)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=256)
    parser.add_argument("--smoke-evaluation-episodes", type=int, default=2)
    parser.add_argument("--smoke-episode-seconds", type=float, default=5.0)
    parser.add_argument("--smoke-num-envs", type=int, default=2)
    parser.add_argument("--smoke-workers", type=int, default=2)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-replay",
        default="",
        help="optional audited replay artifact to load for this resume",
    )
    parser.add_argument("--refine-replay-only", action="store_true")
    parser.add_argument("--refinement-epochs", type=int, default=20)
    return parser


def _config_from_args(args: argparse.Namespace) -> TrackedDAggerConfig:
    config = TrackedDAggerConfig(
        output_dir=args.output_dir,
        initial_checkpoint=args.initial_checkpoint,
        initial_replay=args.initial_replay,
        pixel_guard=args.pixel_guard,
        search_workers=args.search_workers,
        record_branch_snapshots=args.record_branch_snapshots,
        evaluate_initial_checkpoint=not args.skip_initial_evaluation,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        trainable_scope=args.trainable_scope,
        epochs_per_round=args.epochs_per_round,
        num_envs=args.num_envs,
        cpu_workers=args.cpu_workers,
        evaluation_workers=args.evaluation_workers,
        evaluation_bullet_count=args.evaluation_bullets,
        evaluation_batch_size=args.evaluation_batch_size,
        collection_causal_action_delay_steps=(
            args.collection_causal_action_delay_steps
        ),
        evaluation_causal_action_delay_steps=(
            args.collection_causal_action_delay_steps
        ),
        observation_size=args.observation_size,
        bullet_count=args.bullets,
        max_objects=args.max_objects,
        tracker_capacity=args.tracker_capacity,
        targeted_bullet_probability=args.targeted_probability,
        collection_episode_seconds=args.collection_episode_seconds,
        collection_seed=args.collection_seed,
        collection_seed_list=tuple(
            int(value.strip())
            for value in args.collection_seeds.split(",")
            if value.strip()
        ),
        repeat_collection_seeds=args.repeat_collection_seeds,
        priority_sample_fraction=args.priority_sample_fraction,
        priority_mode=args.priority_mode,
        regret_priority=args.regret_priority,
        priority_cap=args.priority_cap,
        random_action_probability=args.random_action_probability,
        failure_tail_priority=args.failure_tail_priority,
        failure_tail_decisions=args.failure_tail_decisions,
        deployment_rgb_observation=not args.ideal_semantic_collection,
        teacher_kind=args.teacher_kind,
        teacher_reaction_seconds=args.teacher_reaction_seconds,
        seed=args.seed,
        evaluation_seed=args.evaluation_seed,
        device=args.device,
        bootstrap_with_teacher_behavior=args.bootstrap_with_teacher_behavior,
        smoke_test=args.smoke_test,
        resume=args.resume,
        resume_replay=args.resume_replay,
        refine_replay_only=args.refine_replay_only,
        refinement_epochs=args.refinement_epochs,
    )
    if config.smoke_test:
        # The default smoke path starts from scratch. An explicit request to
        # skip round0 keeps the configured warm-start so the complete
        # migration -> collection -> optimization path can be exercised
        # without spending time on an initial evaluation.
        if config.evaluate_initial_checkpoint:
            config.initial_checkpoint = ""
        config.initial_replay = ""
        config.rounds = 1
        for name, value in (
            ("smoke_samples", args.smoke_samples),
            ("smoke_evaluation_episodes", args.smoke_evaluation_episodes),
            ("smoke_num_envs", args.smoke_num_envs),
            ("smoke_workers", args.smoke_workers),
            ("smoke_epochs", args.smoke_epochs),
            ("smoke_batch_size", args.smoke_batch_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(args.smoke_episode_seconds) <= 0.0:
            raise ValueError("smoke_episode_seconds must be positive")
        config.samples_per_round = int(args.smoke_samples)
        config.replay_capacity = max(512, 2 * config.samples_per_round)
        config.epochs_per_round = int(args.smoke_epochs)
        config.batch_size = int(args.smoke_batch_size)
        config.num_envs = int(args.smoke_num_envs)
        config.cpu_workers = int(args.smoke_workers)
        config.evaluation_episodes = int(args.smoke_evaluation_episodes)
        config.evaluation_workers = int(args.smoke_workers)
        config.evaluation_episode_limit_seconds = float(args.smoke_episode_seconds)
        config.collection_episode_seconds = float(args.smoke_episode_seconds)
        config.device = "cpu"
    return config


def main() -> None:
    config = _config_from_args(_build_cli_parser().parse_args())
    train_tracked_policy(config)


if __name__ == "__main__":
    main()
