"""Recurrent DAgger pretraining for the screen-only barrage policy."""

import argparse
import csv
import io
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .artifacts import (
    atomic_copy, atomic_torch_save, atomic_write_json, atomic_write_text,
    git_revision, prepare_new_output, sha256_file,
)
from .baselines import fast_planner_supervision
from .env import PROJECT_ROOT, BarrageVisionEnv, BatchedBarrageEnv
from .evaluate_visual_set import evaluate
from .plot import save_round_summary_plot
from .scenarios import ScenarioSampler
from .visual_set import (
    ScreenOnlyAgent,
    SemanticFrameExtractor,
    VisualSetRecurrentQNetwork,
    VisualSetSpec,
)


@dataclass
class DAggerConfig:
    output_dir: str = "runs/visual_set_v9"
    initial_checkpoint: str = ""
    rounds: int = 10
    samples_per_round: int = 70_000
    replay_capacity: int = 350_000
    epochs_per_round: int = 8
    batch_size: int = 128
    sequence_length: int = 16
    sequence_stride: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_envs: int = 32
    bullet_count: int = 50
    collection_episode_seconds: float = 12.0
    evaluation_episode_limit_seconds: float = 120.0
    targeted_bullet_probability: float = 0.35
    random_action_probability: float = 0.08
    seed: int = 2_601
    max_objects: int = 64
    object_features: int = 12
    model_width: int = 192
    attention_layers: int = 2
    attention_heads: int = 4
    teacher_horizon_seconds: float = 0.60
    teacher_reaction_seconds: float = 0.30
    teacher_wall_margin: float = 80.0
    teacher_wall_penalty_weight: float = 0.350
    teacher_temperature: float = 0.08
    policy_weight: float = 1.0
    regret_weight: float = 0.5
    unsafe_policy_weight: float = 0.2
    validation_fraction: float = 0.15
    evaluation_episodes: int = 50
    evaluation_seed: int = 1_600_000
    evaluation_workers: int = 8
    core_bullet_size: int = 5
    core_bullet_speed: float = 240.0
    stress_targeted_bullet_probability: float = 0.50
    safety_horizons: Tuple[float, ...] = (0.10,)
    device: str = "cuda"


class EpisodeReplay:
    """Replay that evicts complete episodes so recurrent windows remain valid."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.data: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return 0 if not self.data else len(self.data["actions"])

    def add(self, batch: Dict[str, np.ndarray]) -> None:
        if not self.data:
            self.data = {key: value.copy() for key, value in batch.items()}
        else:
            self.data = {
                key: np.concatenate((self.data[key], batch[key]), axis=0)
                for key in self.data
            }
        if len(self) <= self.capacity:
            return
        episode_ids, inverse, counts = np.unique(
            self.data["episode_ids"], return_inverse=True, return_counts=True
        )
        priority_sum = np.bincount(
            inverse, weights=self.data["priorities"].astype(np.float64)
        )
        # Episode length must not buy retention probability.  Otherwise a
        # 2,000-step easy trajectory automatically evicts a 30-step failure.
        mean_priority = priority_sum / np.maximum(counts, 1)
        failure = np.bincount(
            inverse, weights=self.data["terminated"].astype(np.float64)
        ) > 0
        score = mean_priority + 2.0 * failure.astype(np.float64)
        probabilities = score / np.maximum(score.sum(), 1e-8)
        order = self.rng.choice(
            len(episode_ids), len(episode_ids), replace=False, p=probabilities
        )
        retained: List[int] = []
        retained_count = 0
        for position in order:
            episode_count = int(counts[position])
            if retained and retained_count + episode_count > self.capacity:
                continue
            retained.append(int(position))
            retained_count += episode_count
            if retained_count >= self.capacity:
                break
        keep = np.isin(inverse, np.asarray(retained, dtype=np.int64))
        self.data = {key: value[keep] for key, value in self.data.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, **self.data)
        os.replace(temporary, path)


def _student_probability(round_index: int, warm_start: bool = False) -> float:
    if warm_start:
        # A pretrained policy must query the teacher on its own state
        # distribution.  Reverting to expert-only rollouts is behaviour cloning
        # and caused a large covariate-shift regression in the first round.
        return 1.0
    schedule = (0.0, 0.55, 0.80, 1.0)
    return schedule[min(round_index - 1, len(schedule) - 1)]


def collect_round(
    config: DAggerConfig,
    round_index: int,
    model: VisualSetRecurrentQNetwork,
    extractor: SemanticFrameExtractor,
    device: torch.device,
    progress_path: Path,
) -> Dict[str, np.ndarray]:
    scenario_sampler = ScenarioSampler(
        config.seed + round_index * 100_000 + 31,
        core_size=config.core_bullet_size,
        core_speed=config.core_bullet_speed,
        targeted_probability=config.targeted_bullet_probability,
        stress_targeted_probability=config.stress_targeted_bullet_probability,
    )

    def next_scenario(_env_index: int) -> Dict[str, object]:
        return {"scenario": scenario_sampler.next()}

    envs = [
        BarrageVisionEnv(
            bullet_count=config.bullet_count,
            wall_collision=False,
            targeted_bullet_probability=config.targeted_bullet_probability,
            max_episode_seconds=config.collection_episode_seconds,
            scenario_mix=False,
            core_bullet_size=config.core_bullet_size,
            core_bullet_speed=config.core_bullet_speed,
            stress_targeted_bullet_probability=(
                config.stress_targeted_bullet_probability
            ),
        )
        for _ in range(config.num_envs)
    ]
    batch_env = BatchedBarrageEnv(envs, reset_options_factory=next_scenario)
    seed = config.seed + round_index * 100_000
    observations = batch_env.reset(seed)
    rng = np.random.default_rng(seed + 17)
    student_probability = _student_probability(
        round_index, warm_start=bool(config.initial_checkpoint)
    )
    wall_penalty_weight = config.teacher_wall_penalty_weight
    print(
        f"round={round_index} teacher_wall_penalty={wall_penalty_weight:.2f}",
        flush=True,
    )
    agent = ScreenOnlyAgent(
        model, extractor, device, inference_head="policy",
        use_safety_filter=False,
    )
    agent.reset(config.num_envs)
    episode_numbers = np.zeros(config.num_envs, dtype=np.int64)
    episode_steps = np.zeros(config.num_envs, dtype=np.int32)
    chunks: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "objects",
            "masks",
            "globals",
            "actions",
            "regrets",
            "collisions",
            "urgent",
            "priorities",
            "episode_ids",
            "episode_steps",
            "terminated",
            "scenario_source",
        )
    }
    collected = 0
    model.eval()
    def supervise(env: BarrageVisionEnv):
        return fast_planner_supervision(
            env,
            horizon_seconds=config.teacher_horizon_seconds,
            reaction_seconds=config.teacher_reaction_seconds,
            wall_margin=config.teacher_wall_margin,
            wall_penalty_weight=wall_penalty_weight,
            safety_horizons=config.safety_horizons,
        )

    try:
        while collected < config.samples_per_round:
            objects, masks, globals_ = extractor.extract_batch(observations)
            supervision = [supervise(env) for env in envs]
            teacher_actions = np.asarray([item.action for item in supervision], np.int64)
            regrets = np.stack([item.regrets for item in supervision])
            collisions = np.stack([item.safety_targets for item in supervision])
            urgent = np.asarray([item.urgent for item in supervision], np.bool_)
            student_actions = agent.act_features(objects, masks, globals_)
            behavior = teacher_actions.copy()
            use_student = rng.random(config.num_envs) < student_probability
            behavior[use_student] = student_actions[use_student]
            use_random = rng.random(config.num_envs) < config.random_action_probability
            for env_index in np.flatnonzero(use_random):
                safe_actions = np.flatnonzero(~collisions[env_index, 0].astype(bool))
                behavior[env_index] = int(
                    rng.choice(safe_actions)
                    if len(safe_actions)
                    else teacher_actions[env_index]
                )
            episode_ids = (
                round_index * 1_000_000_000
                + np.arange(config.num_envs, dtype=np.int64) * 1_000_000
                + episode_numbers
            )
            scenario_source = np.asarray(
                [
                    {"core": 0, "broad": 1, "stress": 2}[
                        env.current_scenario.source
                    ]
                    for env in envs
                ],
                dtype=np.uint8,
            )
            chosen_collision = collisions[
                np.arange(config.num_envs), :, student_actions
            ].any(axis=1)
            priorities = (
                1.0
                + 2.0 * urgent.astype(np.float32)
                + 2.0 * (student_actions != teacher_actions).astype(np.float32)
                + 3.0 * chosen_collision.astype(np.float32)
            )
            take = min(config.num_envs, config.samples_per_round - collected)
            values = {
                "objects": objects[:take].astype(np.float16),
                "masks": masks[:take],
                "globals": globals_[:take].astype(np.float16),
                "actions": teacher_actions[:take],
                "regrets": regrets[:take].astype(np.float16),
                "collisions": collisions[:take],
                "urgent": urgent[:take],
                "priorities": priorities[:take],
                "episode_ids": episode_ids[:take],
                "episode_steps": episode_steps[:take].copy(),
                "terminated": np.zeros(take, dtype=np.bool_),
                "scenario_source": scenario_source[:take],
            }
            for key, value in values.items():
                chunks[key].append(value)
            collected += take
            (
                next_observations, _, terminated, truncated, _, _
            ) = batch_env.step_detailed(behavior)
            done_mask = terminated | truncated
            chunks["terminated"][-1][:] = terminated[:take]
            if np.any(done_mask[:take]):
                chunks["priorities"][-1][done_mask[:take]] += 6.0
            episode_steps += 1
            episode_numbers[done_mask] += 1
            episode_steps[done_mask] = 0
            agent.reset_indices(done_mask)
            observations = next_observations
            if collected % 10_000 < config.num_envs:
                print(
                    f"round={round_index} collected={collected}/{config.samples_per_round} "
                    f"student_probability={student_probability:.2f}",
                    flush=True,
                )
                progress = {
                    key: np.concatenate(parts, axis=0)
                    for key, parts in chunks.items()
                }
                temporary = progress_path.with_name(progress_path.name + ".tmp")
                with temporary.open("wb") as file:
                    np.savez_compressed(file, **progress)
                os.replace(temporary, progress_path)
    finally:
        batch_env.close()
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _episode_split(
    episode_ids: np.ndarray, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episode_ids)
    # Stable multiplicative hashing prevents an episode used for training in an
    # early DAgger round from moving into validation when replay grows.
    unsigned = unique.astype(np.uint64, copy=False)
    hashed = (
        unsigned * np.uint64(11400714819323198485) + np.uint64(int(seed) & 0xFFFFFFFF)
    )
    threshold = int(np.clip(validation_fraction, 0.0, 1.0) * 10_000)
    validation_mask = (hashed % np.uint64(10_000)) < np.uint64(threshold)
    validation = unique[validation_mask]
    training = unique[~validation_mask]
    if len(validation) == 0:
        validation = unique[np.argmin(hashed)][None]
        training = unique[unique != validation[0]]
    if len(training) == 0:
        training = validation
    return training, validation


def _sequence_windows(
    episode_ids: np.ndarray,
    episode_steps: np.ndarray,
    allowed_episodes: np.ndarray,
    length: int,
    stride: int,
) -> np.ndarray:
    windows: List[np.ndarray] = []
    for episode_id in allowed_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        if len(indices) == 0:
            continue
        indices = indices[np.argsort(episode_steps[indices])]
        split_points = np.flatnonzero(np.diff(episode_steps[indices]) != 1) + 1
        for segment in np.split(indices, split_points):
            if len(segment) == 0:
                continue
            ends = list(range(0, len(segment), max(1, stride)))
            if ends[-1] != len(segment) - 1:
                ends.append(len(segment) - 1)
            for end in ends:
                start = max(0, end - length + 1)
                window = np.full(length, -1, dtype=np.int64)
                selected = segment[start : end + 1]
                # Right padding keeps every real prefix identical to deployment:
                # future zero padding cannot affect a causal GRU output.  The
                # former left padding advanced the hidden state through fake
                # observations before the first real frame.
                window[: len(selected)] = selected
                windows.append(window)
    if not windows:
        raise ValueError("no contiguous recurrent windows could be built")
    return np.stack(windows)


def _tensor_batch(
    data: Dict[str, np.ndarray], windows: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, ...]:
    valid = windows >= 0
    safe = np.maximum(windows, 0)
    objects = torch.as_tensor(data["objects"][safe], device=device, dtype=torch.float32)
    masks = torch.as_tensor(data["masks"][safe], device=device)
    globals_ = torch.as_tensor(data["globals"][safe], device=device, dtype=torch.float32)
    valid_tensor = torch.as_tensor(valid, device=device)
    objects = objects * valid_tensor[:, :, None, None]
    masks = masks & valid_tensor[:, :, None]
    globals_ = globals_ * valid_tensor[:, :, None]
    lengths = valid.sum(axis=1)
    endpoint_positions = lengths - 1
    endpoints = safe[np.arange(len(windows)), endpoint_positions]
    return (
        objects,
        masks,
        globals_,
        torch.as_tensor(endpoint_positions, device=device),
        torch.as_tensor(data["actions"][endpoints], device=device),
        torch.as_tensor(data["regrets"][endpoints], device=device, dtype=torch.float32),
        torch.as_tensor(data["collisions"][endpoints], device=device, dtype=torch.float32),
    )


def _stratified_training_windows(
    data: Dict[str, np.ndarray],
    windows: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample start, failure-tail, ordinary and stress states by fixed quotas."""
    valid = windows >= 0
    lengths = valid.sum(axis=1)
    endpoints = windows[np.arange(len(windows)), lengths - 1]
    episode_ids = data["episode_ids"][endpoints]
    steps = data["episode_steps"][endpoints]
    failed_episodes = np.unique(data["episode_ids"][data["terminated"].astype(bool)])
    failed = np.isin(episode_ids, failed_episodes)
    final_step = {}
    for episode_id in failed_episodes:
        episode_mask = data["episode_ids"] == episode_id
        final_step[int(episode_id)] = int(data["episode_steps"][episode_mask].max())
    failure_tail = np.asarray(
        [
            is_failed and step >= final_step[int(episode_id)] - 90
            for episode_id, step, is_failed in zip(episode_ids, steps, failed)
        ],
        dtype=np.bool_,
    )
    start = steps < 60
    failure_tail &= ~start
    stress = (data["scenario_source"][endpoints] == 2) & ~start & ~failure_tail
    ordinary = ~(start | failure_tail | stress)
    groups = (start, failure_tail, ordinary, stress)
    fractions = (0.30, 0.10, 0.50, 0.10)
    allocation = [int(round(count * fraction)) for fraction in fractions]
    allocation[2] += count - sum(allocation)
    selected = []
    priorities = data["priorities"][endpoints].astype(np.float64)
    all_indices = np.arange(len(windows))
    for group, amount in zip(groups, allocation):
        pool = np.flatnonzero(group)
        if len(pool) == 0:
            pool = all_indices
        weights = priorities[pool]
        weights = weights / np.maximum(weights.sum(), 1e-8)
        selected.append(rng.choice(pool, amount, replace=True, p=weights))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return windows[indices]


def _cost_sensitive_loss(
    policy_logits: torch.Tensor,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    regrets: torch.Tensor,
    collisions: torch.Tensor,
    config: DAggerConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    teacher_logits = -regrets / max(config.teacher_temperature, 1e-4)
    # Only imminent collisions are hard teacher constraints.  Longer horizons
    # assume the same command is held despite multiple future decision points.
    unsafe_actions = collisions[:, 0, :].bool()
    has_safe = (~unsafe_actions).any(dim=1, keepdim=True)
    teacher_logits = torch.where(
        unsafe_actions & has_safe,
        torch.full_like(teacher_logits, -80.0),
        teacher_logits,
    )
    teacher_probability = nn.functional.softmax(teacher_logits, dim=1)
    policy = nn.functional.kl_div(
        nn.functional.log_softmax(policy_logits, dim=1),
        teacher_probability,
        reduction="batchmean",
    )
    regret_target = -regrets / 20.0
    regret = nn.functional.smooth_l1_loss(q_values, regret_target)
    unsafe = (
        nn.functional.softmax(policy_logits, dim=1) * unsafe_actions.float()
    ).sum(dim=1).mean()
    loss = (
        config.policy_weight * policy
        + config.regret_weight * regret
        + config.unsafe_policy_weight * unsafe
    )
    metrics = {
        "policy": float(policy.item()),
        "regret": float(regret.item()),
        "unsafe": float(unsafe.item()),
        "actor": float((
            config.policy_weight * policy
            + config.regret_weight * regret
            + config.unsafe_policy_weight * unsafe
        ).item()),
        "correct": float((policy_logits.argmax(1) == actions).float().mean().item()),
    }
    return loss, metrics


def _save_checkpoint(
    path: Path,
    model: VisualSetRecurrentQNetwork,
    optimizer: torch.optim.Optimizer,
    config: DAggerConfig,
    spec: VisualSetSpec,
    round_index: int,
    epoch: int,
    validation_loss: float,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "visual_set_spec": asdict(spec),
            "model_version": model.model_version,
            "model_hparams": {
                "width": model.width,
                "attention_layers": model.attention_layers,
                "attention_heads": model.attention_heads,
                "safety_horizons": model.safety_horizons,
            },
            "safety_thresholds": [0.5] * len(model.safety_horizons),
            "use_safety_filter": False,
            "inference_head": "policy",
            "round": round_index,
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        path,
    )


def train_on_replay(
    config: DAggerConfig,
    round_index: int,
    replay: EpisodeReplay,
    model: VisualSetRecurrentQNetwork,
    optimizer: torch.optim.Optimizer,
    spec: VisualSetSpec,
    device: torch.device,
    round_dir: Path,
) -> Path:
    data = replay.data
    training_episodes, validation_episodes = _episode_split(
        data["episode_ids"], config.validation_fraction, config.seed
    )
    training_windows = _sequence_windows(
        data["episode_ids"], data["episode_steps"], training_episodes,
        config.sequence_length, config.sequence_stride,
    )
    validation_windows = _sequence_windows(
        data["episode_ids"], data["episode_steps"], validation_episodes,
        config.sequence_length, config.sequence_stride,
    )
    metrics_path = round_dir / "metrics.csv"
    best_path = round_dir / "validation_best.pt"
    latest_path = round_dir / "latest.pt"
    best_validation_actor = float("inf")
    started = time.perf_counter()
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["round", "epoch", "replay_samples", "train_loss", "validation_loss",
             "validation_actor_loss", "train_accuracy", "validation_accuracy",
             "elapsed_seconds"]
        )
        for epoch in range(1, config.epochs_per_round + 1):
            epoch_rng = np.random.default_rng(
                config.seed + round_index * 100 + epoch
            )
            shuffled = _stratified_training_windows(
                data, training_windows, len(training_windows), epoch_rng
            )
            model.train()
            train_losses: List[float] = []
            train_accuracy: List[float] = []
            for start in range(0, len(shuffled), config.batch_size):
                batch = _tensor_batch(data, shuffled[start : start + config.batch_size], device)
                policy, q_values, _, _ = model.forward_sequence(*batch[:3])
                rows = torch.arange(len(batch[3]), device=device)
                positions = batch[3].long()
                loss, metrics = _cost_sensitive_loss(
                    policy[rows, positions], q_values[rows, positions],
                    *batch[4:], config,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
                train_accuracy.append(metrics["correct"])
            model.eval()
            validation_losses: List[float] = []
            validation_actor_losses: List[float] = []
            validation_accuracy: List[float] = []
            with torch.inference_mode():
                for start in range(0, len(validation_windows), config.batch_size):
                    batch = _tensor_batch(
                        data, validation_windows[start : start + config.batch_size], device
                    )
                    policy, q_values, _, _ = model.forward_sequence(*batch[:3])
                    rows = torch.arange(len(batch[3]), device=device)
                    positions = batch[3].long()
                    loss, metrics = _cost_sensitive_loss(
                        policy[rows, positions], q_values[rows, positions],
                        *batch[4:], config,
                    )
                    validation_losses.append(float(loss.item()))
                    validation_actor_losses.append(metrics["actor"])
                    validation_accuracy.append(metrics["correct"])
            train_loss = float(np.mean(train_losses))
            validation_loss = float(np.mean(validation_losses))
            validation_actor_loss = float(np.mean(validation_actor_losses))
            train_acc = float(np.mean(train_accuracy))
            validation_acc = float(np.mean(validation_accuracy))
            writer.writerow([
                round_index, epoch, len(replay), train_loss, validation_loss,
                validation_actor_loss, train_acc, validation_acc,
                time.perf_counter() - started,
            ])
            file.flush()
            _save_checkpoint(
                latest_path, model, optimizer, config, spec, round_index, epoch,
                validation_loss,
            )
            # This checkpoint is the policy used to collect the next DAgger
            # round.  Select it by actor quality, not by an auxiliary safety
            # head that is deliberately disabled during stage one.
            if validation_actor_loss < best_validation_actor:
                best_validation_actor = validation_actor_loss
                _save_checkpoint(
                    best_path, model, optimizer, config, spec, round_index, epoch,
                    validation_loss,
                )
            print(
                f"round={round_index} epoch={epoch} loss={train_loss:.4f} "
                f"val={validation_loss:.4f} actor_val={validation_actor_loss:.4f} "
                f"acc={train_acc:.3f} val_acc={validation_acc:.3f}",
                flush=True,
            )
    return best_path


def train_dagger(config: DAggerConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    source = None
    if config.initial_checkpoint:
        source = torch.load(config.initial_checkpoint, map_location=device)
        source_hparams = source.get("model_hparams", {})
        source_spec = source.get("visual_set_spec", {})
        config.model_width = int(source_hparams.get("width", config.model_width))
        config.attention_layers = int(
            source_hparams.get("attention_layers", config.attention_layers)
        )
        config.attention_heads = int(
            source_hparams.get("attention_heads", config.attention_heads)
        )
        config.max_objects = int(source_spec.get("max_objects", config.max_objects))
        config.object_features = int(
            source_spec.get("object_features", config.object_features)
        )
    output = Path(config.output_dir)
    prepare_new_output(output)
    atomic_write_json(output / "config.json", asdict(config))
    atomic_write_json(
        output / "run_manifest.json",
        {
            "git_revision": git_revision(PROJECT_ROOT),
            "training_seed": config.seed,
            "validation_seed": config.evaluation_seed,
            "evaluation_episode_limit_seconds": config.evaluation_episode_limit_seconds,
            "model_version": VisualSetRecurrentQNetwork.model_version,
            "initial_checkpoint": config.initial_checkpoint,
            "initial_checkpoint_sha256": (
                sha256_file(Path(config.initial_checkpoint))
                if config.initial_checkpoint else ""
            ),
        },
    )
    spec = VisualSetSpec(
        max_objects=config.max_objects, object_features=config.object_features
    )
    extractor = SemanticFrameExtractor(spec)
    model = VisualSetRecurrentQNetwork(
        spec,
        len(BarrageVisionEnv.ACTIONS),
        config.model_width,
        config.attention_layers,
        config.attention_heads,
        config.safety_horizons,
    ).to(device)
    if source is not None:
        model.load_state_dict(source["model"], strict=True)
        print(
            f"warm_start={Path(config.initial_checkpoint).resolve()}", flush=True
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    replay = EpisodeReplay(config.replay_capacity, config.seed + 7)
    best_score = (float("-inf"),) * 5
    best_path = output / "best.pt"
    summaries = []
    for round_index in range(1, config.rounds + 1):
        round_dir = output / f"round{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        progress_path = round_dir / "collection_progress.npz"
        new_data = collect_round(
            config, round_index, model, extractor, device, progress_path
        )
        replay.add(new_data)
        replay.save(output / "replay_latest.npz")
        if progress_path.exists():
            progress_path.unlink()
        validation_best = train_on_replay(
            config, round_index, replay, model, optimizer, spec, device, round_dir
        )
        checkpoint = torch.load(validation_best, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        atomic_write_json(output / "config.json", asdict(config))
        evaluation = evaluate(
            str(validation_best), config.evaluation_episodes, config.bullet_count,
            config.evaluation_seed, config.device, str(round_dir / "evaluation"),
            config.targeted_bullet_probability, 40.0,
            config.evaluation_episode_limit_seconds,
            config.core_bullet_size, config.core_bullet_size,
            config.core_bullet_speed, config.core_bullet_speed,
            config.evaluation_workers,
            progress=True,
        )
        episode_ids, first_episode_rows = np.unique(
            new_data["episode_ids"], return_index=True
        )
        episode_sources = new_data["scenario_source"][first_episode_rows]
        summary = {
            "round": round_index,
            "new_samples": len(new_data["actions"]),
            "collection_episodes": len(episode_ids),
            "collection_core_episodes": int(np.sum(episode_sources == 0)),
            "collection_broad_episodes": int(np.sum(episode_sources == 1)),
            "collection_stress_episodes": int(np.sum(episode_sources == 2)),
            "collection_failures": int(new_data["terminated"].sum()),
            "replay_samples": len(replay),
            "student_probability": _student_probability(
                round_index, warm_start=bool(config.initial_checkpoint)
            ),
            "teacher_wall_penalty_weight": config.teacher_wall_penalty_weight,
            **evaluation,
        }
        summaries.append(summary)
        atomic_write_json(round_dir / "summary.json", summary)
        score = (
            evaluation["success_at_limit"],
            evaluation.get("model_cvar5", 0.0),
            evaluation.get("model_p5", 0.0),
            evaluation["model_p10"],
            evaluation["model_iqm"],
        )
        if score > best_score:
            best_score = score
            atomic_copy(validation_best, best_path)
            atomic_write_json(output / "best_summary.json", summary)
        summary_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(summary_buffer, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
        atomic_write_text(
            output / "round_summaries.csv", summary_buffer.getvalue()
        )
        save_round_summary_plot(
            output / "round_summaries.csv", output / "results.png",
            output / "config.json",
        )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recurrent screen-only DAgger policy")
    parser.add_argument("--output-dir", default="runs/visual_set_v9")
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--samples-per-round", type=int, default=70_000)
    parser.add_argument("--replay-capacity", type=int, default=350_000)
    parser.add_argument("--epochs-per-round", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--bullets", type=int, default=50)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument("--collection-episode-seconds", type=float, default=12.0)
    parser.add_argument("--evaluation-episode-limit-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=2_601)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = DAggerConfig(
        output_dir=args.output_dir,
        initial_checkpoint=args.initial_checkpoint,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        replay_capacity=args.replay_capacity,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        num_envs=args.num_envs,
        bullet_count=args.bullets,
        evaluation_episodes=args.evaluation_episodes,
        collection_episode_seconds=args.collection_episode_seconds,
        evaluation_episode_limit_seconds=args.evaluation_episode_limit_seconds,
        seed=args.seed,
        device=args.device,
    )
    if args.smoke_test:
        config.rounds = 1
        config.samples_per_round = 256
        config.replay_capacity = 256
        config.epochs_per_round = 1
        config.batch_size = 8
        config.num_envs = 4
        config.evaluation_episodes = 2
        config.collection_episode_seconds = 0.5
        config.sequence_length = 4
        config.sequence_stride = 2
        config.model_width = 64
        config.teacher_horizon_seconds = 0.30
        config.safety_horizons = (0.10,)
    checkpoint = train_dagger(config)
    print(f"best_checkpoint={checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
