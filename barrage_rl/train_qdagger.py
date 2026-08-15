"""Online recurrent Double-DQN fine-tuning with planner distillation."""

import argparse
import copy
import csv
import io
import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .artifacts import (
    atomic_torch_save,
    atomic_copy,
    atomic_write_json,
    atomic_write_text,
    git_revision,
    prepare_new_output,
    sha256_file,
)
from .baselines import privileged_planner_supervision
from .env import PROJECT_ROOT, BarrageVisionEnv, BatchedBarrageEnv
from .evaluate_visual_set import evaluate
from .plot import save_qdagger_results_plot
from .planner_pool import ProcessPlannerPool
from .visual_set import (
    ScreenOnlyAgent,
    SemanticFrameExtractor,
    VisualSetRecurrentQNetwork,
    VisualSetSpec,
    calibrate_safety_thresholds,
    safety_filter_is_usable,
)


@dataclass
class QDaggerConfig:
    checkpoint: str
    output_dir: str = "runs/visual_set_v9_qdagger"
    total_steps: int = 2_000_000
    learning_starts: int = 10_000
    replay_capacity: int = 400_000
    batch_size: int = 32
    sequence_length: int = 64
    burn_in: int = 16
    n_step: int = 15
    gradient_steps: int = 1
    learning_rate: float = 1e-4
    gamma: float = 0.9997
    discount_half_life_seconds: float = 90.0
    target_update_interval: int = 2_000
    checkpoint_interval: int = 100_000
    evaluation_interval: int = 200_000
    cpu_workers: int = 8
    async_updates: bool = True
    epsilon_start: float = 0.15
    epsilon_end: float = 0.02
    distillation_start: float = 1.0
    distillation_end: float = 0.20
    teacher_replay_steps: int = 50_000
    offline_pretraining_updates: int = 2_000
    teacher_batch_fraction: float = 0.20
    online_teacher_fraction: float = 0.25
    failure_tail_fraction: float = 0.30
    urgent_fraction: float = 0.20
    failure_tail_seconds: float = 3.0
    replay_chunk_steps: int = 512
    q_temperature: float = 0.05
    collision_weight: float = 0.25
    collision_positive_weight_max: float = 8.0
    policy_weight: float = 0.20
    num_envs: int = 22
    bullet_count: int = 50
    bullet_size_min: int = 1
    bullet_size_max: int = 7
    bullet_speed_min: float = 60.0
    bullet_speed_max: float = 300.0
    targeted_bullet_probability: float = 0.35
    max_episode_seconds: float = 180.0
    evaluation_episode_limit_seconds: float = 120.0
    evaluation_episodes: int = 200
    evaluation_workers: int = 8
    evaluation_seed: int = 1_800_000
    scenario_mix: bool = True
    core_bullet_size: int = 5
    core_bullet_speed: float = 240.0
    stress_targeted_bullet_probability: float = 0.50
    action_repeat: int = 4
    teacher_horizon_seconds: float = 1.20
    teacher_reaction_seconds: float = 0.30
    safety_horizons: Tuple[float, ...] = (0.10, 0.30, 0.60, 1.20)
    safety_thresholds: Tuple[float, ...] = (0.50, 0.50, 0.50, 0.50)
    safety_max_false_negative_rate: float = 0.01
    safety_max_all_unsafe_rate: float = 0.02
    safety_calibration_samples: int = 2_048
    use_safety_filter: bool = False
    seed: int = 3_701
    device: str = "cuda"


class EpisodeTransitionReplay:
    """Bounded replay of complete trajectories for recurrent TD learning."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.episodes: List[List[Dict[str, np.ndarray]]] = []
        self.failure_episodes: List[List[Dict[str, np.ndarray]]] = []
        self.urgent_entries: List[Tuple[List[Dict[str, np.ndarray]], np.ndarray]] = []
        self.transition_count = 0
        self.lock = threading.Lock()

    def __len__(self) -> int:
        with self.lock:
            return self.transition_count

    def add_episode(self, episode: List[Dict[str, np.ndarray]]) -> None:
        if not episode:
            return
        with self.lock:
            self.episodes.append(episode)
            if bool(episode[-1].get("terminals", episode[-1].get("dones", False))):
                self.failure_episodes.append(episode)
            urgent = np.flatnonzero(
                [bool(transition.get("urgent", False)) for transition in episode]
            )
            if len(urgent):
                self.urgent_entries.append((episode, urgent + 1))
            self.transition_count += len(episode)
            while self.transition_count > self.capacity and len(self.episodes) > 1:
                removed = self.episodes.pop(0)
                self.transition_count -= len(removed)
                self.failure_episodes = [
                    item for item in self.failure_episodes if item is not removed
                ]
                self.urgent_entries = [
                    item for item in self.urgent_entries if item[0] is not removed
                ]

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        failure_fraction: float = 0.0,
        urgent_fraction: float = 0.0,
        failure_tail_steps: int = 90,
    ) -> List[Tuple[List[Dict[str, np.ndarray]], bool]]:
        with self.lock:
            if not self.episodes:
                raise ValueError("cannot sample an empty replay")
            probabilities = np.asarray(
                [len(episode) for episode in self.episodes], np.float64
            )
            probabilities /= probabilities.sum()
            samples: List[Tuple[List[Dict[str, np.ndarray]], bool]] = []

            def append_window(episode, end: int) -> None:
                end = int(np.clip(end, 1, len(episode)))
                start = max(0, end - sequence_length)
                starts_at_episode_start = bool(
                    start == 0 and episode[0].get("episode_start", True)
                )
                samples.append((episode[start:end], starts_at_episode_start))

            failure_count = min(
                batch_size,
                int(round(batch_size * max(0.0, failure_fraction)))
                if self.failure_episodes else 0,
            )
            urgent_count = min(
                batch_size - failure_count,
                int(round(batch_size * max(0.0, urgent_fraction)))
                if self.urgent_entries else 0,
            )
            for _ in range(failure_count):
                episode = self.failure_episodes[
                    int(self.rng.integers(len(self.failure_episodes)))
                ]
                lower = max(1, len(episode) - max(1, int(failure_tail_steps)) + 1)
                append_window(episode, int(self.rng.integers(lower, len(episode) + 1)))
            for _ in range(urgent_count):
                episode, endpoints = self.urgent_entries[
                    int(self.rng.integers(len(self.urgent_entries)))
                ]
                append_window(episode, int(self.rng.choice(endpoints)))
            for _ in range(batch_size - len(samples)):
                episode = self.episodes[
                    int(self.rng.choice(len(self.episodes), p=probabilities))
                ]
                end = int(self.rng.integers(1, len(episode) + 1))
                append_window(episode, end)
            self.rng.shuffle(samples)
            return samples


def _stack_sequences(
    sequences: Sequence[Tuple[Sequence[Dict[str, np.ndarray]], bool]],
    length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    exemplar = sequences[0][0][0]
    batch = len(sequences)
    object_shape = exemplar["objects"].shape
    global_shape = exemplar["globals"].shape
    arrays = {
        # A single aligned state chain s_0 ... s_T is unrolled by both the
        # online and target networks.  Current and bootstrap Q values are then
        # taken at different positions in this same chain.  This is essential
        # for recurrent TD learning: independently unrolling state and
        # next-state sequences gives them incompatible hidden-state histories.
        "objects": np.zeros((batch, length + 1, *object_shape), np.float32),
        "masks": np.zeros((batch, length + 1, object_shape[0]), np.bool_),
        "globals": np.zeros((batch, length + 1, *global_shape), np.float32),
        "actions": np.zeros((batch, length), np.int64),
        "rewards": np.zeros((batch, length), np.float32),
        "terminals": np.ones((batch, length), np.float32),
        "truncations": np.zeros((batch, length), np.float32),
        "regrets": np.zeros((batch, length, len(BarrageVisionEnv.ACTIONS)), np.float32),
        "collisions": np.zeros(
            (batch, length, *exemplar["collisions"].shape), np.float32
        ),
        "teacher_valid": np.zeros((batch, length), np.bool_),
        "valid": np.zeros((batch, length), np.bool_),
        "starts_at_episode_start": np.zeros(batch, np.bool_),
    }
    transition_keys = (
        "actions", "rewards", "terminals", "truncations", "regrets", "collisions",
        "teacher_valid",
    )
    for row, (sequence, starts_at_episode_start) in enumerate(sequences):
        arrays["starts_at_episode_start"][row] = starts_at_episode_start
        for column, transition in enumerate(sequence):
            arrays["objects"][row, column] = transition["objects"]
            arrays["masks"][row, column] = transition["masks"]
            arrays["globals"][row, column] = transition["globals"]
            arrays["objects"][row, column + 1] = transition["next_objects"]
            arrays["masks"][row, column + 1] = transition["next_masks"]
            arrays["globals"][row, column + 1] = transition["next_globals"]
            arrays["valid"][row, column] = True
            for key in transition_keys:
                if key == "terminals":
                    value = transition.get("terminals", transition.get("dones", 0.0))
                elif key == "truncations":
                    value = transition.get("truncations", 0.0)
                elif key == "teacher_valid":
                    # Historical and offline-teacher transitions are fully
                    # supervised.  Online transitions explicitly opt out when
                    # the exact MPC teacher was not queried for that state.
                    value = transition.get("teacher_valid", True)
                else:
                    value = transition[key]
                arrays[key][row, column] = value
    return tuple(
        torch.as_tensor(arrays[key], device=device)
        for key in (
            "objects", "masks", "globals", "actions", "rewards", "terminals",
            "truncations",
            "regrets", "collisions", "teacher_valid", "valid",
            "starts_at_episode_start",
        )
    )


def _n_step_double_q_targets(
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    truncations: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    n_step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build n-step Double-DQN targets from one recurrently aligned unroll."""
    if online_q.shape != target_q.shape or online_q.ndim != 3:
        raise ValueError("online_q and target_q must have matching [B, T+1, A] shapes")
    if not (
        rewards.shape == terminals.shape == truncations.shape == valid.shape
    ):
        raise ValueError(
            "rewards, terminals, truncations, and valid must match [B, T]"
        )
    n_step = max(1, int(n_step))
    batch, length = rewards.shape
    returns = torch.zeros_like(rewards)
    discount = torch.ones_like(rewards)
    alive = valid.bool().clone()
    boundary_seen = torch.zeros_like(valid, dtype=torch.bool)
    state_actions = online_q.argmax(dim=-1, keepdim=True)
    state_values = target_q.gather(-1, state_actions).squeeze(-1)

    for offset in range(n_step):
        shifted_reward = torch.zeros_like(rewards)
        shifted_terminal = torch.zeros_like(terminals)
        shifted_truncation = torch.zeros_like(truncations)
        shifted_next_value = torch.zeros_like(rewards)
        shifted_valid = torch.zeros_like(valid, dtype=torch.bool)
        if offset < length:
            width = length - offset
            shifted_reward[:, :width] = rewards[:, offset:]
            shifted_terminal[:, :width] = terminals[:, offset:]
            shifted_truncation[:, :width] = truncations[:, offset:]
            shifted_valid[:, :width] = valid[:, offset:]
            shifted_next_value[:, :width] = state_values[
                :, offset + 1 : offset + 1 + width
            ]
        active = alive & shifted_valid
        returns = returns + discount * shifted_reward * active.to(rewards.dtype)
        terminal = active & shifted_terminal.bool()
        truncation = active & shifted_truncation.bool() & ~terminal
        returns = returns + (
            discount * float(gamma) * shifted_next_value
            * truncation.to(rewards.dtype)
        )
        boundary_seen |= terminal | truncation
        alive = active & ~terminal & ~truncation
        discount = discount * float(gamma)

    bootstrap_q = torch.zeros_like(rewards)
    bootstrap_available = torch.zeros_like(valid, dtype=torch.bool)
    available_width = min(length, max(online_q.shape[1] - n_step, 0))
    if available_width:
        bootstrap_q[:, :available_width] = state_values[
            :, n_step : n_step + available_width
        ]
        bootstrap_available[:, :available_width] = True

    can_bootstrap = alive & bootstrap_available
    returns = returns + discount * bootstrap_q * can_bootstrap.to(rewards.dtype)
    target_valid = valid.bool() & (boundary_seen | can_bootstrap)
    return returns, target_valid


def _linear_schedule(start: float, end: float, progress: float) -> float:
    return start + np.clip(progress, 0.0, 1.0) * (end - start)


def _extract_features(
    extractor: SemanticFrameExtractor,
    observations: np.ndarray,
    executor: Optional[ThreadPoolExecutor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if executor is None:
        return extractor.extract_batch(observations)
    extracted = list(executor.map(extractor.extract, observations))
    objects, masks, globals_ = zip(*extracted)
    return np.stack(objects), np.stack(masks), np.stack(globals_)


def _planner_supervision_batch(
    envs: Sequence[BarrageVisionEnv],
    executor: Optional[ThreadPoolExecutor],
    planner_pool: Optional[ProcessPlannerPool],
    safety_horizons: Sequence[float],
    horizon_seconds: float,
    reaction_seconds: float,
) -> list:
    if planner_pool is not None:
        return planner_pool.supervise(envs)

    def supervise(env: BarrageVisionEnv):
        return privileged_planner_supervision(
            env,
            horizon_seconds=horizon_seconds,
            reaction_seconds=reaction_seconds,
            safety_horizons=safety_horizons,
        )
    if executor is None:
        return [supervise(env) for env in envs]
    return list(executor.map(supervise, envs))


def _teacher_distribution(
    regrets: torch.Tensor, collisions: torch.Tensor, temperature: float = 0.08
) -> torch.Tensor:
    logits = -regrets / max(temperature, 1e-4)
    # Only the nearest horizon is an immediate action constraint.  The longer
    # constant-action rollouts remain auxiliary supervision for the risk head.
    unsafe = collisions.select(dim=-2, index=0).bool() if collisions.ndim == regrets.ndim + 1 else collisions.bool()
    has_safe = (~unsafe).any(dim=-1, keepdim=True)
    logits = torch.where(
        unsafe & has_safe, torch.full_like(logits, -80.0), logits
    )
    return nn.functional.softmax(logits, dim=-1)


def _update(
    model: VisualSetRecurrentQNetwork,
    target: VisualSetRecurrentQNetwork,
    optimizer: torch.optim.Optimizer,
    sequences: Sequence[Tuple[Sequence[Dict[str, np.ndarray]], bool]],
    config: QDaggerConfig,
    device: torch.device,
    distillation_weight: float,
) -> Dict[str, float]:
    model.train()
    batch = _stack_sequences(
        sequences, config.sequence_length, device,
    )
    (
        objects, masks, globals_, actions, rewards, terminals, truncations,
        regrets, collisions, teacher_valid,
        valid, starts_at_episode_start,
    ) = batch
    policy_all, online_q_all, collision_all, _ = model.forward_sequence(
        objects.float(), masks.bool(), globals_.float()
    )
    with torch.no_grad():
        _, target_q_all, _, _ = target.forward_sequence(
            objects.float(), masks.bool(), globals_.float()
        )
        td_target, target_valid = _n_step_double_q_targets(
            online_q_all.detach(), target_q_all, rewards, terminals, truncations, valid,
            config.gamma, config.n_step,
        )
    policy_logits = policy_all[:, :-1]
    q_values = online_q_all[:, :-1]
    collision_logits = collision_all[:, :-1]
    chosen_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    time_indices = torch.arange(config.sequence_length, device=device)[None, :]
    burn_in = torch.where(
        starts_at_episode_start[:, None],
        torch.zeros_like(time_indices),
        torch.full_like(time_indices, config.burn_in),
    )
    distillation_mask = valid & (time_indices >= burn_in)
    td_mask = distillation_mask & target_valid
    teacher_mask = distillation_mask & teacher_valid.bool()
    if torch.any(td_mask):
        td_loss = nn.functional.smooth_l1_loss(
            chosen_q[td_mask], td_target[td_mask]
        )
    else:
        td_loss = chosen_q.sum() * 0.0
    teacher = _teacher_distribution(regrets, collisions)
    if torch.any(teacher_mask):
        q_distillation = nn.functional.kl_div(
            nn.functional.log_softmax(q_values / config.q_temperature, dim=-1),
            teacher,
            reduction="none",
        ).sum(dim=-1)[teacher_mask].mean()
        policy_distillation = nn.functional.kl_div(
            nn.functional.log_softmax(policy_logits, dim=-1),
            teacher,
            reduction="none",
        ).sum(dim=-1)[teacher_mask].mean()
        collision_loss = nn.functional.binary_cross_entropy_with_logits(
            collision_logits[teacher_mask], collisions[teacher_mask],
            pos_weight=torch.clamp(
                (1.0 - collisions[teacher_mask].mean(dim=0))
                / collisions[teacher_mask].mean(dim=0).clamp_min(1e-3),
                1.0,
                config.collision_positive_weight_max,
            ),
        )
    else:
        zero = chosen_q.sum() * 0.0
        q_distillation = zero
        policy_distillation = zero
        collision_loss = zero
    loss = (
        td_loss
        + distillation_weight * q_distillation
        + config.policy_weight * policy_distillation
        + config.collision_weight * collision_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "td_loss": float(td_loss.item()),
        "q_distillation": float(q_distillation.item()),
    }


def _update_many(
    model: VisualSetRecurrentQNetwork,
    target: VisualSetRecurrentQNetwork,
    optimizer: torch.optim.Optimizer,
    sampled_batches: Sequence[
        Sequence[Tuple[Sequence[Dict[str, np.ndarray]], bool]]
    ],
    config: QDaggerConfig,
    device: torch.device,
    distillation_weight: float,
) -> Dict[str, float]:
    last: Dict[str, float] = {}
    for sequences in sampled_batches:
        last = _update(
            model, target, optimizer, sequences, config, device,
            distillation_weight,
        )
    return last


def _collect_teacher_replay(
    batch_env: BatchedBarrageEnv,
    extractor: SemanticFrameExtractor,
    executor: Optional[ThreadPoolExecutor],
    planner_pool: Optional[ProcessPlannerPool],
    replay: EpisodeTransitionReplay,
    transition_target: int,
    seed: int,
    safety_horizons: Sequence[float],
    teacher_horizon_seconds: float,
    teacher_reaction_seconds: float,
) -> None:
    """Collect privileged-planner trajectories for offline Q calibration."""
    transition_target = max(0, int(transition_target))
    if transition_target == 0:
        return
    observations = batch_env.reset(seed)
    objects, masks, globals_ = _extract_features(extractor, observations, None)
    active_episodes: List[List[Dict[str, np.ndarray]]] = [
        [] for _ in range(batch_env.num_envs)
    ]
    generated = 0
    next_log = max(transition_target // 10, batch_env.num_envs)
    while generated < transition_target:
        supervision = _planner_supervision_batch(
            batch_env.envs, executor, planner_pool, safety_horizons, teacher_horizon_seconds,
            teacher_reaction_seconds,
        )
        actions = np.asarray([item.action for item in supervision], np.int64)
        (
            next_observations, rewards, terminals, truncations,
            final_observations, _,
        ) = batch_env.step_detailed(actions)
        next_objects, next_masks, next_globals = _extract_features(
            extractor, next_observations, None
        )
        final_objects, final_masks, final_globals = _extract_features(
            extractor, final_observations, None
        )
        done_mask = terminals | truncations
        for index in range(batch_env.num_envs):
            active_episodes[index].append(
                {
                    "objects": objects[index].astype(np.float16),
                    "masks": masks[index],
                    "globals": globals_[index].astype(np.float16),
                    "next_objects": (
                        final_objects[index] if done_mask[index] else next_objects[index]
                    ).astype(np.float16),
                    "next_masks": (
                        final_masks[index] if done_mask[index] else next_masks[index]
                    ),
                    "next_globals": (
                        final_globals[index] if done_mask[index] else next_globals[index]
                    ).astype(np.float16),
                    "actions": np.asarray(actions[index], np.int64),
                    "rewards": np.asarray(rewards[index], np.float32),
                    "terminals": np.asarray(terminals[index], np.float32),
                    "truncations": np.asarray(truncations[index], np.float32),
                    "regrets": supervision[index].regrets.astype(np.float16),
                    "collisions": supervision[index].safety_targets.astype(np.float16),
                    "teacher_valid": np.asarray(True, np.bool_),
                    "urgent": np.asarray(supervision[index].urgent, np.bool_),
                }
            )
            if done_mask[index]:
                replay.add_episode(active_episodes[index])
                active_episodes[index] = []
        generated += batch_env.num_envs
        if generated >= next_log:
            print(
                f"teacher_replay generated={min(generated, transition_target)}/"
                f"{transition_target} committed={len(replay)}",
                flush=True,
            )
            next_log += max(transition_target // 10, batch_env.num_envs)
        observations = next_observations
        objects, masks, globals_ = next_objects, next_masks, next_globals

    # A partial trajectory is still valid teacher data.  Its last n-1
    # transitions simply will not receive a TD loss until a terminal or full
    # n-step target is available; distillation remains valid throughout.
    for episode in active_episodes:
        replay.add_episode(episode)


def _offline_pretrain_q(
    model: VisualSetRecurrentQNetwork,
    target: VisualSetRecurrentQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: EpisodeTransitionReplay,
    config: QDaggerConfig,
    device: torch.device,
) -> None:
    """Calibrate the inherited regret head as a Bellman Q head before acting."""
    updates = max(0, int(config.offline_pretraining_updates))
    if updates == 0:
        target.load_state_dict(model.state_dict())
        return
    if len(replay) == 0:
        raise ValueError("offline Q pretraining requires a non-empty teacher replay")
    target_period = max(
        1, int(np.ceil(config.target_update_interval / max(config.num_envs, 1)))
    )
    model.train()
    for update_index in range(1, updates + 1):
        sequences = replay.sample(config.batch_size, config.sequence_length)
        metrics = _update(
            model, target, optimizer, sequences, config, device,
            config.distillation_start,
        )
        if update_index % target_period == 0:
            target.load_state_dict(model.state_dict())
            target.memory.flatten_parameters()
        if update_index == 1 or update_index % max(updates // 20, 1) == 0:
            print(
                f"offline_pretrain update={update_index}/{updates} "
                f"loss={metrics['loss']:.4f} td={metrics['td_loss']:.4f} "
                f"distill={metrics['q_distillation']:.4f}",
                flush=True,
            )
    target.load_state_dict(model.state_dict())
    target.memory.flatten_parameters()
    model.eval()


def _calibrate_safety_on_replay(
    model: VisualSetRecurrentQNetwork,
    replay: EpisodeTransitionReplay,
    config: QDaggerConfig,
    device: torch.device,
) -> Tuple[Tuple[float, ...], bool]:
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    remaining = max(1, int(config.safety_calibration_samples))
    model.eval()
    with torch.inference_mode():
        while remaining > 0:
            batch_size = min(config.batch_size, remaining)
            batch = _stack_sequences(
                replay.sample(batch_size, config.sequence_length),
                config.sequence_length,
                device,
            )
            objects, masks, globals_ = batch[:3]
            labels = batch[8]
            teacher_valid = batch[9].bool()
            valid = batch[10].bool() & teacher_valid
            _, _, logits, _ = model.forward_sequence(
                objects.float(), masks.bool(), globals_.float()
            )
            probabilities.append(torch.sigmoid(logits[:, :-1])[valid].cpu().numpy())
            targets.append(labels[valid].bool().cpu().numpy())
            remaining -= batch_size
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets)
    thresholds = calibrate_safety_thresholds(
        probability_array, target_array,
        config.safety_max_false_negative_rate,
        config.safety_max_all_unsafe_rate,
    )
    usable = safety_filter_is_usable(
        probability_array,
        target_array,
        thresholds,
        maximum_false_negative_rate=max(
            0.05, 2.0 * config.safety_max_false_negative_rate
        ),
        maximum_all_unsafe_rate=max(
            0.025, 1.25 * config.safety_max_all_unsafe_rate
        ),
    )
    return thresholds, usable


def _save_checkpoint(
    path: Path,
    model: VisualSetRecurrentQNetwork,
    optimizer: torch.optim.Optimizer,
    config: QDaggerConfig,
    spec: VisualSetSpec,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "safety_thresholds": list(
                getattr(config, "safety_thresholds", (0.5,) * len(model.safety_horizons))
            ),
            "use_safety_filter": bool(config.use_safety_filter),
            "inference_head": "q",
            "global_step": global_step,
        },
        path,
    )


def _evaluate_checkpoint(
    checkpoint_path: Path,
    config: QDaggerConfig,
    global_step: int,
    output: Path,
) -> Dict[str, float]:
    evaluation = evaluate(
        str(checkpoint_path), config.evaluation_episodes, config.bullet_count,
        config.evaluation_seed, config.device,
        str(output / "evaluations" / f"step_{global_step:09d}"),
        config.targeted_bullet_probability, 40.0,
        config.evaluation_episode_limit_seconds,
        config.core_bullet_size, config.core_bullet_size,
        config.core_bullet_speed, config.core_bullet_speed,
        config.evaluation_workers,
        progress_interval=10,
        print_summary=False,
    )
    return {
        "global_step": float(global_step),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        **evaluation,
    }


def _write_evaluation_history(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    file = io.StringIO(newline="")
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, file.getvalue())


def train_qdagger(config: QDaggerConfig) -> Path:
    if config.n_step < 1 or config.n_step > config.sequence_length:
        raise ValueError("n_step must be between 1 and sequence_length")
    if config.offline_pretraining_updates > 0 and config.teacher_replay_steps <= 0:
        raise ValueError(
            "teacher_replay_steps must be positive when offline pretraining is enabled"
        )
    if config.teacher_batch_fraction < 0.0 or config.teacher_batch_fraction >= 1.0:
        raise ValueError("teacher_batch_fraction must be in [0, 1)")
    if not 0.0 <= config.online_teacher_fraction <= 1.0:
        raise ValueError("online_teacher_fraction must be in [0, 1]")
    if config.discount_half_life_seconds <= 0:
        raise ValueError("discount_half_life_seconds must be positive")
    if config.collision_positive_weight_max < 1.0:
        raise ValueError("collision_positive_weight_max must be at least 1")
    config.gamma = float(
        np.exp(
            -np.log(2.0)
            * (config.action_repeat / 120.0)
            / config.discount_half_life_seconds
        )
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    source = torch.load(config.checkpoint, map_location=device)
    source_version = int(source.get("model_version", 0))
    if source_version not in (8, 9):
        raise ValueError("QDagger requires a compatible v8/v9 DAgger checkpoint")
    spec = VisualSetSpec(**source["visual_set_spec"])
    source_hparams = dict(source["model_hparams"])
    if source_version == 8:
        source_hparams["safety_horizons"] = (0.30,)
    config.safety_horizons = tuple(source_hparams.get("safety_horizons", (0.30,)))
    config.safety_thresholds = tuple(
        source.get("safety_thresholds", (0.50,) * len(config.safety_horizons))
    )
    model = VisualSetRecurrentQNetwork(
        spec, len(BarrageVisionEnv.ACTIONS), **source_hparams
    ).to(device)
    if source_version == 8:
        migrated = dict(source["model"])
        # A one-horizon v9 head has the same tensor shape as the v8 collision
        # head, so the remaining weights migrate without approximation.
        model.load_state_dict(migrated)
    else:
        model.load_state_dict(source["model"])
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    extractor = SemanticFrameExtractor(spec)
    del source
    envs = [
        BarrageVisionEnv(
            bullet_count=config.bullet_count,
            bullet_size_min=config.bullet_size_min,
            bullet_size_max=config.bullet_size_max,
            bullet_speed_min=config.bullet_speed_min,
            bullet_speed_max=config.bullet_speed_max,
            targeted_bullet_probability=config.targeted_bullet_probability,
            max_episode_seconds=config.max_episode_seconds,
            wall_collision=False,
            action_repeat=config.action_repeat,
            scenario_mix=config.scenario_mix,
            core_bullet_size=config.core_bullet_size,
            core_bullet_speed=config.core_bullet_speed,
            stress_targeted_bullet_probability=(
                config.stress_targeted_bullet_probability
            ),
        )
        for _ in range(config.num_envs)
    ]
    batch_env = BatchedBarrageEnv(envs)
    planner_pool = (
        ProcessPlannerPool(
            envs[0], config.cpu_workers, config.teacher_horizon_seconds,
            config.teacher_reaction_seconds, config.safety_horizons,
        )
        if config.cpu_workers > 1 else None
    )
    cpu_executor = (
        ThreadPoolExecutor(max_workers=config.cpu_workers, thread_name_prefix="qd-cpu")
        if config.cpu_workers > 1 else None
    )
    output = Path(config.output_dir)
    prepare_new_output(output)
    atomic_write_json(output / "config.json", asdict(config))
    atomic_write_json(
        output / "run_manifest.json",
        {
            "source_checkpoint": str(Path(config.checkpoint).resolve()),
            "source_checkpoint_sha256": sha256_file(Path(config.checkpoint)),
            "git_revision": git_revision(PROJECT_ROOT),
            "training_seed": config.seed,
            "validation_seed": config.evaluation_seed,
            "evaluation_episode_limit_seconds": config.evaluation_episode_limit_seconds,
            "model_version": model.model_version,
        },
    )

    teacher_replay = EpisodeTransitionReplay(
        max(config.teacher_replay_steps, config.sequence_length), config.seed + 5
    )
    _collect_teacher_replay(
        batch_env, extractor, cpu_executor, planner_pool, teacher_replay,
        config.teacher_replay_steps, config.seed + 1_000_000,
        config.safety_horizons, config.teacher_horizon_seconds,
        config.teacher_reaction_seconds,
    )
    _offline_pretrain_q(
        model, target, optimizer, teacher_replay, config, device
    )
    config.safety_thresholds, config.use_safety_filter = _calibrate_safety_on_replay(
        model, teacher_replay, config, device
    )
    _save_checkpoint(
        output / "offline_pretrained.pt", model, optimizer, config, spec, 0
    )

    # Only expose the Q head to the environment after it has been calibrated
    # on teacher trajectories.  The actor copy is therefore created after the
    # offline phase rather than from the inherited regret head.
    actor_model = copy.deepcopy(model).eval() if config.async_updates else model
    agent = ScreenOnlyAgent(
        actor_model, extractor, device, inference_head="q",
        safety_thresholds=config.safety_thresholds,
        use_safety_filter=config.use_safety_filter,
    )
    observations = batch_env.reset(config.seed)
    agent.reset(config.num_envs)
    active_episodes: List[List[Dict[str, np.ndarray]]] = [
        [] for _ in range(config.num_envs)
    ]
    active_starts_at_episode_start = np.ones(config.num_envs, dtype=np.bool_)
    replay = EpisodeTransitionReplay(config.replay_capacity, config.seed + 9)
    rng = np.random.default_rng(config.seed + 17)
    learner_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="qd-learner")
        if config.async_updates else None
    )
    pending_update: Optional[Future] = None
    objects, masks, globals_ = _extract_features(extractor, observations, None)
    global_step = 0
    update_count = 0
    next_target_step = max(config.target_update_interval, config.num_envs)
    next_checkpoint_step = max(config.checkpoint_interval, config.num_envs)
    next_evaluation_step = max(config.evaluation_interval, config.num_envs)
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    final_path = output / "final.pt"
    evaluation_rows: List[Dict[str, float]] = []
    best_iqm = float("-inf")
    best_selection = (float("-inf"),) * 5
    best_step = 0
    last_evaluation_step = -1
    last_logged_update = 0

    def finish_pending_update() -> Optional[Dict[str, float]]:
        nonlocal pending_update, update_count, last_logged_update
        if pending_update is None:
            return None
        future = pending_update
        pending_update = None
        result = future.result()
        update_count += config.gradient_steps
        if update_count // 100 > last_logged_update // 100:
            print(
                f"step={global_step} replay={len(replay)} epsilon={epsilon:.3f} "
                f"loss={result['loss']:.4f} td={result['td_loss']:.4f}",
                flush=True,
            )
            last_logged_update = update_count
        return result

    def evaluate_and_select(checkpoint_path: Path) -> None:
        nonlocal best_iqm, best_selection, best_step, last_evaluation_step
        calibrated, filter_usable = _calibrate_safety_on_replay(
            model, teacher_replay, config, device
        )
        config.safety_thresholds = calibrated
        config.use_safety_filter = filter_usable
        agent.safety_thresholds = torch.as_tensor(
            calibrated, device=device, dtype=torch.float32
        )
        agent.use_safety_filter = filter_usable
        _save_checkpoint(
            checkpoint_path, model, optimizer, config, spec, global_step
        )
        atomic_write_json(output / "config.json", asdict(config))
        archived = output / "checkpoints" / f"step_{global_step:09d}.pt"
        archived.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(checkpoint_path, archived)
        result = _evaluate_checkpoint(
            archived, config, global_step, output
        )
        evaluation_rows.append(result)
        _write_evaluation_history(output / "evaluation_history.csv", evaluation_rows)
        last_evaluation_step = global_step
        iqm = float(result["model_iqm"])
        selection = (
            float(result["success_at_limit"]),
            float(result["model_cvar5"]),
            float(result["model_p5"]),
            float(result["model_p10"]),
            iqm,
        )
        improved = selection > best_selection
        if improved:
            best_selection = selection
            best_iqm = iqm
            best_step = global_step
            atomic_copy(archived, best_path)
            atomic_write_json(output / "best_summary.json", result)
        evaluation_dir = output / "evaluations" / f"step_{global_step:09d}"
        atomic_write_json(
            evaluation_dir / "manifest.json",
            {
                "global_step": global_step,
                "checkpoint": str(archived.resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "selection_key": list(selection),
                "is_best": improved,
                "selection_seed": config.evaluation_seed,
                "episode_limit_seconds": result["episode_limit_seconds"],
            },
        )
        save_qdagger_results_plot(
            output / "evaluation_history.csv", output / "config.json",
            output / "results.png",
        )
        print(
            f"evaluation step={global_step} success={result['success_at_limit']:.3f} "
            f"iqm={iqm:.3f} "
            f"p10={result['model_p10']:.3f} "
            f"best_step={best_step}{' new_best' if improved else ''}",
            flush=True,
        )

    try:
        while global_step < config.total_steps:
            progress = global_step / max(config.total_steps, 1)
            epsilon = _linear_schedule(
                config.epsilon_start, config.epsilon_end, progress
            )
            distillation = _linear_schedule(
                config.distillation_start, config.distillation_end, progress
            )
            actions = agent.act_features(
                objects, masks, globals_, epsilon=epsilon, rng=rng
            )
            teacher_count = int(round(
                config.num_envs * config.online_teacher_fraction
            ))
            if config.online_teacher_fraction > 0.0:
                teacher_count = max(1, teacher_count)
            teacher_count = min(config.num_envs, teacher_count)
            teacher_indices = np.sort(
                rng.choice(config.num_envs, size=teacher_count, replace=False)
            ) if teacher_count else np.empty(0, dtype=np.int64)
            selected_supervision = _planner_supervision_batch(
                [envs[int(index)] for index in teacher_indices],
                cpu_executor,
                planner_pool,
                config.safety_horizons,
                config.teacher_horizon_seconds,
                config.teacher_reaction_seconds,
            )
            supervision = {
                int(index): item
                for index, item in zip(teacher_indices, selected_supervision)
            }
            (
                next_observations, rewards, terminals, truncations,
                final_observations, _,
            ) = batch_env.step_detailed(actions)
            next_objects, next_masks, next_globals = _extract_features(
                extractor, next_observations, None
            )
            final_objects, final_masks, final_globals = _extract_features(
                extractor, final_observations, None
            )
            done_mask = terminals | truncations
            for index in range(config.num_envs):
                teacher = supervision.get(index)
                teacher_valid = teacher is not None
                active_episodes[index].append(
                    {
                        "objects": objects[index].astype(np.float16),
                        "masks": masks[index],
                        "globals": globals_[index].astype(np.float16),
                        "next_objects": (
                            final_objects[index] if done_mask[index]
                            else next_objects[index]
                        ).astype(np.float16),
                        "next_masks": (
                            final_masks[index] if done_mask[index]
                            else next_masks[index]
                        ),
                        "next_globals": (
                            final_globals[index] if done_mask[index]
                            else next_globals[index]
                        ).astype(np.float16),
                        "actions": np.asarray(actions[index], np.int64),
                        "rewards": np.asarray(rewards[index], np.float32),
                        "terminals": np.asarray(terminals[index], np.float32),
                        "truncations": np.asarray(truncations[index], np.float32),
                        "regrets": (
                            teacher.regrets.astype(np.float16)
                            if teacher_valid else
                            np.zeros(len(BarrageVisionEnv.ACTIONS), np.float16)
                        ),
                        "collisions": (
                            teacher.safety_targets.astype(np.float16)
                            if teacher_valid else
                            np.zeros(
                                (len(config.safety_horizons),
                                 len(BarrageVisionEnv.ACTIONS)),
                                np.float16,
                            )
                        ),
                        "teacher_valid": np.asarray(teacher_valid, np.bool_),
                        "urgent": np.asarray(
                            teacher.urgent if teacher_valid else False, np.bool_
                        ),
                        "episode_start": np.asarray(
                            active_starts_at_episode_start[index]
                            and not active_episodes[index],
                            np.bool_,
                        ),
                    }
                )
                if done_mask[index]:
                    replay.add_episode(active_episodes[index])
                    active_episodes[index] = []
                    active_starts_at_episode_start[index] = True
                elif len(active_episodes[index]) >= config.replay_chunk_steps:
                    replay.add_episode(active_episodes[index])
                    active_episodes[index] = []
                    active_starts_at_episode_start[index] = False
            agent.reset_indices(done_mask)
            observations = next_observations
            objects, masks, globals_ = next_objects, next_masks, next_globals
            global_step += config.num_envs
            finish_pending_update()
            if len(replay) >= config.learning_starts:
                teacher_count = int(round(config.batch_size * config.teacher_batch_fraction))
                online_count = config.batch_size - teacher_count
                tail_steps = max(
                    1, int(round(config.failure_tail_seconds / (config.action_repeat / 120.0)))
                )
                sampled_batches = [
                    (
                        replay.sample(
                            online_count,
                            config.sequence_length,
                            failure_fraction=config.failure_tail_fraction,
                            urgent_fraction=config.urgent_fraction,
                            failure_tail_steps=tail_steps,
                        )
                        + teacher_replay.sample(
                            teacher_count, config.sequence_length
                        )
                    )
                    for _ in range(config.gradient_steps)
                ]
                if learner_executor is None:
                    pending_update = Future()
                    try:
                        pending_update.set_result(
                            _update_many(
                                model, target, optimizer, sampled_batches, config,
                                device, distillation,
                            )
                        )
                    except BaseException as error:
                        pending_update.set_exception(error)
                else:
                    pending_update = learner_executor.submit(
                        _update_many, model, target, optimizer, sampled_batches,
                        config, device, distillation,
                    )
            maintenance_due = (
                global_step >= next_target_step
                or global_step >= next_checkpoint_step
                or global_step >= next_evaluation_step
            )
            if maintenance_due:
                finish_pending_update()
            if global_step >= next_target_step:
                target.load_state_dict(model.state_dict())
                target.memory.flatten_parameters()
                if actor_model is not model:
                    actor_model.load_state_dict(model.state_dict())
                    actor_model.memory.flatten_parameters()
                while next_target_step <= global_step:
                    next_target_step += max(
                        config.target_update_interval, config.num_envs
                    )
            if global_step >= next_checkpoint_step:
                _save_checkpoint(
                    latest_path, model, optimizer, config, spec, global_step
                )
                while next_checkpoint_step <= global_step:
                    next_checkpoint_step += max(
                        config.checkpoint_interval, config.num_envs
                    )
            if global_step >= next_evaluation_step:
                _save_checkpoint(
                    latest_path, model, optimizer, config, spec, global_step
                )
                evaluate_and_select(latest_path)
                while next_evaluation_step <= global_step:
                    next_evaluation_step += max(
                        config.evaluation_interval, config.num_envs
                    )
        finish_pending_update()
    finally:
        if pending_update is not None:
            finish_pending_update()
        if learner_executor is not None:
            learner_executor.shutdown(wait=True)
        if cpu_executor is not None:
            cpu_executor.shutdown(wait=True)
        if planner_pool is not None:
            planner_pool.shutdown()
        batch_env.close()
    _save_checkpoint(final_path, model, optimizer, config, spec, global_step)
    _save_checkpoint(latest_path, model, optimizer, config, spec, global_step)
    if last_evaluation_step != global_step:
        evaluate_and_select(final_path)
    results_plot = save_qdagger_results_plot(
        output / "evaluation_history.csv", output / "config.json", output / "results.png"
    )
    print(
        f"final_checkpoint={final_path.resolve()} best_step={best_step} "
        f"best_iqm={best_iqm:.3f} results_plot={results_plot.resolve()}",
        flush=True,
    )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Online recurrent QDagger fine-tuning")
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", default="runs/visual_set_v9_qdagger")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-step", type=int, default=15)
    parser.add_argument("--teacher-replay-steps", type=int, default=50_000)
    parser.add_argument("--online-teacher-fraction", type=float, default=0.25)
    parser.add_argument("--offline-pretraining-updates", type=int, default=2_000)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--evaluation-interval", type=int, default=200_000)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument(
        "--async-updates", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    parser.add_argument("--max-episode-seconds", type=float, default=180.0)
    parser.add_argument("--evaluation-episode-limit-seconds", type=float, default=120.0)
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = QDaggerConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        n_step=args.n_step,
        teacher_replay_steps=args.teacher_replay_steps,
        online_teacher_fraction=args.online_teacher_fraction,
        offline_pretraining_updates=args.offline_pretraining_updates,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.evaluation_interval,
        cpu_workers=args.cpu_workers,
        async_updates=args.async_updates,
        evaluation_episodes=args.evaluation_episodes,
        max_episode_seconds=args.max_episode_seconds,
        evaluation_episode_limit_seconds=args.evaluation_episode_limit_seconds,
        evaluation_workers=args.evaluation_workers,
        device=args.device,
    )
    if args.smoke_test:
        config.total_steps = 512
        config.learning_starts = 64
        config.replay_capacity = 512
        config.batch_size = 4
        config.sequence_length = 8
        config.burn_in = 2
        config.n_step = 3
        config.num_envs = 4
        config.teacher_replay_steps = 128
        config.offline_pretraining_updates = 4
        config.max_episode_seconds = 1.0
        config.evaluation_episodes = 2
        config.target_update_interval = 64
        config.checkpoint_interval = 128
        config.evaluation_interval = 256
        config.cpu_workers = 4
        config.safety_calibration_samples = 32
    checkpoint = train_qdagger(config)
    print(f"best_checkpoint={checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
