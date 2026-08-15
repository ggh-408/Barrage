"""针对 RTX 4060 优化的精简 PPO 训练入口。"""

import argparse
import csv
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .env import BarrageVisionEnv, BatchedBarrageEnv
from .model import ActorCritic
from .plot import save_results_plot


@dataclass
class TrainConfig:
    # 默认批量大小约为4096帧，适合8GB显存的RTX 4060
    total_steps: int = 2_000_000
    num_envs: int = 32
    rollout_steps: int = 128
    minibatch_size: int = 512
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    learning_rate_min_ratio: float = 0.1
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_min: float = 0.001
    entropy_coef_max: float = 0.03
    entropy_target: float = 1.4
    entropy_adaptation_rate: float = 0.05
    target_kl: float = 0.015
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_bullets: int = 50
    observation_size: int = 96
    frame_stack: int = 4
    action_repeat: int = 4
    checkpoint_interval: int = 25
    seed: int = 1
    output_dir: str = "runs/barrage_ppo"
    device: str = "cuda"
    model_version: int = 3
    randomize_initial_phase: bool = True
    initial_phase_min_seconds: float = 0.75
    initial_phase_max_seconds: float = 3.0
    dense_reward_scale: float = 0.25
    danger_horizon_seconds: float = 2.0
    curriculum_start_bullets: int = 20
    curriculum_step_bullets: int = 5
    curriculum_baseline_episodes: int = 64
    curriculum_window_episodes: int = 200
    curriculum_improvement_ratio: float = 1.4
    training_version: int = 4


def parse_args() -> Tuple[TrainConfig, Optional[str], bool]:
    parser = argparse.ArgumentParser(description="训练只读取像素的Barrage CNN智能体")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--target-bullets", type=int, default=50)
    parser.add_argument("--observation-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/barrage_ppo")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", help="从checkpoint继续训练")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只运行极短训练，用于检查环境、模型和CUDA",
    )
    args = parser.parse_args()

    config = TrainConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        target_bullets=args.target_bullets,
        observation_size=args.observation_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
    if args.smoke_test:
        config.total_steps = 128
        config.num_envs = 2
        config.rollout_steps = 32
        config.minibatch_size = 64
        config.update_epochs = 2
        config.checkpoint_interval = 1
        config.target_bullets = min(config.target_bullets, 10)
        config.curriculum_start_bullets = config.target_bullets
        config.curriculum_baseline_episodes = 4
        config.curriculum_window_episodes = 4
    return config, args.resume, args.smoke_test


def configure_runtime(config: TrainConfig) -> torch.device:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA，请检查PyTorch安装或使用 --device cpu")
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        # 固定输入尺寸下让cuDNN选择最快卷积实现；TF32兼顾速度和PPO稳定性
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def create_batch_env(config: TrainConfig, initial_bullets: int) -> BatchedBarrageEnv:
    envs = [
        BarrageVisionEnv(
            bullet_count=initial_bullets,
            observation_size=config.observation_size,
            frame_stack=config.frame_stack,
            action_repeat=config.action_repeat,
            wall_collision=False,
            render_mode=None,
            randomize_initial_phase=config.randomize_initial_phase,
            initial_phase_min_seconds=config.initial_phase_min_seconds,
            initial_phase_max_seconds=config.initial_phase_max_seconds,
            dense_reward_scale=config.dense_reward_scale,
            danger_horizon_seconds=config.danger_horizon_seconds,
        )
        for _ in range(config.num_envs)
    ]
    return BatchedBarrageEnv(envs)


def estimate_noop_baseline(config: TrainConfig, bullet_count: int) -> float:
    """使用固定种子估算当前难度下原地不动的平均存活时间。"""
    env = BarrageVisionEnv(
        bullet_count=bullet_count,
        observation_size=config.observation_size,
        frame_stack=config.frame_stack,
        action_repeat=config.action_repeat,
        wall_collision=False,
        randomize_initial_phase=config.randomize_initial_phase,
        initial_phase_min_seconds=config.initial_phase_min_seconds,
        initial_phase_max_seconds=config.initial_phase_max_seconds,
        dense_reward_scale=0.0,
        danger_horizon_seconds=config.danger_horizon_seconds,
    )
    survival_times: List[float] = []
    try:
        for episode in range(config.curriculum_baseline_episodes):
            _, _ = env.reset(seed=100_000 + bullet_count * 1_000 + episode)
            terminated = False
            truncated = False
            info = {"survival_seconds": 0.0}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(0)
            survival_times.append(float(info["survival_seconds"]))
    finally:
        env.close()
    return float(np.mean(survival_times))


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    global_step: int,
    update: int,
    training_state: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "global_step": global_step,
            "update": update,
            "training_state": training_state,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_model_version = int(checkpoint_config.get("model_version", 1))
    if checkpoint_model_version != model.model_version:
        raise ValueError(
            "checkpoint使用模型v%d，当前训练使用模型v%d；新网络需要从头训练"
            % (checkpoint_model_version, model.model_version)
        )
    checkpoint_training_version = int(checkpoint_config.get("training_version", 1))
    if checkpoint_training_version != 4:
        raise ValueError(
            "checkpoint使用训练方案v%d，当前为v4；奖励、视觉和课程已改变，需要从头训练"
            % checkpoint_training_version
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        int(checkpoint.get("global_step", 0)),
        int(checkpoint.get("update", 0)),
        dict(checkpoint.get("training_state", {})),
    )


def train(config: TrainConfig, resume: Optional[str] = None) -> Path:
    device = configure_runtime(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.csv"
    checkpoint_path = output_dir / "latest.pt"
    best_checkpoint_path = output_dir / "best.pt"

    initial_bullets = min(config.curriculum_start_bullets, config.target_bullets)
    environments = create_batch_env(config, initial_bullets)
    model = ActorCritic(
        config.frame_stack,
        config.observation_size,
        len(BarrageVisionEnv.ACTIONS),
        model_version=config.model_version,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    global_step = 0
    start_update = 0
    current_bullets = initial_bullets
    entropy_coefficient = config.entropy_coef
    best_mastery_ratio = float("-inf")
    resume_learning_rate_cap: Optional[float] = None
    if resume:
        global_step, start_update, training_state = load_checkpoint(
            resume, model, optimizer, device
        )
        current_bullets = int(training_state.get("current_bullets", initial_bullets))
        entropy_coefficient = float(
            training_state.get("entropy_coefficient", config.entropy_coef)
        )
        best_mastery_ratio = float(
            training_state.get("best_mastery_ratio", float("-inf"))
        )
        # Extending --total-steps must not make a decayed learning rate jump up.
        resume_learning_rate_cap = float(optimizer.param_groups[0]["lr"])
        environments.set_bullet_count(current_bullets)

    batch_size = config.num_envs * config.rollout_steps
    if batch_size % config.minibatch_size != 0:
        raise ValueError("num_envs * rollout_steps 必须能被 minibatch_size 整除")
    total_updates = max(1, int(np.ceil(config.total_steps / batch_size)))

    # 像素轨迹使用uint8放在内存中，4096帧约占151MB，而非float32的604MB
    observations = torch.empty(
        (
            config.rollout_steps,
            config.num_envs,
            config.frame_stack,
            config.observation_size,
            config.observation_size,
        ),
        dtype=torch.uint8,
    )
    actions = torch.empty((config.rollout_steps, config.num_envs), dtype=torch.long)
    log_probabilities = torch.empty((config.rollout_steps, config.num_envs))
    rewards = torch.empty((config.rollout_steps, config.num_envs))
    dones = torch.empty((config.rollout_steps, config.num_envs))
    values = torch.empty((config.rollout_steps, config.num_envs))

    next_observation = environments.reset(config.seed)
    next_done = torch.zeros(config.num_envs)
    curriculum_window = config.curriculum_window_episodes
    recent_returns: Deque[float] = deque(maxlen=curriculum_window)
    recent_survival: Deque[float] = deque(maxlen=curriculum_window)
    recent_scores: Deque[float] = deque(maxlen=curriculum_window)
    session_start_step = global_step
    start_time = time.perf_counter()
    noop_baseline = estimate_noop_baseline(config, current_bullets)
    print(
        "curriculum bullets=%d noop_baseline=%.3fs promote_at=%.3fs"
        % (
            current_bullets,
            noop_baseline,
            noop_baseline * config.curriculum_improvement_ratio,
        )
    )

    # 新训练覆盖旧日志，只有明确resume时才追加，避免一个CSV混入多次实验
    log_mode = "a" if resume else "w"
    write_header = log_mode == "w" or not log_path.exists()
    log_file = log_path.open(log_mode, newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            [
                "global_step",
                "update",
                "steps_per_second",
                "mean_return_100",
                "mean_survival_seconds_100",
                "mean_score_100",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "bullet_count",
                "noop_baseline_seconds",
                "mastery_ratio",
                "entropy_coefficient",
                "ppo_minibatches",
            ]
        )

    print("device=%s, envs=%d, rollout=%d, batch=%d" % (
        device,
        config.num_envs,
        config.rollout_steps,
        batch_size,
    ))
    if device.type == "cuda":
        print("gpu=%s" % torch.cuda.get_device_name(device))

    last_update = start_update
    try:
        for update in range(start_update + 1, total_updates + 1):
            last_update = update
            environments.set_bullet_count(current_bullets)

            # 线性降低学习率，不增加额外调参项
            linear_progress = 1.0 - (update - 1) / total_updates
            progress_remaining = max(
                config.learning_rate_min_ratio, linear_progress
            )
            scheduled_learning_rate = config.learning_rate * progress_remaining
            if resume_learning_rate_cap is not None:
                scheduled_learning_rate = min(
                    scheduled_learning_rate, resume_learning_rate_cap
                )
            optimizer.param_groups[0]["lr"] = scheduled_learning_rate

            model.eval()
            for step in range(config.rollout_steps):
                global_step += config.num_envs
                observations[step].copy_(torch.from_numpy(next_observation))
                dones[step].copy_(next_done)

                with torch.inference_mode():
                    observation_gpu = torch.as_tensor(next_observation, device=device)
                    action, log_probability, _, value = model.get_action_and_value(
                        observation_gpu
                    )

                actions[step].copy_(action.cpu())
                log_probabilities[step].copy_(log_probability.cpu())
                values[step].copy_(value.cpu())
                next_observation, reward, done, completed = environments.step(
                    action.cpu().numpy()
                )
                rewards[step].copy_(torch.from_numpy(reward))
                next_done = torch.from_numpy(done)

                for episode in completed:
                    recent_returns.append(episode["return"])
                    recent_survival.append(episode["survival_seconds"])
                    recent_scores.append(episode["score"])

            with torch.inference_mode():
                next_value = model.get_value(
                    torch.as_tensor(next_observation, device=device)
                ).cpu()

            advantages = torch.zeros_like(rewards)
            last_advantage = torch.zeros(config.num_envs)
            for step in reversed(range(config.rollout_steps)):
                if step == config.rollout_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]
                delta = (
                    rewards[step]
                    + config.gamma * next_values * next_nonterminal
                    - values[step]
                )
                last_advantage = (
                    delta
                    + config.gamma
                    * config.gae_lambda
                    * next_nonterminal
                    * last_advantage
                )
                advantages[step] = last_advantage
            returns = advantages + values

            flat_observations = observations.reshape(
                (-1, config.frame_stack, config.observation_size, config.observation_size)
            )
            flat_actions = actions.reshape(-1)
            flat_log_probabilities = log_probabilities.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values.reshape(-1)

            policy_losses: List[float] = []
            value_losses: List[float] = []
            entropies: List[float] = []
            approximate_kls: List[float] = []
            clip_fractions: List[float] = []
            model.train()

            stop_ppo_early = False
            ppo_minibatches = 0
            for _ in range(config.update_epochs):
                permutation = torch.randperm(batch_size)
                epoch_kls: List[float] = []
                for start in range(0, batch_size, config.minibatch_size):
                    indices = permutation[start : start + config.minibatch_size]
                    batch_observations = flat_observations[indices].to(
                        device, non_blocking=True
                    )
                    batch_actions = flat_actions[indices].to(device)
                    old_log_probability = flat_log_probabilities[indices].to(device)
                    batch_advantage = flat_advantages[indices].to(device)
                    batch_return = flat_returns[indices].to(device)
                    old_value = flat_values[indices].to(device)

                    _, new_log_probability, entropy, new_value = model.get_action_and_value(
                        batch_observations, batch_actions
                    )
                    log_ratio = new_log_probability - old_log_probability
                    ratio = log_ratio.exp()
                    normalized_advantage = (
                        batch_advantage - batch_advantage.mean()
                    ) / (batch_advantage.std() + 1e-8)

                    unclipped_loss = -normalized_advantage * ratio
                    clipped_loss = -normalized_advantage * torch.clamp(
                        ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                    )
                    policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()

                    clipped_value = old_value + torch.clamp(
                        new_value - old_value, -config.clip_coef, config.clip_coef
                    )
                    value_loss = 0.5 * torch.maximum(
                        (new_value - batch_return).pow(2),
                        (clipped_value - batch_return).pow(2),
                    ).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - entropy_coefficient * entropy_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        approximate_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fraction = (
                            (ratio - 1.0).abs() > config.clip_coef
                        ).float().mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_loss.item()))
                    approximate_kls.append(float(approximate_kl.item()))
                    epoch_kls.append(float(approximate_kl.item()))
                    clip_fractions.append(float(clip_fraction.item()))
                    ppo_minibatches += 1

                    if (
                        len(epoch_kls) >= 2
                        and float(np.mean(epoch_kls)) > config.target_kl
                    ):
                        stop_ppo_early = True
                        break
                if stop_ppo_early:
                    break

            mean_entropy_value = float(np.mean(entropies))
            entropy_coefficient *= float(
                np.exp(
                    config.entropy_adaptation_rate
                    * (config.entropy_target - mean_entropy_value)
                )
            )
            entropy_coefficient = float(
                np.clip(
                    entropy_coefficient,
                    config.entropy_coef_min,
                    config.entropy_coef_max,
                )
            )

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            steps_per_second = int((global_step - session_start_step) / elapsed)
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            mean_survival = (
                float(np.mean(recent_survival)) if recent_survival else float("nan")
            )
            mean_score = float(np.mean(recent_scores)) if recent_scores else float("nan")
            mastery_ratio = (
                mean_survival / noop_baseline
                if np.isfinite(mean_survival) and noop_baseline > 0
                else float("nan")
            )
            metrics = [
                global_step,
                update,
                steps_per_second,
                mean_return,
                mean_survival,
                mean_score,
                float(np.mean(policy_losses)),
                float(np.mean(value_losses)),
                mean_entropy_value,
                float(np.mean(approximate_kls)),
                float(np.mean(clip_fractions)),
                current_bullets,
                noop_baseline,
                mastery_ratio,
                entropy_coefficient,
                ppo_minibatches,
            ]
            log_writer.writerow(metrics)
            log_file.flush()

            if (
                len(recent_survival) >= curriculum_window
                and np.isfinite(mastery_ratio)
                and mastery_ratio > best_mastery_ratio
            ):
                best_mastery_ratio = mastery_ratio
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    global_step,
                    update,
                    {
                        "current_bullets": current_bullets,
                        "entropy_coefficient": entropy_coefficient,
                        "best_mastery_ratio": best_mastery_ratio,
                    },
                )
                print(
                    "best checkpoint update=%d bullets=%d mastery=%.3fx path=%s"
                    % (
                        update,
                        current_bullets,
                        best_mastery_ratio,
                        best_checkpoint_path.resolve(),
                    )
                )
            print(
                "update=%d/%d step=%d sps=%d bullets=%d score=%.1f survival=%.2f entropy=%.3f kl=%.4f"
                % (
                    update,
                    total_updates,
                    global_step,
                    steps_per_second,
                    current_bullets,
                    mean_score,
                    mean_survival,
                    metrics[8],
                    metrics[9],
                )
            )

            if (
                current_bullets < config.target_bullets
                and len(recent_survival) >= curriculum_window
                and mastery_ratio >= config.curriculum_improvement_ratio
            ):
                previous_bullets = current_bullets
                current_bullets = min(
                    current_bullets + config.curriculum_step_bullets,
                    config.target_bullets,
                )
                noop_baseline = estimate_noop_baseline(config, current_bullets)
                environments.set_bullet_count(current_bullets)
                next_observation = environments.reset(config.seed + global_step)
                next_done.zero_()
                recent_returns.clear()
                recent_survival.clear()
                recent_scores.clear()
                best_mastery_ratio = float("-inf")
                print(
                    "curriculum promoted %d->%d bullets, noop_baseline=%.3fs promote_at=%.3fs"
                    % (
                        previous_bullets,
                        current_bullets,
                        noop_baseline,
                        noop_baseline * config.curriculum_improvement_ratio,
                    )
                )

            if update % config.checkpoint_interval == 0 or update == total_updates:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config,
                    global_step,
                    update,
                    {
                        "current_bullets": current_bullets,
                        "entropy_coefficient": entropy_coefficient,
                        "best_mastery_ratio": best_mastery_ratio,
                    },
                )
    except KeyboardInterrupt:
        # 用户中断时保存最近一次有效网络，避免长时间训练成果丢失
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config,
            global_step,
            last_update,
            {
                "current_bullets": current_bullets,
                "entropy_coefficient": entropy_coefficient,
                "best_mastery_ratio": best_mastery_ratio,
            },
        )
        print("训练已中断，checkpoint已保存到 %s" % checkpoint_path.resolve())
    finally:
        log_file.close()
        environments.close()

    results_plot = save_results_plot(log_path, output_dir / "results.png")
    print("results_plot=%s" % results_plot.resolve())

    return checkpoint_path


def main() -> None:
    config, resume, _ = parse_args()
    checkpoint = train(config, resume)
    print("checkpoint=%s" % checkpoint.resolve())


if __name__ == "__main__":
    main()
