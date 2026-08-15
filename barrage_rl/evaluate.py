"""加载训练好的CNN策略并评估存活时间。"""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .env import BarrageVisionEnv
from .model import ActorCritic


def load_agent(checkpoint_path: str, device: torch.device) -> tuple:
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ActorCritic(
        int(config["frame_stack"]),
        int(config["observation_size"]),
        len(BarrageVisionEnv.ACTIONS),
        model_version=int(config.get("model_version", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def evaluate(
    checkpoint_path: str,
    episodes: int,
    render: bool,
    bullet_count: int,
    seed: int,
    device_name: str,
    compare_noop: bool = False,
    deterministic: bool = True,
) -> np.ndarray:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA，请使用 --device cpu")
    device = torch.device(device_name)
    model, config = load_agent(checkpoint_path, device)
    env = BarrageVisionEnv(
        bullet_count=bullet_count,
        observation_size=int(config["observation_size"]),
        frame_stack=int(config["frame_stack"]),
        action_repeat=int(config["action_repeat"]),
        render_mode="human" if render else None,
        randomize_initial_phase=bool(config.get("randomize_initial_phase", False)),
        initial_phase_min_seconds=float(config.get("initial_phase_min_seconds", 0.75)),
        initial_phase_max_seconds=float(config.get("initial_phase_max_seconds", 3.0)),
    )

    survival_times = []
    action_histogram = np.zeros(len(BarrageVisionEnv.ACTIONS), dtype=np.int64)
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            info = {"survival_seconds": 0.0, "score": 0}
            while not (terminated or truncated):
                with torch.inference_mode():
                    observation_tensor = torch.as_tensor(
                        observation[None], device=device
                    )
                    action, _, _, _ = model.get_action_and_value(
                        observation_tensor, deterministic=deterministic
                    )
                observation, _, terminated, truncated, info = env.step(
                    int(action.item())
                )
                action_histogram[int(action.item())] += 1
            survival = float(info["survival_seconds"])
            survival_times.append(survival)
            print(
                "episode=%d score=%d survival=%.2fs"
                % (episode + 1, int(info["score"]), survival)
            )
    finally:
        env.close()

    result = np.asarray(survival_times, dtype=np.float32)
    print(
        "mean=%.2fs median=%.2fs min=%.2fs max=%.2fs"
        % (result.mean(), np.median(result), result.min(), result.max())
    )
    print("action_histogram=%s" % action_histogram.tolist())

    if compare_noop:
        noop_env = BarrageVisionEnv(
            bullet_count=bullet_count,
            observation_size=int(config["observation_size"]),
            frame_stack=int(config["frame_stack"]),
            action_repeat=int(config["action_repeat"]),
            randomize_initial_phase=bool(config.get("randomize_initial_phase", False)),
            initial_phase_min_seconds=float(
                config.get("initial_phase_min_seconds", 0.75)
            ),
            initial_phase_max_seconds=float(
                config.get("initial_phase_max_seconds", 3.0)
            ),
        )
        noop_times = []
        try:
            for episode in range(episodes):
                noop_env.reset(seed=seed + episode)
                terminated = False
                truncated = False
                noop_info = {"survival_seconds": 0.0}
                while not (terminated or truncated):
                    _, _, terminated, truncated, noop_info = noop_env.step(0)
                noop_times.append(float(noop_info["survival_seconds"]))
        finally:
            noop_env.close()
        noop_result = np.asarray(noop_times, dtype=np.float32)
        ratio = float(result.mean() / noop_result.mean())
        print(
            "paired_noop_mean=%.2fs model/noop=%.3fx target_140=%.2fs"
            % (noop_result.mean(), ratio, 1.4 * noop_result.mean())
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="评估Barrage纯视觉CNN智能体")
    parser.add_argument("checkpoint", help="训练产生的latest.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--bullets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--render", action="store_true", help="实时显示智能体游戏画面")
    parser.add_argument(
        "--compare-noop",
        action="store_true",
        help="使用完全相同的种子配对评估原地不动基线",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="按策略概率采样动作；默认使用argmax确定性动作",
    )
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    evaluate(
        checkpoint,
        args.episodes,
        args.render,
        args.bullets,
        args.seed,
        args.device,
        args.compare_noop,
        not args.stochastic,
    )


if __name__ == "__main__":
    main()
