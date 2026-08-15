"""Evaluate a visual-set checkpoint with an image-only policy boundary."""

import argparse
import csv
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .artifacts import atomic_write_text, sha256_file
from .env import BarrageVisionEnv
from .parallel_evaluation import run_parallel_rollout
from .visual_set import (
    ScreenOnlyAgent,
    SemanticFrameExtractor,
    VisualSetRecurrentQNetwork,
    VisualSetSpec,
)


def _bootstrap_confidence_intervals(
    model_times: np.ndarray,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> Dict[str, float]:
    """Return percentile-bootstrap 95% intervals for the model mean."""
    count = len(model_times)
    if count == 0:
        raise ValueError("model evaluation must be non-empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(bootstrap_samples, count))
    model_means = model_times[indices].mean(axis=1)

    def interval(values: np.ndarray) -> Tuple[float, float]:
        low, high = np.percentile(values, (2.5, 97.5))
        return float(low), float(high)

    model_low, model_high = interval(model_means)
    return {
        "model_mean_ci95_low": model_low,
        "model_mean_ci95_high": model_high,
    }


def load_agent(path: str, device: torch.device) -> Tuple[ScreenOnlyAgent, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    spec = VisualSetSpec(**checkpoint["visual_set_spec"])
    version = int(checkpoint.get("model_version", 0))
    if version not in (8, 9):
        raise ValueError("only v8/v9 visual-set checkpoints are supported")
    model_hparams = dict(checkpoint.get("model_hparams", {}))
    if version == 8:
        model_hparams["safety_horizons"] = (0.30,)
    model = VisualSetRecurrentQNetwork(
        spec,
        len(BarrageVisionEnv.ACTIONS),
        **model_hparams,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        ScreenOnlyAgent(
            model,
            SemanticFrameExtractor(spec),
            device,
            inference_head=checkpoint.get("inference_head", "policy"),
            safety_thresholds=checkpoint.get("safety_thresholds"),
            use_safety_filter=bool(
                checkpoint.get("use_safety_filter", version >= 9)
            ),
        ),
        checkpoint,
    )


def _interquartile_mean(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) < 4:
        return float(ordered.mean())
    lower = int(np.floor(len(ordered) * 0.25))
    upper = int(np.ceil(len(ordered) * 0.75))
    return float(ordered[lower:upper].mean())


def _lower_tail_mean(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(np.ceil(len(ordered) * float(fraction))))
    return float(ordered[:count].mean())


def _wilson_interval(successes: int, count: int, z: float = 1.96) -> Tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    margin = z * np.sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count * count)
    ) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _print_evaluation_summary(result: Dict[str, float]) -> None:
    """Print core metrics only; full-precision details are still saved."""
    print(
        "evaluation "
        f"episodes={int(result['episodes'])} "
        f"mean={result['model_mean']:.2f}s "
        f"median={result['model_median']:.2f}s "
        f"IQM={result['model_iqm']:.2f}s "
        f"P10={result['model_p10']:.2f}s "
        f"pass={100.0 * result['success_at_limit']:.2f}% "
        f"fail<10s={100.0 * result['failure_before_10s']:.2f}% "
        f"wall<40px={100.0 * result['wall_episode_fraction']:.2f}% "
        f"median_wall={result['median_min_wall_distance']:.2f}px",
        flush=True,
    )


def evaluate(
    checkpoint_path: str,
    episodes: int,
    bullets: int,
    seed: int,
    device_name: str,
    output_dir: str = "",
    targeted_bullet_probability: Optional[float] = None,
    wall_threshold: float = 40.0,
    max_episode_seconds: Optional[float] = None,
    bullet_size_min: Optional[int] = None,
    bullet_size_max: Optional[int] = None,
    bullet_speed_min: Optional[float] = None,
    bullet_speed_max: Optional[float] = None,
    workers: int = 8,
    timings: Optional[Dict[str, float]] = None,
    progress: bool = False,
    progress_interval: int = 0,
    print_summary: bool = True,
) -> Dict[str, float]:
    total_started = time.perf_counter()
    timing_output = timings if timings is not None else {}
    phase_started = time.perf_counter()
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    agent, checkpoint = load_agent(checkpoint_path, device)
    timing_output["evaluation.load_checkpoint_and_agent"] = (
        time.perf_counter() - phase_started
    )
    checkpoint_config = checkpoint.get("config", {})
    if max_episode_seconds is None:
        max_episode_seconds = float(
            checkpoint_config.get("max_episode_seconds", 60.0)
        )
    if bullet_size_min is None:
        bullet_size_min = int(checkpoint_config.get("bullet_size_min", 5))
    if bullet_size_max is None:
        bullet_size_max = int(checkpoint_config.get("bullet_size_max", bullet_size_min))
    if bullet_speed_min is None:
        bullet_speed_min = float(checkpoint_config.get("bullet_speed_min", 240.0))
    if bullet_speed_max is None:
        bullet_speed_max = float(
            checkpoint_config.get("bullet_speed_max", bullet_speed_min)
        )
    if targeted_bullet_probability is None:
        targeted_bullet_probability = float(
            checkpoint_config.get("targeted_bullet_probability", 0.0)
        )
    env_kwargs = {
        "bullet_count": bullets,
        "wall_collision": False,
        "targeted_bullet_probability": targeted_bullet_probability,
        "max_episode_seconds": max_episode_seconds,
        "bullet_size_min": bullet_size_min,
        "bullet_size_max": bullet_size_max,
        "bullet_speed_min": bullet_speed_min,
        "bullet_speed_max": bullet_speed_max,
        "randomize_initial_phase": False,
    }
    mode_started = time.perf_counter()
    model_rollout = run_parallel_rollout(
        agent=agent,
        spec=agent.extractor.spec,
        episodes=episodes,
        workers=workers,
        seed=seed,
        env_kwargs=env_kwargs,
        wall_threshold=wall_threshold,
        progress_interval=progress_interval,
    )
    if progress:
        print(
            f"profile phase=model_rollout episodes={episodes}/{episodes} "
            f"elapsed={time.perf_counter() - mode_started:.1f}s",
            flush=True,
        )
    timing_output["evaluation.model_rollout"] = time.perf_counter() - mode_started
    model_array = model_rollout.survival_times.astype(np.float32)
    phase_started = time.perf_counter()
    confidence = _bootstrap_confidence_intervals(
        model_array, seed=seed + 9_000_000
    )
    timing_output["evaluation.bootstrap_ci"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    success_horizon = 120.0
    success_observable = float(max_episode_seconds) >= success_horizon
    success_count = (
        int(np.sum(model_array >= success_horizon - 1e-9))
        if success_observable else 0
    )
    success_low, success_high = _wilson_interval(success_count, episodes)
    result = {
        "episodes": float(episodes),
        "bullets": float(bullets),
        "seed": float(seed),
        "model_mean": float(model_array.mean()),
        "model_median": float(np.median(model_array)),
        "model_iqm": _interquartile_mean(model_array),
        "model_p10": float(np.percentile(model_array, 10.0)),
        "model_p1": float(np.percentile(model_array, 1.0)),
        "model_p5": float(np.percentile(model_array, 5.0)),
        "model_cvar1": _lower_tail_mean(model_array, 0.01),
        "model_cvar5": _lower_tail_mean(model_array, 0.05),
        "episode_limit_seconds": float(max_episode_seconds),
        "bullet_size_min": float(bullet_size_min),
        "bullet_size_max": float(bullet_size_max),
        "bullet_speed_min": float(bullet_speed_min),
        "bullet_speed_max": float(bullet_speed_max),
        "success_at_limit": float(np.mean(model_array >= max_episode_seconds)),
        "selection_horizon_seconds": success_horizon,
        "success_at_120": float(success_count / episodes),
        "success_at_120_count": float(success_count),
        "success_at_120_ci95_low": success_low,
        "success_at_120_ci95_high": success_high,
        "success_at_120_observable": float(success_observable),
        "failure_before_1s": float(np.mean(model_array < 1.0)),
        "failure_before_10s": float(np.mean(model_array < 10.0)),
        "targeted_bullet_probability": float(targeted_bullet_probability),
        "wall_threshold": float(wall_threshold),
        "wall_step_fraction": float(
            model_rollout.wall_steps / max(model_rollout.model_steps, 1)
        ),
        "wall_episode_fraction": float(
            np.mean(model_rollout.minimum_wall_distances < wall_threshold)
        ),
        "median_min_wall_distance": float(
            np.median(model_rollout.minimum_wall_distances)
        ),
        "safety_filtered_actions": float(agent.filtered_action_count),
        "safety_all_unsafe_decisions": float(agent.all_unsafe_count),
        "safety_decisions": float(agent.decision_count),
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        **confidence,
    }
    timing_output["evaluation.metrics_and_checkpoint_hash"] = (
        time.perf_counter() - phase_started
    )
    if print_summary:
        _print_evaluation_summary(result)
    if output_dir:
        phase_started = time.perf_counter()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(summary_buffer, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
        atomic_write_text(output / "evaluation_summary.csv", summary_buffer.getvalue())

        episode_buffer = io.StringIO(newline="")
        writer = csv.writer(episode_buffer)
        writer.writerow(
            [
                "episode", "seed", "model_survival_seconds",
                "termination_reason", "bullet_size", "bullet_speed",
                "scenario_source", "reset_mode",
            ]
        )
        for episode, model_time in enumerate(model_array):
            writer.writerow([
                episode, seed + episode, model_time,
                model_rollout.termination_reasons[episode],
                model_rollout.bullet_sizes[episode],
                model_rollout.bullet_speeds[episode],
                model_rollout.scenario_sources[episode],
                model_rollout.reset_modes[episode],
            ])
        atomic_write_text(
            output / "evaluation_episodes.csv", episode_buffer.getvalue()
        )
        atomic_write_text(
            output / "evaluation_report.txt",
            "Visual-set screen-only evaluation\n"
            + "=" * 34
            + "\n"
            + "\n".join("%s: %s" % item for item in result.items())
            + "\naction_histogram: %s\n"
            % model_rollout.action_histogram.tolist()
            + "checkpoint_step: %s\n" % checkpoint.get("global_step", checkpoint.get("epoch")),
        )
        timing_output["evaluation.write_reports"] = time.perf_counter() - phase_started
    else:
        timing_output["evaluation.write_reports"] = 0.0
    timing_output["evaluation.total"] = time.perf_counter() - total_started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visual-set screen-only policy")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--bullets", type=int, default=40)
    parser.add_argument("--seed", type=int, default=500_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--targeted-bullet-probability", type=float, default=None)
    parser.add_argument("--wall-threshold", type=float, default=40.0)
    parser.add_argument("--max-episode-seconds", type=float, default=None)
    parser.add_argument("--bullet-size-min", type=int, default=None)
    parser.add_argument("--bullet-size-max", type=int, default=None)
    parser.add_argument("--bullet-speed-min", type=float, default=None)
    parser.add_argument("--bullet-speed-max", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.episodes,
        args.bullets,
        args.seed,
        args.device,
        args.output_dir,
        args.targeted_bullet_probability,
        args.wall_threshold,
        args.max_episode_seconds,
        args.bullet_size_min,
        args.bullet_size_max,
        args.bullet_speed_min,
        args.bullet_speed_max,
        args.workers,
    )


if __name__ == "__main__":
    main()
