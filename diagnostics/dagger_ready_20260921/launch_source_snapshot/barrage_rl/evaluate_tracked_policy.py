"""Fixed held-out evaluation for the action-query tracked policy."""

from __future__ import annotations
from barrage_rl.timing import PHYSICS_FPS

import argparse
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_write_json, atomic_write_text, prepare_new_output, atomic_torch_save
from .metrics import (
    bootstrap_confidence_intervals,
    interquartile_mean,
    lower_tail_mean,
    wilson_lower_bound,
)
from .parallel_evaluation import (
    ParallelRolloutProgress,
    ParallelRolloutResult,
    run_parallel_rollout,
)
from .distilled_student import (
    DistilledStudentNetwork,
    DistilledStudentSpec,
    UnifiedDistilledAgent,
)
from .task_spec import (
    BarrageTaskSpec,
    PRODUCTION_ANALYTIC_SHIELD,
    PRODUCTION_ANALYTIC_SHIELD_GATE,
    TARGET_TASK,
)
from .tracked_policy import (
    ActionQueryPolicy,
    TrackedPolicyAgent,
    TrackedPolicySpec,
)

PRODUCTION_EVALUATION_BULLETS = 300


@dataclass
class _RolloutAccumulator:
    """Append fixed evaluation batches without repeatedly copying old results."""

    episodes: int
    completed: int = 0
    survival_times: np.ndarray = field(init=False)
    termination_reasons: list[str] = field(init=False)
    bullet_sizes: np.ndarray = field(init=False)
    bullet_speeds: np.ndarray = field(init=False)
    reset_modes: list[str] = field(init=False)
    minimum_wall_distances: np.ndarray = field(init=False)
    minimum_bullet_clearances: np.ndarray = field(init=False)
    wall_steps: int = 0
    model_steps: int = 0
    action_histogram: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.survival_times = np.empty(self.episodes, dtype=np.float64)
        self.termination_reasons = [""] * self.episodes
        self.bullet_sizes = np.empty(self.episodes, dtype=np.int64)
        self.bullet_speeds = np.empty(self.episodes, dtype=np.float64)
        self.reset_modes = [""] * self.episodes
        self.minimum_wall_distances = np.empty(self.episodes, dtype=np.float64)
        self.minimum_bullet_clearances = np.empty(self.episodes, dtype=np.float64)

    def append(self, rollout: ParallelRolloutResult) -> ParallelRolloutProgress:
        count = len(rollout.survival_times)
        if count <= 0 or self.completed + count > self.episodes:
            raise ValueError("evaluation batch exceeds the configured episode count")
        if not all(
            len(values) == count
            for values in (
                rollout.termination_reasons,
                rollout.bullet_sizes,
                rollout.bullet_speeds,
                rollout.reset_modes,
                rollout.minimum_wall_distances,
            )
        ):
            raise ValueError("evaluation batch fields have inconsistent lengths")
        start = self.completed
        stop = start + count
        destination = slice(start, stop)
        self.survival_times[destination] = rollout.survival_times
        self.termination_reasons[destination] = rollout.termination_reasons
        self.bullet_sizes[destination] = rollout.bullet_sizes
        self.bullet_speeds[destination] = rollout.bullet_speeds
        self.reset_modes[destination] = rollout.reset_modes
        self.minimum_wall_distances[destination] = rollout.minimum_wall_distances
        if len(rollout.minimum_bullet_clearances) == count:
            self.minimum_bullet_clearances[destination] = (
                rollout.minimum_bullet_clearances
            )
        else:
            self.minimum_bullet_clearances[destination] = np.nan
        self.wall_steps += int(rollout.wall_steps)
        self.model_steps += int(rollout.model_steps)
        histogram = np.asarray(rollout.action_histogram, dtype=np.int64)
        if self.action_histogram is None:
            self.action_histogram = np.zeros_like(histogram)
        if histogram.shape != self.action_histogram.shape:
            raise ValueError("evaluation batches have inconsistent action histograms")
        self.action_histogram += histogram
        self.completed = stop
        return ParallelRolloutProgress(
            completed_indices=np.arange(stop, dtype=np.int64),
            survival_times=self.survival_times[:stop],
            termination_reasons=self.termination_reasons[:stop],
            minimum_wall_distances=self.minimum_wall_distances[:stop],
        )

    def result(self) -> ParallelRolloutResult:
        if self.completed != self.episodes or self.action_histogram is None:
            raise RuntimeError("evaluation result requested before all episodes completed")
        return ParallelRolloutResult(
            survival_times=self.survival_times,
            termination_reasons=self.termination_reasons,
            bullet_sizes=self.bullet_sizes,
            bullet_speeds=self.bullet_speeds,
            reset_modes=self.reset_modes,
            minimum_wall_distances=self.minimum_wall_distances,
            wall_steps=self.wall_steps,
            model_steps=self.model_steps,
            action_histogram=self.action_histogram,
            minimum_bullet_clearances=self.minimum_bullet_clearances,
        )


def load_tracked_agent(
    checkpoint_path: str,
    device: torch.device,
    *,
    analytic_shield: bool = False,
    analytic_guard_horizon_seconds: float = 0.10,
    analytic_clearance_margin: float = 0.0,
    analytic_shield_gate: str = "always",
    analytic_min_tracked_count_ratio: float = 0.90,
    analytic_max_model_risk_increase: float = 0.005,
    analytic_max_selected_violation: float = 0.50,
    long_horizon_risk_weight: float = 0.0,
    teacher_cost_ranking_weight: float = 0.0,
    action_hysteresis_bonus: float = 0.0,
    experimental_controller: str | None = None,
) -> tuple[TrackedPolicyAgent, TrackedPolicySpec, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_controller = checkpoint.get("experimental_controller")
    if required_controller is not None and required_controller != experimental_controller:
        raise ValueError(
            "This checkpoint requires its experimental image controller; "
            "use tools/train_targeted_dagger.py --evaluate CHECKPOINT. "
            "The default deployment controller cannot evaluate it interchangeably."
        )
    spec = TrackedPolicySpec(**checkpoint["tracked_policy_spec"])
    model_hparams = dict(checkpoint.get("model_hparams", {}))
    if (
        int(checkpoint.get("model_version", 10)) <= 10
        and "geometry_statistics" not in model_hparams
    ):
        model_hparams["geometry_statistics"] = ("minimum",)
    model = ActionQueryPolicy(
        spec,
        **model_hparams,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    agent_kwargs = {
        "analytic_shield": analytic_shield,
        "analytic_guard_horizon_seconds": analytic_guard_horizon_seconds,
        "analytic_clearance_margin": analytic_clearance_margin,
        "analytic_shield_gate": analytic_shield_gate,
        "analytic_min_tracked_count_ratio": analytic_min_tracked_count_ratio,
        "analytic_max_model_risk_increase": analytic_max_model_risk_increase,
        "analytic_max_selected_violation": analytic_max_selected_violation,
        "long_horizon_risk_weight": long_horizon_risk_weight,
        "teacher_cost_ranking_weight": teacher_cost_ranking_weight,
        "action_hysteresis_bonus": action_hysteresis_bonus,
    }
    if "distilled_student" in checkpoint:
        schema = int(checkpoint.get("distilled_student_schema_version", -1))
        if schema not in DistilledStudentNetwork.compatible_schema_versions:
            raise ValueError(
                f"unsupported distilled student schema: {schema}"
            )
        distilled_spec = DistilledStudentSpec.from_checkpoint(
            checkpoint["distilled_student_spec"]
        )
        distilled = DistilledStudentNetwork(distilled_spec).to(device)
        distilled.load_compatible_state_dict(checkpoint["distilled_student"])
        distilled.eval()
        agent = UnifiedDistilledAgent(
            model,
            distilled,
            device,
            **agent_kwargs,
        )
    else:
        agent = TrackedPolicyAgent(model, device, **agent_kwargs)
    return agent, spec, checkpoint


def checkpoint_action_delay_steps(checkpoint: dict[str, object]) -> int:
    """Return the control timing that the checkpoint was trained to use.

    Current checkpoints record their timing in the saved training config.
    Missing metadata retains immediate, same-boundary action application.
    """
    config = dict(checkpoint.get("config", {}))
    for key in (
        "evaluation_causal_action_delay_steps",
        "collection_causal_action_delay_steps",
    ):
        if key in config:
            delay = int(config[key])
            if delay not in (0, 1):
                raise ValueError(f"checkpoint {key} must be zero or one")
            return delay
    return 0


def _episode_csv(
    episode_indices: np.ndarray | list[int] | range,
    episode_seeds: list[int] | tuple[int, ...],
    survival_times: np.ndarray,
    termination_reasons: list[str],
    minimum_wall_distances: np.ndarray,
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow((
        "episode", "seed", "model_survival_seconds", "termination_reason",
        "minimum_wall_distance",
    ))
    rows = sorted(
        zip(
            (int(value) for value in episode_indices),
            (int(value) for value in episode_seeds),
            (float(value) for value in survival_times),
            termination_reasons,
            (float(value) for value in minimum_wall_distances),
        ),
        key=lambda row: row[0],
    )
    writer.writerows(rows)
    return buffer.getvalue()


def _write_evaluation_progress(
    output: Path,
    progress: ParallelRolloutProgress,
    *,
    all_episode_seeds: tuple[int, ...],
    episodes: int,
    episode_limit_seconds: float,
    finalized: bool = False,
) -> None:
    indices = progress.completed_indices.astype(np.int64, copy=False)
    survival_times = progress.survival_times.astype(np.float32, copy=False)
    completed_seeds = tuple(all_episode_seeds[int(index)] for index in indices)
    atomic_write_text(
        output / "evaluation_episodes.partial.csv",
        _episode_csv(
            indices,
            completed_seeds,
            survival_times,
            progress.termination_reasons,
            progress.minimum_wall_distances,
        ),
    )
    completed = int(len(indices))
    success_count = int(
        np.count_nonzero(survival_times >= episode_limit_seconds)
    )
    atomic_write_json(
        output / "evaluation_progress.json",
        {
            "complete": bool(finalized and completed == episodes),
            "completed_episodes": completed,
            "episodes": int(episodes),
            "success_count": success_count,
            "failure_count": completed - success_count,
            "success_at_limit_so_far": float(success_count / max(completed, 1)),
            "model_mean_so_far": float(survival_times.mean()),
            "model_iqm_so_far": interquartile_mean(survival_times),
            "episode_limit_seconds": float(episode_limit_seconds),
            "physics_fps": PHYSICS_FPS,
        },
    )


def evaluate_tracked_checkpoint(
    checkpoint_path: str,
    *,
    episodes: int = 200,
    workers: int = 10,
    seed: int = 2_000_000,
    output_dir: str = "",
    device_name: str = "cuda",
    episode_limit_seconds: float = 120.0,
    bullet_count: int = PRODUCTION_EVALUATION_BULLETS,
    targeted_bullet_probability: float = TARGET_TASK.targeted_bullet_probability,
    rendered_rgb: bool = True,
    smoke_test: bool = False,
    supplemental_test: bool = False,
    episode_seeds: list[int] | tuple[int, ...] | None = None,
    evaluation_batch_size: int = 10,
    causal_action_delay_steps: int | None = None,
    analytic_shield: bool = PRODUCTION_ANALYTIC_SHIELD,
    analytic_guard_horizon_seconds: float = 0.10,
    analytic_clearance_margin: float = 0.0,
    analytic_shield_gate: str = PRODUCTION_ANALYTIC_SHIELD_GATE,
    analytic_min_tracked_count_ratio: float = 0.90,
    analytic_max_model_risk_increase: float = 0.005,
    analytic_max_selected_violation: float = 0.50,
    long_horizon_risk_weight: float = 0.0,
    teacher_cost_ranking_weight: float = 0.0,
    action_hysteresis_bonus: float = 0.0,
    pixel_guard: str = "receding",
    search_workers: int = 1,
) -> dict[str, float]:
    if supplemental_test and (smoke_test or episode_seeds is None):
        raise ValueError("supplemental testing requires explicit seeds and full evaluation semantics")
    if episodes != 200 and not smoke_test and not supplemental_test:
        raise ValueError("tracked-policy evaluation requires exactly 200 episodes")
    if episodes <= 0 or episode_limit_seconds <= 0.0:
        raise ValueError("episodes and episode_limit_seconds must be positive")
    if not smoke_test and episode_limit_seconds != 120.0:
        raise ValueError("production tracked-policy evaluation is locked to 120 seconds")
    if not smoke_test and bullet_count != PRODUCTION_EVALUATION_BULLETS:
        raise ValueError("production tracked-policy evaluation requires 300 bullets")
    if not smoke_test and not np.isclose(
        targeted_bullet_probability,
        TARGET_TASK.targeted_bullet_probability,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "production tracked-policy evaluation requires targeted bullet probability 0.10"
        )
    if episode_seeds is not None and len(episode_seeds) != episodes:
        raise ValueError("episode_seeds must contain exactly one seed per episode")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    device = torch.device(
        device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
    )
    agent, spec, checkpoint = load_tracked_agent(
        checkpoint_path,
        device,
        analytic_shield=analytic_shield,
        analytic_guard_horizon_seconds=analytic_guard_horizon_seconds,
        analytic_clearance_margin=analytic_clearance_margin,
        analytic_shield_gate=analytic_shield_gate,
        analytic_min_tracked_count_ratio=analytic_min_tracked_count_ratio,
        analytic_max_model_risk_increase=analytic_max_model_risk_increase,
        analytic_max_selected_violation=analytic_max_selected_violation,
        long_horizon_risk_weight=long_horizon_risk_weight,
        teacher_cost_ranking_weight=teacher_cost_ranking_weight,
        action_hysteresis_bonus=action_hysteresis_bonus,
    )
    from .deployment import configure_image_controller
    controller_config = configure_image_controller(agent, pixel_guard, search_workers=search_workers)
    checkpoint_delay_steps = checkpoint_action_delay_steps(checkpoint)
    if causal_action_delay_steps is None:
        causal_action_delay_steps = checkpoint_delay_steps
    if causal_action_delay_steps not in (0, 1):
        raise ValueError("causal_action_delay_steps must be zero or one")
    if not smoke_test and causal_action_delay_steps != checkpoint_delay_steps:
        raise ValueError(
            "production tracked-policy evaluation timing must match the checkpoint: "
            f"requested={causal_action_delay_steps} checkpoint={checkpoint_delay_steps}"
        )
    observation_size = int(checkpoint.get("observation_size", 192))
    normalized_episode_seeds = tuple(
        int(episode_seeds[episode]) if episode_seeds is not None else seed + episode
        for episode in range(episodes)
    )
    output = Path(output_dir) if output_dir else None
    if output is not None:
        prepare_new_output(output)
        atomic_torch_save({"model": agent.model.state_dict(),
                           "model_version": agent.model.model_version},
                          output / "evaluated_model.pt")
        atomic_write_json(
            output / "evaluation_config.json",
            {
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "controller": controller_config,
                "episodes": int(episodes),
                "supplemental_test": bool(supplemental_test),
                "workers": int(workers),
                "seed": int(seed),
                "episode_seeds": list(normalized_episode_seeds),
                "device": str(device),
                "episode_limit_seconds": float(episode_limit_seconds),
                "physics_fps": PHYSICS_FPS,
                "bullet_count": int(bullet_count),
                "targeted_bullet_probability": float(targeted_bullet_probability),
                "rendered_rgb": bool(rendered_rgb),
                "evaluation_batch_size": int(evaluation_batch_size),
                "causal_action_delay_steps": int(
                    causal_action_delay_steps
                ),
                "policy_architecture": "policy_teacher_cost",
                "action_selector_mode": agent.action_selector_mode,
                "analytic_shield": bool(agent.analytic_shield),
                "analytic_guard_horizon_seconds": float(
                    analytic_guard_horizon_seconds
                ),
                "analytic_clearance_margin": float(
                    analytic_clearance_margin
                ),
                "teacher_cost_ranking_weight": float(
                    teacher_cost_ranking_weight
                ),
                "action_hysteresis_bonus": float(action_hysteresis_bonus),
            },
        )
    task = BarrageTaskSpec(
        bullet_count=int(bullet_count),
        targeted_bullet_probability=float(targeted_bullet_probability),
        observation_size=observation_size,
        episode_limit_seconds=float(episode_limit_seconds),
    )
    env_kwargs = task.env_kwargs()
    accumulator = _RolloutAccumulator(episodes)
    batch_size = min(int(evaluation_batch_size), episodes)
    for batch_start in range(0, episodes, batch_size):
        agent.reset_state()
        batch_stop = min(batch_start + batch_size, episodes)
        batch_seeds = normalized_episode_seeds[batch_start:batch_stop]
        batch_rollout = run_parallel_rollout(
            agent=agent,
            spec=spec,
            episodes=len(batch_seeds),
            workers=workers,
            seed=batch_seeds[0],
            env_kwargs=env_kwargs,
            wall_threshold=40.0,
            episode_seeds=batch_seeds,
            rendered_rgb=rendered_rgb,
            causal_action_delay_steps=causal_action_delay_steps,
        )
        progress = accumulator.append(batch_rollout)
        completed_times = accumulator.survival_times[:batch_stop]
        if output is not None:
            _write_evaluation_progress(
                output,
                progress,
                all_episode_seeds=normalized_episode_seeds,
                episodes=episodes,
                episode_limit_seconds=episode_limit_seconds,
            )
        batch_successes = int(
            np.count_nonzero(completed_times >= episode_limit_seconds)
        )
        print(
            "evaluation_batch_progress "
            f"completed={batch_stop}/{episodes} "
            f"success={batch_successes}/{batch_stop} ",
            flush=True,
        )
    rollout = accumulator.result()
    times = rollout.survival_times.astype(np.float32)
    success_count = int(np.count_nonzero(times >= episode_limit_seconds))
    confidence = bootstrap_confidence_intervals(times, seed + 9_000_000)
    result: dict[str, Any] = {
        "policy_architecture": "policy_teacher_cost",
        "supplemental_test": bool(supplemental_test),
        "episodes": float(episodes),
        "bullets": float(bullet_count),
        "seed": float(seed),
        "observation_size": float(observation_size),
        "model_mean": float(times.mean()),
        "model_median": float(np.median(times)),
        "model_iqm": interquartile_mean(times),
        "model_p10": float(np.percentile(times, 10.0)),
        "model_p1": float(np.percentile(times, 1.0)),
        "model_p5": float(np.percentile(times, 5.0)),
        "model_cvar1": lower_tail_mean(times, 0.01),
        "model_cvar5": lower_tail_mean(times, 0.05),
        "model_cvar90": lower_tail_mean(times, 0.90),
        "model_rmst": float(np.minimum(times, episode_limit_seconds).mean()),
        "episode_limit_seconds": float(episode_limit_seconds),
        "physics_fps": PHYSICS_FPS,
        "bullet_size_min": 5.0,
        "bullet_size_max": 5.0,
        "bullet_speed_min": 240.0,
        "bullet_speed_max": 240.0,
        "targeted_bullet_probability": float(targeted_bullet_probability),
        "rendered_rgb": bool(rendered_rgb),
        "causal_action_delay_steps": float(causal_action_delay_steps),
        "success_at_limit": float(success_count / episodes),
        "success_at_limit_ci95_low": wilson_lower_bound(success_count, episodes),
        "failure_before_1s": float(np.mean(times < 1.0)),
        "failure_before_10s": float(np.mean(times < 10.0)),
        "wall_threshold": 40.0,
        "wall_step_fraction": float(
            rollout.wall_steps / max(rollout.model_steps, 1)
        ),
        "wall_episode_fraction": float(
            np.mean(rollout.minimum_wall_distances < 40.0)
        ),
        "median_min_wall_distance": float(
            np.median(rollout.minimum_wall_distances)
        ),
        "controller_overridden_decisions": float(agent.overridden_decision_count),
        "controller_decisions": float(agent.decision_count),
        "action_selector_mode": agent.action_selector_mode,
        "analytic_guard_horizon_seconds": float(
            analytic_guard_horizon_seconds
        ),
        "analytic_clearance_margin": float(analytic_clearance_margin),
        "teacher_cost_ranking_weight": float(teacher_cost_ranking_weight),
        "action_hysteresis_bonus": float(action_hysteresis_bonus),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "controller": controller_config,
        **confidence,
    }
    if isinstance(agent, UnifiedDistilledAgent):
        result.update({
            "distilled_student_schema_version": float(
                agent.distilled.schema_version
            ),
            "distilled_student_decisions": float(
                agent.student_decision_count
            ),
        })
    print(
        "tracked_evaluation "
        f"episodes={episodes} mean={result['model_mean']:.2f}s "
        f"median={result['model_median']:.2f}s iqm={result['model_iqm']:.2f}s "
        f"success={100.0 * result['success_at_limit']:.1f}%",
        flush=True,
    )
    if output is not None:
        atomic_write_json(output / "evaluation_summary.json", result)
        atomic_write_text(
            output / "evaluation_episodes.csv",
            _episode_csv(
                range(episodes),
                normalized_episode_seeds,
                times,
                rollout.termination_reasons,
                rollout.minimum_wall_distances,
            ),
        )
        atomic_write_json(
            output / "action_histogram.json", rollout.action_histogram.tolist()
        )
        _write_evaluation_progress(
            output,
            progress,
            all_episode_seeds=normalized_episode_seeds,
            episodes=episodes,
            episode_limit_seconds=episode_limit_seconds,
            finalized=True,
        )
    return result


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tracked policy")
    parser.add_argument("--pixel-guard", choices=("off", "receding"), default="receding")
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=200, choices=(200,))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bullets", type=int, default=PRODUCTION_EVALUATION_BULLETS)
    parser.add_argument(
        "--targeted-probability",
        type=float,
        default=TARGET_TASK.targeted_bullet_probability,
    )
    parser.add_argument("--ideal-semantic", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--smoke-limit-seconds", type=float, default=5.0)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=10,
        help="episodes per parallel batch; each completed batch commits progress",
    )
    parser.add_argument(
        "--causal-action-delay-steps",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "override checkpoint timing for a smoke diagnostic; production "
            "evaluation automatically uses the checkpoint's trained timing"
        ),
    )
    parser.add_argument(
        "--analytic-shield",
        action=argparse.BooleanOptionalAction,
        default=PRODUCTION_ANALYTIC_SHIELD,
        help=(
            "enable the optional independent image-derived analytic "
            "safety shield"
        ),
    )
    parser.add_argument(
        "--analytic-guard-horizon-seconds", type=float, default=0.10
    )
    parser.add_argument("--analytic-clearance-margin", type=float, default=0.0)
    parser.add_argument("--teacher-cost-ranking-weight", type=float, default=0.0)
    parser.add_argument("--action-hysteresis-bonus", type=float, default=0.0)
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    evaluate_tracked_checkpoint(
        args.checkpoint,
        episodes=args.smoke_episodes if args.smoke_test else args.episodes,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        pixel_guard=args.pixel_guard,
        search_workers=args.search_workers,
        device_name=args.device,
        episode_limit_seconds=(
            args.smoke_limit_seconds if args.smoke_test else 120.0
        ),
        bullet_count=args.bullets,
        targeted_bullet_probability=args.targeted_probability,
        rendered_rgb=not args.ideal_semantic,
        smoke_test=args.smoke_test,
        evaluation_batch_size=args.evaluation_batch_size,
        causal_action_delay_steps=args.causal_action_delay_steps,
        analytic_shield=args.analytic_shield,
        analytic_guard_horizon_seconds=args.analytic_guard_horizon_seconds,
        analytic_clearance_margin=args.analytic_clearance_margin,

        teacher_cost_ranking_weight=args.teacher_cost_ranking_weight,
        action_hysteresis_bonus=args.action_hysteresis_bonus,
    )


if __name__ == "__main__":
    main()
