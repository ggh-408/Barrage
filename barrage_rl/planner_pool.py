"""True multi-core execution for the exact privileged planner."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Sequence, Tuple

from .baselines import PlannerSupervision, privileged_planner_supervision
from .env import BarrageVisionEnv, EnvSnapshot


_WORKER_ENV: BarrageVisionEnv | None = None
_WORKER_PLANNER_ARGS: Tuple[float, float, Tuple[float, ...]] | None = None


def _worker_environment_kwargs(env: BarrageVisionEnv) -> Dict[str, Any]:
    """Capture the static simulator settings needed by planner workers."""
    return {
        "bullet_count": env.bullet_count,
        "bullet_size": env.bullet_size,
        "bullet_size_min": env.bullet_size_min,
        "bullet_size_max": env.bullet_size_max,
        "bullet_speed_min": env.bullet_speed_min,
        "bullet_speed_max": env.bullet_speed_max,
        "observation_size": env.observation_size,
        "frame_stack": env.frame_stack,
        "action_repeat": env.action_repeat,
        "max_episode_seconds": env.max_episode_steps / env.metadata["render_fps"],
        "wall_collision": env.wall_collision,
        "render_mode": None,
        "screen_width": env.screen_width,
        "screen_height": env.screen_height,
        "randomize_initial_phase": False,
        "initial_phase_min_seconds": env.initial_phase_min_seconds,
        "initial_phase_max_seconds": env.initial_phase_max_seconds,
        "dense_reward_scale": env.dense_reward_scale,
        "danger_horizon_seconds": env.danger_horizon_seconds,
        "targeted_bullet_probability": env.targeted_bullet_probability,
        "targeted_prediction_scale_min": env.targeted_prediction_scale_min,
        "targeted_prediction_scale_max": env.targeted_prediction_scale_max,
        "targeted_angular_noise": env.targeted_angular_noise,
        "scenario_mix": False,
        "core_bullet_size": env.core_bullet_size,
        "core_bullet_speed": env.core_bullet_speed,
        "stress_targeted_bullet_probability": env.stress_targeted_bullet_probability,
        "recoverability_seconds": env.recoverability_seconds,
    }


def _planner_task(env: BarrageVisionEnv) -> tuple:
    """Serialize only simulator state used by exact planning."""
    snapshot = env.capture_state()
    compact_snapshot = EnvSnapshot(
        plane_position=snapshot.plane_position,
        plane_velocity=snapshot.plane_velocity,
        bullet_positions=snapshot.bullet_positions,
        bullet_velocities=snapshot.bullet_velocities,
        bullet_is_targeted=snapshot.bullet_is_targeted,
        frames=(),
        episode_steps=snapshot.episode_steps,
        physics_steps=snapshot.physics_steps,
        rng_state=snapshot.rng_state,
    )
    return (
        compact_snapshot,
        int(env.bullet_size),
        float(env.bullet_speed),
        float(env.targeted_bullet_probability),
    )


def _initialize_worker(
    environment_kwargs: Dict[str, Any],
    horizon_seconds: float,
    reaction_seconds: float,
    safety_horizons: Tuple[float, ...],
) -> None:
    global _WORKER_ENV, _WORKER_PLANNER_ARGS
    _WORKER_ENV = BarrageVisionEnv(**environment_kwargs)
    _WORKER_PLANNER_ARGS = (
        float(horizon_seconds),
        float(reaction_seconds),
        tuple(float(value) for value in safety_horizons),
    )


def _supervise_task(task: tuple) -> PlannerSupervision:
    if _WORKER_ENV is None or _WORKER_PLANNER_ARGS is None:
        raise RuntimeError("planner worker was not initialized")
    snapshot, bullet_size, bullet_speed, targeted_probability = task
    env = _WORKER_ENV
    env.bullet_count = len(snapshot.bullet_positions)
    env.bullet_size = int(bullet_size)
    env.bullet_speed = float(bullet_speed)
    env.targeted_bullet_probability = float(targeted_probability)
    env._load_collision_assets()
    env.restore_state(snapshot)
    horizon_seconds, reaction_seconds, safety_horizons = _WORKER_PLANNER_ARGS
    return privileged_planner_supervision(
        env,
        horizon_seconds=horizon_seconds,
        reaction_seconds=reaction_seconds,
        safety_horizons=safety_horizons,
    )


class ProcessPlannerPool:
    """Keep one simulator per process and distribute independent states."""

    def __init__(
        self,
        template_env: BarrageVisionEnv,
        workers: int,
        horizon_seconds: float,
        reaction_seconds: float,
        safety_horizons: Sequence[float],
    ) -> None:
        if workers < 2:
            raise ValueError("ProcessPlannerPool requires at least two workers")
        horizons = tuple(float(value) for value in safety_horizons)
        self.executor = ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_initialize_worker,
            initargs=(
                _worker_environment_kwargs(template_env),
                float(horizon_seconds),
                float(reaction_seconds),
                horizons,
            ),
        )

    def supervise(self, envs: Sequence[BarrageVisionEnv]) -> list[PlannerSupervision]:
        return list(self.executor.map(_supervise_task, map(_planner_task, envs), chunksize=1))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
