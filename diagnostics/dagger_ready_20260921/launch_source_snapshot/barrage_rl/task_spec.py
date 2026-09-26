"""Single source of truth for the current 300-bullet deployment task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .timing import DEFAULT_ACTION_REPEAT, PHYSICS_FPS


MIN_TRACKING_CAPACITY = 384
TRACKING_CAPACITY_HEADROOM = 64
TRACKING_CAPACITY_ALIGNMENT = 32


def tracking_capacity_for(bullet_count: int) -> int:
    """Keep temporary/occluded tracks without truncating the current barrage."""

    required = max(
        MIN_TRACKING_CAPACITY,
        int(bullet_count) + TRACKING_CAPACITY_HEADROOM,
    )
    return (
        (required + TRACKING_CAPACITY_ALIGNMENT - 1)
        // TRACKING_CAPACITY_ALIGNMENT
        * TRACKING_CAPACITY_ALIGNMENT
    )


@dataclass(frozen=True)
class BarrageTaskSpec:
    """Environment parameters that define comparable training and evaluation."""

    bullet_count: int = 300
    targeted_bullet_probability: float = 0.10
    bullet_size: int = 5
    bullet_speed: float = 240.0
    observation_size: int = 384
    action_repeat: int = DEFAULT_ACTION_REPEAT
    episode_limit_seconds: float = 120.0
    wall_collision: bool = False
    randomize_initial_phase: bool = False

    def env_kwargs(
        self,
        *,
        episode_limit_seconds: float | None = None,
        observation_size: int | None = None,
    ) -> dict[str, Any]:
        return {
            "bullet_count": int(self.bullet_count),
            "bullet_size_min": int(self.bullet_size),
            "bullet_size_max": int(self.bullet_size),
            "bullet_speed_min": float(self.bullet_speed),
            "bullet_speed_max": float(self.bullet_speed),
            "targeted_bullet_probability": float(
                self.targeted_bullet_probability
            ),
            "observation_size": int(
                self.observation_size
                if observation_size is None
                else observation_size
            ),
            "action_repeat": int(self.action_repeat),
            "max_episode_seconds": float(
                self.episode_limit_seconds
                if episode_limit_seconds is None
                else episode_limit_seconds
            ),
            "wall_collision": bool(self.wall_collision),
            "randomize_initial_phase": bool(self.randomize_initial_phase),
        }

    def manifest(self) -> dict[str, Any]:
        return {**asdict(self), "physics_fps": PHYSICS_FPS}


TARGET_TASK = BarrageTaskSpec()
TARGET_TRACKING_CAPACITY = tracking_capacity_for(TARGET_TASK.bullet_count)

# Production uses policy actions followed by image-derived pixel planning.
# The optional independent analytic geometry diagnostic is disabled by default.
PRODUCTION_ANALYTIC_SHIELD = False
PRODUCTION_ANALYTIC_SHIELD_GATE = "always"
