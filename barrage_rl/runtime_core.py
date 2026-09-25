"""Single shared game kernel used by training, evaluation, and the real game.

Only geometry and random spawning live here.  Keeping these functions free of
Gym/Pygame state prevents the simulator, teacher, and ``Barrage.py`` from
silently learning three different games.
"""

from __future__ import annotations

from .timing import PHYSICS_FPS

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


__all__ = [
    "ACTION_VECTORS",
    "BulletFieldConfig",
    "DIAGONAL_COMPONENT",
    "OpeningBulletLayout",
    "OPENING_BATCH_COUNT",
    "OPENING_BATCH_INTERVAL_SECONDS",
    "advance_bullet_field",
    "advance_plane",
    "colliding_bullet_indices",
    "initialize_opening_bullets",
    "opening_batch_physics_step",
    "opening_batch_size",
    "normalized_direction",
    "render_world_surface",
    "respawn_bullet_indices",
    "sample_episode_parameters",
    "snapshot_surface_rgb",
    "spawn_bullet",
    "spawn_bullets",
    "targeted_velocity",
    "tracker_prediction_hints",
]


OPENING_BATCH_COUNT = 10
OPENING_BATCH_INTERVAL_SECONDS = 0.1
DIAGONAL_COMPONENT = 2.0 ** -0.5
ACTION_VECTORS = np.asarray(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-DIAGONAL_COMPONENT, -DIAGONAL_COMPONENT],
        [DIAGONAL_COMPONENT, -DIAGONAL_COMPONENT],
        [-DIAGONAL_COMPONENT, DIAGONAL_COMPONENT],
        [DIAGONAL_COMPONENT, DIAGONAL_COMPONENT],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class OpeningBulletLayout:
    """A staggered opening field produced without safety-gate rejection."""

    positions: np.ndarray
    velocities: np.ndarray
    targeted: np.ndarray
    phase_max_seconds: float


@dataclass(frozen=True)
class BulletFieldConfig:
    """Immutable bullet lifecycle rules shared by every runtime adapter."""

    wall_collision: bool
    screen_width: float
    screen_height: float
    bullet_speed: float
    targeted_probability: float
    prediction_scale_min: float
    prediction_scale_max: float
    angular_noise: float


def opening_batch_size(
    total_count: int,
    batch_index: int,
    batch_count: int = OPENING_BATCH_COUNT,
) -> int:
    """Return one opening batch size without leaving a legal opening empty."""
    total_count = int(total_count)
    batch_index = int(batch_index)
    batch_count = int(batch_count)
    if total_count < 0:
        raise ValueError("total_count must be non-negative")
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if not 0 <= batch_index < batch_count:
        raise ValueError("batch_index is out of range")
    if 0 < total_count < batch_count:
        return int(batch_index < total_count)
    regular_size = total_count // batch_count
    if batch_index == batch_count - 1:
        return total_count - regular_size * (batch_count - 1)
    return regular_size


def opening_batch_physics_step(
    batch_index: int,
    physics_fps: int = PHYSICS_FPS,
    interval_seconds: float = OPENING_BATCH_INTERVAL_SECONDS,
) -> int:
    """Return the elapsed physics-step boundary for one opening batch."""
    batch_index = int(batch_index)
    physics_fps = int(physics_fps)
    interval_seconds = float(interval_seconds)
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    if physics_fps < 1 or interval_seconds <= 0.0:
        raise ValueError("physics_fps and interval_seconds must be positive")
    return int(round(batch_index * interval_seconds * physics_fps))


def normalized_direction(x: float, y: float) -> np.ndarray:
    """Return a unit-length player direction while preserving zero input."""
    direction = np.asarray([x, y], dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length > 1.0:
        direction /= length
    return direction


def targeted_velocity(
    spawn_positions: np.ndarray,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    bullet_speed: float,
    prediction_scale: np.ndarray,
    angular_noise: np.ndarray,
) -> np.ndarray:
    """Aim bullets at a noisy, under-predicted interception point."""
    spawn_positions = np.asarray(spawn_positions, dtype=np.float64)
    plane_position = np.asarray(plane_position, dtype=np.float64)
    plane_velocity = np.asarray(plane_velocity, dtype=np.float64)
    relative = plane_position[None, :] - spawn_positions
    speed = float(bullet_speed)
    a = float(np.dot(plane_velocity, plane_velocity) - speed * speed)
    b = 2.0 * (relative @ plane_velocity)
    c = np.sum(relative * relative, axis=1)
    discriminant = np.maximum(b * b - 4.0 * a * c, 0.0)
    if abs(a) > 1e-8:
        first = (-b - np.sqrt(discriminant)) / (2.0 * a)
        second = (-b + np.sqrt(discriminant)) / (2.0 * a)
        roots = np.column_stack((first, second))
        roots[roots <= 0.0] = np.inf
        intercept_time = roots.min(axis=1)
    else:
        intercept_time = np.where(b < -1e-8, -c / b, np.inf)
    fallback_time = np.sqrt(c) / max(speed, 1e-8)
    intercept_time = np.where(
        np.isfinite(intercept_time), intercept_time, fallback_time
    )
    target = (
        plane_position[None, :]
        + plane_velocity[None, :]
        * intercept_time[:, None]
        * np.asarray(prediction_scale, dtype=np.float64)[:, None]
    )
    aim = target - spawn_positions
    angle = np.arctan2(aim[:, 1], aim[:, 0]) + np.asarray(
        angular_noise, dtype=np.float64
    )
    return np.column_stack((speed * np.cos(angle), speed * np.sin(angle))).astype(
        np.float32
    )


def spawn_bullets(
    count: int,
    screen_width: float,
    screen_height: float,
    bullet_speed: float,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    targeted_probability: float,
    prediction_scale_min: float,
    prediction_scale_max: float,
    angular_noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate edge positions, velocities, and targeted flags.

    Every call consumes random values in one fixed order.  The headless
    simulator and real game can therefore be compared seed-for-seed.
    """
    count = int(count)
    walls = rng.integers(0, 4, size=count)
    angles = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=count)
    horizontal = float(bullet_speed) * np.cos(angles)
    vertical = float(bullet_speed) * np.sin(angles)
    random_x = rng.random(count) * float(screen_width)
    random_y = rng.random(count) * float(screen_height)
    positions = np.empty((count, 2), dtype=np.float32)
    velocities = np.empty((count, 2), dtype=np.float32)

    left = walls == 0
    right = walls == 1
    top = walls == 2
    bottom = walls == 3
    positions[left] = np.column_stack((np.zeros(left.sum()), random_y[left]))
    velocities[left] = np.column_stack((horizontal[left], vertical[left]))
    positions[right] = np.column_stack(
        (np.full(right.sum(), screen_width), random_y[right])
    )
    velocities[right] = np.column_stack((-horizontal[right], vertical[right]))
    positions[top] = np.column_stack((random_x[top], np.zeros(top.sum())))
    velocities[top] = np.column_stack((vertical[top], horizontal[top]))
    positions[bottom] = np.column_stack(
        (random_x[bottom], np.full(bottom.sum(), screen_height))
    )
    velocities[bottom] = np.column_stack((vertical[bottom], -horizontal[bottom]))

    targeted = rng.random(count) < float(targeted_probability)
    if np.any(targeted):
        scale = rng.uniform(
            float(prediction_scale_min), float(prediction_scale_max), targeted.sum()
        )
        noise = (
            rng.normal(0.0, float(angular_noise), targeted.sum())
            if angular_noise > 0.0
            else np.zeros(targeted.sum(), dtype=np.float64)
        )
        velocities[targeted] = targeted_velocity(
            positions[targeted], plane_position, plane_velocity, bullet_speed,
            scale, noise,
        )
    return positions, velocities, targeted.astype(np.bool_)


def spawn_bullet(
    screen_width: float,
    screen_height: float,
    bullet_speed: float,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    targeted_probability: float,
    prediction_scale_min: float,
    prediction_scale_max: float,
    angular_noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generate one complete bullet with ``spawn_bullets(1)`` semantics.

    The scalar path avoids allocating the masks and temporary arrays used by
    the intentionally field-batched opening generator.  Its random draws and
    float32 outputs remain exactly equivalent to ``spawn_bullets(1)``.
    """
    wall = int(rng.integers(0, 4))
    angle = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
    horizontal = float(bullet_speed) * np.cos(angle)
    vertical = float(bullet_speed) * np.sin(angle)
    random_x = float(rng.random()) * float(screen_width)
    random_y = float(rng.random()) * float(screen_height)
    position = np.empty(2, dtype=np.float32)
    velocity = np.empty(2, dtype=np.float32)
    if wall == 0:
        position[:] = (0.0, random_y)
        velocity[:] = (horizontal, vertical)
    elif wall == 1:
        position[:] = (screen_width, random_y)
        velocity[:] = (-horizontal, vertical)
    elif wall == 2:
        position[:] = (random_x, 0.0)
        velocity[:] = (vertical, horizontal)
    else:
        position[:] = (random_x, screen_height)
        velocity[:] = (vertical, -horizontal)

    targeted = bool(rng.random() < float(targeted_probability))
    if targeted:
        scale = np.asarray(
            [rng.uniform(prediction_scale_min, prediction_scale_max)],
            dtype=np.float64,
        )
        noise = np.asarray(
            [rng.normal(0.0, angular_noise) if angular_noise > 0.0 else 0.0],
            dtype=np.float64,
        )
        velocity[:] = targeted_velocity(
            position[None, :],
            plane_position,
            plane_velocity,
            bullet_speed,
            scale,
            noise,
        )[0]
    return position, velocity, targeted


def sample_episode_parameters(
    bullet_size_min: int,
    bullet_size_max: int,
    bullet_speed_min: float,
    bullet_speed_max: float,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Sample task parameters without consuming RNG for fixed ranges."""
    bullet_size = (
        int(bullet_size_min)
        if int(bullet_size_min) == int(bullet_size_max)
        else int(rng.integers(int(bullet_size_min), int(bullet_size_max) + 1))
    )
    bullet_speed = (
        float(bullet_speed_min)
        if float(bullet_speed_min) == float(bullet_speed_max)
        else float(rng.uniform(float(bullet_speed_min), float(bullet_speed_max)))
    )
    return bullet_size, bullet_speed


def advance_plane(
    position: np.ndarray,
    direction: np.ndarray,
    speed: float,
    delta_time: float,
    half_size: np.ndarray,
    screen_width: float,
    screen_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance and clamp one plane using the canonical float32 arithmetic."""
    previous = np.asarray(position, dtype=np.float32)
    updated = previous.copy()
    updated += np.asarray(direction, dtype=np.float32) * float(speed) * float(delta_time)
    half_size = np.asarray(half_size, dtype=np.float32)
    updated[0] = np.clip(
        updated[0], half_size[0], float(screen_width) - half_size[0]
    )
    updated[1] = np.clip(
        updated[1], half_size[1], float(screen_height) - half_size[1]
    )
    velocity = (updated - previous) / max(float(delta_time), 1e-8)
    return updated, velocity.astype(np.float32, copy=False)


def respawn_bullet_indices(
    positions: np.ndarray,
    velocities: np.ndarray,
    targeted: np.ndarray,
    indices: np.ndarray,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    rng: np.random.Generator,
    config: BulletFieldConfig,
) -> None:
    """Respawn complete bullets sequentially in stable index order."""
    for index in np.sort(np.asarray(indices, dtype=np.int64).reshape(-1)):
        position, velocity, is_targeted = spawn_bullet(
            config.screen_width,
            config.screen_height,
            config.bullet_speed,
            plane_position,
            plane_velocity,
            config.targeted_probability,
            config.prediction_scale_min,
            config.prediction_scale_max,
            config.angular_noise,
            rng,
        )
        positions[index] = position
        velocities[index] = velocity
        targeted[index] = is_targeted


def advance_bullet_field(
    positions: np.ndarray,
    velocities: np.ndarray,
    targeted: np.ndarray,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    rng: np.random.Generator,
    delta_time: float,
    config: BulletFieldConfig,
    *,
    respawn_outside: bool = True,
) -> None:
    """Advance the complete bullet lifecycle by one fixed physics substep."""
    if config.wall_collision:
        left = positions[:, 0] < 0.0
        right = positions[:, 0] > config.screen_width
        top = positions[:, 1] < 0.0
        bottom = positions[:, 1] > config.screen_height
        positions[left, 0] *= -1.0
        positions[right, 0] = 2.0 * config.screen_width - positions[right, 0]
        positions[top, 1] *= -1.0
        positions[bottom, 1] = 2.0 * config.screen_height - positions[bottom, 1]
        velocities[left | right, 0] *= -1.0
        velocities[top | bottom, 1] *= -1.0
    elif respawn_outside:
        outside = (
            (positions[:, 0] < 0.0)
            | (positions[:, 0] > config.screen_width)
            | (positions[:, 1] < 0.0)
            | (positions[:, 1] > config.screen_height)
        )
        respawn_bullet_indices(
            positions,
            velocities,
            targeted,
            np.flatnonzero(outside),
            plane_position,
            plane_velocity,
            rng,
            config,
        )
    positions += velocities * float(delta_time)


def colliding_bullet_indices(
    plane_position: np.ndarray,
    bullet_positions: np.ndarray,
    plane_surface: Any,
    bullet_surface: Any,
    plane_mask: Any,
    bullet_mask: Any,
) -> np.ndarray:
    """Return pixel-mask collisions using the same Pygame center rounding."""
    plane_position = np.asarray(plane_position, dtype=np.float32)
    bullet_positions = np.asarray(bullet_positions, dtype=np.float32)
    plane_rect = plane_surface.get_rect(center=tuple(plane_position))
    half_extent = max(plane_surface.get_width(), plane_surface.get_height())
    nearby = (
        (np.abs(bullet_positions[:, 0] - plane_position[0]) <= half_extent)
        & (np.abs(bullet_positions[:, 1] - plane_position[1]) <= half_extent)
    )
    collisions: list[int] = []
    for bullet_index in np.flatnonzero(nearby):
        bullet_rect = bullet_surface.get_rect(
            center=tuple(bullet_positions[bullet_index])
        )
        if not plane_rect.colliderect(bullet_rect):
            continue
        offset = (
            bullet_rect.left - plane_rect.left,
            bullet_rect.top - plane_rect.top,
        )
        if plane_mask.overlap(bullet_mask, offset) is not None:
            collisions.append(int(bullet_index))
    return np.asarray(collisions, dtype=np.int64)


def render_world_surface(
    surface: Any,
    plane_surface: Any,
    plane_position: np.ndarray,
    bullet_surface: Any,
    bullet_positions: np.ndarray,
) -> None:
    """Draw the canonical world-only RGB frame without UI overlays."""
    surface.fill((0, 0, 0))
    surface.blit(
        plane_surface,
        plane_surface.get_rect(center=tuple(plane_position)),
    )
    blits = getattr(surface, "blits", None)
    if blits is None:
        # Keep adapters exposing only the original single-blit interface valid.
        for position in np.asarray(bullet_positions):
            surface.blit(
                bullet_surface,
                bullet_surface.get_rect(center=tuple(position)),
            )
        return
    # Submit the same ordered rectangles in one C call. Keep Pygame's existing
    # center rounding and default alpha blending, including overlapping bullets.
    bullet_rect = bullet_surface.get_rect
    blits(
        [
            (bullet_surface, bullet_rect(center=tuple(position)))
            for position in np.asarray(bullet_positions)
        ],
        doreturn=False,
    )


def snapshot_surface_rgb(surface: Any) -> np.ndarray:
    """Copy current pixels into an owned, read-only HWC RGB snapshot.

    Copy a temporary SDL channel view into independent RGB storage. Retain
    the byte-export fallback when the optional kernel is unavailable.
    Subsequent rendering cannot change the returned frame; no surface lock
    survives this call.
    """
    import pygame

    width, height = surface.get_size()
    # SDL's byte export expands reduced-depth channels differently from
    # surfarray (e.g. RGB565). Keep the established pixel interpretation there.
    if any(surface.get_losses()[:3]):
        rgb = np.ascontiguousarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
        rgb.setflags(write=False)
        return rgb
    from .rgb_capture_kernel import copy_rgb
    if copy_rgb is not None and width and height and surface.get_bitsize() in (24, 32):
        view = np.asarray(surface.get_view('3')).transpose(1, 0, 2)
        rgb = copy_rgb(view)
        rgb.setflags(write=False)
        return rgb
    export_bytes = getattr(pygame.image, "tobytes", pygame.image.tostring)
    return np.frombuffer(export_bytes(surface, "RGB"), dtype=np.uint8).reshape(
        height, width, 3
    )


def tracker_prediction_hints(
    tracker: Any,
    rgb_shape: tuple[int, ...],
    decision_steps: int = 1,
    *, reuse_for_update: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build image-derived detector search windows from prior tracks."""
    tracks = tracker.tracks
    if not tracks:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    known = np.asarray([track.velocity_known for track in tracks], dtype=np.bool_)
    positions = tracker.predict_positions(decision_steps)
    if reuse_for_update:
        tracker._pending_predictions = (tracker._step, max(1, int(decision_steps)), positions)
    positions = positions.astype(np.float32, copy=False)
    uncertainty = np.asarray(
        [track.position_uncertainty for track in tracks], dtype=np.float32
    )
    steps = max(1, int(decision_steps))
    known_radius = np.clip(uncertainty + 4.0 + 2.0 * (steps - 1), 6.0, 24.0)
    unknown_radius = np.clip(uncertainty + 8.0 * steps, 24.0, 64.0)
    source_radii = np.where(known, known_radius, unknown_radius)
    height, width = rgb_shape[:2]
    pixel_scale = max(width, height) / max(tracker.source_size, 1.0)
    return positions / tracker.source_size, source_radii * pixel_scale


def _advance_staggered_phases(
    positions: np.ndarray,
    velocities: np.ndarray,
    targeted: np.ndarray,
    phase_steps: np.ndarray,
    *,
    screen_width: float,
    screen_height: float,
    bullet_speed: float,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    targeted_probability: float,
    prediction_scale_min: float,
    prediction_scale_max: float,
    angular_noise: float,
    physics_fps: int,
    wall_collision: bool,
    rng: np.random.Generator,
) -> None:
    """Advance independent virtual bullet ages through the normal lifecycle."""
    delta_time = 1.0 / int(physics_fps)
    if not wall_collision:
        # Event-driven advancement is exactly equivalent to checking bounds on
        # every physics step, but avoids separate small NumPy passes per physics step.
        step_velocity = velocities * delta_time
        exit_steps = np.full(len(positions), np.inf, np.float64)
        positive_x = step_velocity[:, 0] > 0.0
        negative_x = step_velocity[:, 0] < 0.0
        positive_y = step_velocity[:, 1] > 0.0
        negative_y = step_velocity[:, 1] < 0.0
        exit_steps[positive_x] = np.minimum(
            exit_steps[positive_x],
            np.floor(
                (float(screen_width) - positions[positive_x, 0])
                / step_velocity[positive_x, 0]
            ) + 1.0,
        )
        exit_steps[negative_x] = np.minimum(
            exit_steps[negative_x],
            np.floor(
                positions[negative_x, 0] / -step_velocity[negative_x, 0]
            ) + 1.0,
        )
        exit_steps[positive_y] = np.minimum(
            exit_steps[positive_y],
            np.floor(
                (float(screen_height) - positions[positive_y, 1])
                / step_velocity[positive_y, 1]
            ) + 1.0,
        )
        exit_steps[negative_y] = np.minimum(
            exit_steps[negative_y],
            np.floor(
                positions[negative_y, 1] / -step_velocity[negative_y, 1]
            ) + 1.0,
        )
        direct = phase_steps <= exit_steps
        positions[direct] += (
            step_velocity[direct] * phase_steps[direct, None]
        )

        for index in np.flatnonzero(~direct):
            first_segment = int(exit_steps[index])
            positions[index] += step_velocity[index] * first_segment
            remaining = int(phase_steps[index]) - first_segment
            while remaining > 0:
                new_position, new_velocity, new_targeted = spawn_bullet(
                    screen_width,
                    screen_height,
                    bullet_speed,
                    plane_position,
                    plane_velocity,
                    targeted_probability,
                    prediction_scale_min,
                    prediction_scale_max,
                    angular_noise,
                    rng,
                )
                positions[index] = new_position
                velocities[index] = new_velocity
                targeted[index] = new_targeted
                position = positions[index]
                velocity = velocities[index]
                candidates = []
                if velocity[0] > 0.0:
                    candidates.append(
                        int(
                            np.floor(
                                (float(screen_width) - position[0])
                                / (velocity[0] * delta_time)
                            )
                        )
                        + 1
                    )
                elif velocity[0] < 0.0:
                    candidates.append(
                        int(np.floor(position[0] / (-velocity[0] * delta_time))) + 1
                    )
                if velocity[1] > 0.0:
                    candidates.append(
                        int(
                            np.floor(
                                (float(screen_height) - position[1])
                                / (velocity[1] * delta_time)
                            )
                        )
                        + 1
                    )
                elif velocity[1] < 0.0:
                    candidates.append(
                        int(np.floor(position[1] / (-velocity[1] * delta_time))) + 1
                    )
                steps_to_exit = max(1, min(candidates))
                if remaining <= steps_to_exit:
                    positions[index] += velocity * (remaining * delta_time)
                    break
                positions[index] += velocity * (steps_to_exit * delta_time)
                remaining -= steps_to_exit
        return

    for step in range(int(phase_steps.max(initial=0))):
        active = phase_steps > step
        left = active & (positions[:, 0] < 0.0)
        right = active & (positions[:, 0] > float(screen_width))
        top = active & (positions[:, 1] < 0.0)
        bottom = active & (positions[:, 1] > float(screen_height))
        positions[left, 0] *= -1.0
        positions[right, 0] = 2.0 * float(screen_width) - positions[right, 0]
        positions[top, 1] *= -1.0
        positions[bottom, 1] = 2.0 * float(screen_height) - positions[bottom, 1]
        velocities[left | right, 0] *= -1.0
        velocities[top | bottom, 1] *= -1.0
        positions[active] += velocities[active] * delta_time


def initialize_opening_bullets(
    count: int,
    screen_width: float,
    screen_height: float,
    bullet_speed: float,
    plane_position: np.ndarray,
    plane_velocity: np.ndarray,
    targeted_probability: float,
    prediction_scale_min: float,
    prediction_scale_max: float,
    angular_noise: float,
    rng: np.random.Generator,
    *,
    phase_max_seconds: float = 0.6,
    physics_fps: int = PHYSICS_FPS,
    wall_collision: bool = False,
) -> OpeningBulletLayout:
    """Create one opening field with independent fixed-range virtual ages."""
    count = int(count)
    plane_position = np.asarray(plane_position, dtype=np.float32)
    effective_phase = float(phase_max_seconds)
    maximum_phase_steps = int(np.floor(effective_phase * int(physics_fps)))

    positions, velocities, targeted = spawn_bullets(
        count,
        screen_width,
        screen_height,
        bullet_speed,
        plane_position,
        plane_velocity,
        targeted_probability,
        prediction_scale_min,
        prediction_scale_max,
        angular_noise,
        rng,
    )
    phase_steps = rng.integers(0, maximum_phase_steps + 1, size=count)
    _advance_staggered_phases(
        positions,
        velocities,
        targeted,
        phase_steps,
        screen_width=screen_width,
        screen_height=screen_height,
        bullet_speed=bullet_speed,
        plane_position=plane_position,
        plane_velocity=plane_velocity,
        targeted_probability=targeted_probability,
        prediction_scale_min=prediction_scale_min,
        prediction_scale_max=prediction_scale_max,
        angular_noise=angular_noise,
        physics_fps=physics_fps,
        wall_collision=wall_collision,
        rng=rng,
    )
    return OpeningBulletLayout(
        positions,
        velocities,
        targeted,
        effective_phase,
    )
