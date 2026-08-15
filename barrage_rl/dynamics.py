"""Shared deterministic barrage dynamics used by training and the real game.

Only geometry and random spawning live here.  Keeping these functions free of
Gym/Pygame state prevents the simulator, teacher, and ``Barrage.py`` from
silently learning three different games.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


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
