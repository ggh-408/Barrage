"""Shared image-derived geometry for policy features and safety arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class TrackedPolicyGeometrySpec(Protocol):
    source_size: float
    bullet_speed: float
    collision_radius: float


@dataclass(frozen=True)
class ImageGeometryBelief:
    """Geometry decoded from the current tracked image features."""

    relative_position: torch.Tensor
    measured_bullet_velocity: torch.Tensor
    masks: torch.Tensor


def build_image_geometry_belief(
    spec: TrackedPolicyGeometrySpec,
    objects: torch.Tensor,
    masks: torch.Tensor,
    globals_: torch.Tensor,
) -> ImageGeometryBelief:
    """Decode tracked tensors once for model geometry and safety shields."""

    relative = objects[..., :2] * float(spec.source_size)
    plane_velocity = globals_[:, None, 6:8] * float(spec.bullet_speed)
    measured_bullet_velocity = (
        objects[..., 2:4] * float(spec.bullet_speed) + plane_velocity
    )
    return ImageGeometryBelief(
        relative_position=relative,
        measured_bullet_velocity=measured_bullet_velocity,
        masks=masks,
    )


def constant_action_clearance_by_object(
    belief: ImageGeometryBelief,
    action_vectors: torch.Tensor,
    horizons: torch.Tensor,
    *,
    bullet_speed: float,
    collision_radius: float,
) -> torch.Tensor:
    """Return closest-approach clearance for all actions and horizons."""

    candidate_relative_velocity = (
        belief.measured_bullet_velocity[:, None, :, :]
        - action_vectors[None, :, None, :] * float(bullet_speed)
    )
    relative_position = belief.relative_position[:, None, :, :]
    speed_squared = candidate_relative_velocity.square().sum(dim=-1)
    time_to_closest = -(
        relative_position * candidate_relative_velocity
    ).sum(dim=-1) / speed_squared.clamp_min(1e-6)
    time_to_closest = torch.clamp(time_to_closest, min=0.0)
    time_to_closest = torch.minimum(
        time_to_closest[..., None], horizons[None, None, None, :]
    )
    closest = (
        relative_position[..., None, :]
        + candidate_relative_velocity[..., None, :] * time_to_closest[..., None]
    )
    clearance = torch.linalg.vector_norm(closest, dim=-1) - float(
        collision_radius
    )
    return clearance.masked_fill(~belief.masks[:, None, :, None], torch.inf)
