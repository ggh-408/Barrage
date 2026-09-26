"""Opt-in, exact inference helpers for the single-frame window controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .tracked_policy import ActionQueryPolicy


class _ActionQueryCache:
    """Reuse the image-independent query prefix while model weights are fixed.

    The cache is deliberately absent from checkpoints. Tensor version counters
    invalidate it after optimizer updates and ordinary in-place edits; tensor
    identities also cover parameter replacement and ``load_state_dict(assign=True)``.
    Training, gradients, autocast, and tensors without version counters use the
    original calculation. Only the window controller enables this helper.
    """

    def __init__(self) -> None:
        self._key: tuple[object, ...] | None = None
        self._value: torch.Tensor | None = None
        self._dependencies: tuple[object, ...] = ()

    @staticmethod
    def _compute(model: ActionQueryPolicy) -> torch.Tensor:
        action_ids = torch.arange(
            model.action_count, device=model.action_vectors.device
        )
        return (
            model.action_embedding(action_ids)[None, :, :]
            + model.action_vector_encoder(model.action_vectors)[None, :, :]
        )

    def get(self, model: ActionQueryPolicy) -> torch.Tensor:
        modules = (
            model.action_embedding,
            *model.action_vector_encoder.modules(),
        )
        device_type = model.action_vectors.device.type
        if (
            model.training
            or not torch.is_inference_mode_enabled()
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled(device_type)
            or any(module.training for module in modules)
        ):
            self._key = None
            self._value = None
            self._dependencies = ()
            return self._compute(model)

        tensors = (
            model.action_vectors,
            *model.action_embedding.parameters(),
            *model.action_vector_encoder.parameters(),
        )
        try:
            key = (
                model.action_count,
                torch.get_num_threads(),
                *(id(module) for module in modules),
                *((id(tensor), tensor._version, tensor.device, tensor.dtype)
                  for tensor in tensors),
            )
        except RuntimeError:
            # Parameters constructed inside inference_mode have no version
            # counter, so their future edits cannot be tracked reliably.
            self._key = None
            self._value = None
            self._dependencies = ()
            return self._compute(model)
        if key != self._key or self._value is None:
            self._value = self._compute(model)
            self._key = key
            # Keep old objects alive until refresh so a replaced parameter or
            # module cannot recycle a cached identity with the same version.
            self._dependencies = (*modules, *tensors)
        return self._value


def enable_window_inference(model: ActionQueryPolicy) -> None:
    """Enable a private non-module cache without changing checkpoint schemas."""

    model._window_action_query_cache = _ActionQueryCache()
    from .window_geometry_kernel import warmup

    if model.action_vectors.device.type == 'cpu':
        warmup()
    model._window_clearance = window_clearance


def window_clearance(belief, actions, horizons, *, bullet_speed, collision_radius):
    """Return exact CPU float32 clearance or defer to the shared Torch path."""
    import numpy as np
    from .window_geometry_kernel import clearance_kernel

    tensors=(belief.relative_position,belief.measured_bullet_velocity,actions,horizons)
    if (clearance_kernel is None or not torch.is_inference_mode_enabled()
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cpu')
            or any(t.device.type!='cpu' or t.dtype!=torch.float32 or not t.is_contiguous() for t in tensors)
            or belief.masks.device.type!='cpu' or belief.masks.dtype!=torch.bool
            or not belief.masks.is_contiguous()):
        return None
    result=clearance_kernel(belief.relative_position.numpy(),belief.measured_bullet_velocity.numpy(),
                            belief.masks.numpy(),actions.numpy(),horizons.numpy(),
                            np.float32(bullet_speed),np.float32(collision_radius))
    return torch.from_numpy(result)
