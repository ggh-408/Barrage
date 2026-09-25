"""Persistent-track features and action-conditioned barrage policy."""

from __future__ import annotations

from .timing import DECISION_DT

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .action_selector import ActionSelection, UnifiedActionSelector
from .runtime_core import ACTION_VECTORS
from .image_oracle import PersistentImageTracker
from .task_spec import TARGET_TASK, TARGET_TRACKING_CAPACITY
from .action_geometry import (
    build_image_geometry_belief,
    constant_action_clearance_by_object,
)


@dataclass(frozen=True)
class TrackedPolicySpec:
    max_objects: int = TARGET_TRACKING_CAPACITY
    object_features: int = 16
    global_features: int = 16
    tracker_capacity: int = TARGET_TRACKING_CAPACITY
    expected_bullet_count: int = TARGET_TASK.bullet_count
    source_size: float = 820.0
    bullet_speed: float = 240.0
    collision_radius: float = 11.5


def continuation_plan_context(plans: Sequence[dict | None]) -> np.ndarray:
    """Encode internal image-derived route memory, never simulator state."""
    context = np.zeros((len(plans), 11), np.float32)
    for row, plan in enumerate(plans):
        if plan is not None and len(plan.get("path", ())):
            context[row, 0] = 1.
            context[row, 1] = min(float(plan.get("remaining", len(plan["path"]))) / 18., 1.)
            context[row, 2 + int(plan["path"][0])] = 1.
    return context


class TrackedFeatureExtractor:
    """Stateful image-only feature boundary for one logical episode."""

    def __init__(
        self,
        spec: TrackedPolicySpec = TrackedPolicySpec(),
        *,
        refit_known_velocity: bool = True,
        decision_dt: float = DECISION_DT,
        tracker_class=PersistentImageTracker,
    ) -> None:
        self.spec = spec
        self.tracker = tracker_class(
            decision_dt=decision_dt,
            source_size=spec.source_size,
            bullet_speed=spec.bullet_speed,
            target_track_count=max(spec.tracker_capacity, spec.max_objects),
            expected_bullet_count=spec.expected_bullet_count,
            refit_known_velocity=refit_known_velocity,
        )
        self.initialized = False

    def reset(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.tracker.initialize(observation)
        self.initialized = True
        return self._features()

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.initialized:
            return self.reset(observation)
        self.tracker.update(observation)
        return self._features()

    def reset_detections(
        self,
        bullet_positions: np.ndarray,
        plane_position: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset from full-resolution image detections in normalized x/y units."""
        self.tracker.initialize_detections(
            bullet_positions, plane_position, normalized=True
        )
        self.initialized = True
        return self._features()

    def step_detections(
        self,
        bullet_positions: np.ndarray,
        plane_position: np.ndarray | None,
        *,
        decision_steps: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance from image detections without a lossy semantic re-rasterization."""
        if not self.initialized:
            return self.reset_detections(bullet_positions, plane_position)
        self.tracker.update_detections(
            bullet_positions,
            plane_position,
            normalized=True,
            decision_steps=decision_steps,
        )
        return self._features()

    def _features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spec = self.spec
        tracks = self.tracker.tracks
        objects = np.zeros(
            (spec.max_objects, spec.object_features), dtype=np.float32
        )
        mask = np.zeros(spec.max_objects, dtype=np.bool_)
        plane = self.tracker.plane_position.astype(np.float32)
        plane_velocity = self.tracker.plane_velocity.astype(np.float32)
        if tracks:
            position = self.tracker._track_vectors([track.position for track in tracks], dtype=np.float32)
            velocity = self.tracker._track_vectors([track.velocity for track in tracks], dtype=np.float32)
            relative = position - plane[None, :]
            relative_velocity = velocity - plane_velocity[None, :]
            distance = np.linalg.norm(relative, axis=1)
            speed_squared = np.square(relative_velocity).sum(axis=1)
            closing_projection = -np.sum(relative * relative_velocity, axis=1)
            ttc = np.clip(
                closing_projection / np.maximum(speed_squared, 1e-6),
                0.0,
                5.0,
            )
            closest_vector = relative + relative_velocity * ttc[:, None]
            clearance = np.linalg.norm(closest_vector, axis=1) - spec.collision_radius
            closing = closing_projection / np.maximum(
                distance * spec.bullet_speed, 1e-6
            )
            confidence = np.asarray([track.confidence for track in tracks], np.float32)
            missed = np.asarray([track.missed for track in tracks], np.float32)
            age = np.asarray([track.age for track in tracks], np.float32)
            uncertainty = np.asarray(
                [track.position_uncertainty for track in tracks], np.float32
            )
            occluded = np.asarray(
                [track.occluded_steps for track in tracks], np.float32
            )
            group_size = np.asarray(
                [track.association_group_size for track in tracks], np.float32
            )
            velocity_known = np.asarray(
                [track.velocity_known for track in tracks], np.float32
            )
            threat = (
                np.exp(-np.maximum(clearance, 0.0) / 34.0)
                * np.exp(-ttc / 1.5)
                * confidence
            )
            order = np.lexsort((distance, -threat))[: spec.max_objects]
            count = len(order)
            columns = np.column_stack((
                relative[order, 0] / spec.source_size,
                relative[order, 1] / spec.source_size,
                relative_velocity[order, 0] / spec.bullet_speed,
                relative_velocity[order, 1] / spec.bullet_speed,
                distance[order] / spec.source_size,
                np.clip(closing[order], -2.0, 2.0),
                ttc[order] / 5.0,
                np.clip(clearance[order] / 100.0, -1.0, 8.0),
                confidence[order],
                missed[order] / max(self.tracker.maximum_missed, 1),
                # Divide by the observation rate to preserve the original
                # age / 30.0 float32 projection at the 120 Hz timebase.
                np.minimum(age[order] / (1.0 / self.tracker.decision_dt), 1.0),
                threat[order],
                np.clip(uncertainty[order] / 96.0, 0.0, 1.0),
                np.clip(
                    occluded[order] / max(self.tracker.maximum_missed, 1),
                    0.0,
                    1.0,
                ),
                np.clip(group_size[order] / 8.0, 0.0, 1.0),
                velocity_known[order],
            )).astype(np.float32)
            width = min(spec.object_features, columns.shape[1])
            objects[:count, :width] = columns[:, :width]
            mask[:count] = True

        half_plane = np.asarray([9.0, 9.0], np.float32)
        walls = np.asarray((
            plane[0] - half_plane[0],
            spec.source_size - half_plane[0] - plane[0],
            plane[1] - half_plane[1],
            spec.source_size - half_plane[1] - plane[1],
        ), np.float32) / spec.source_size
        track_count = len(tracks)
        expected = max(self.tracker.expected_bullet_count, 1)
        ambiguous_fraction = (
            self.tracker.last_ambiguous_track_count / max(track_count, 1)
        )
        if tracks:
            near_threat_fraction = float(np.mean(
                (clearance < 80.0) & (ttc < 1.5)
            ))
            mean_uncertainty = float(np.mean(uncertainty) / 96.0)
        else:
            near_threat_fraction = 0.0
            mean_uncertainty = 0.0
        full_globals = np.asarray((
            plane[0] / spec.source_size,
            plane[1] / spec.source_size,
            walls[0], walls[1], walls[2], walls[3],
            plane_velocity[0] / spec.bullet_speed,
            plane_velocity[1] / spec.bullet_speed,
            self.tracker.known_velocity_fraction,
            min(track_count / expected, 1.5),
            min(self.tracker.last_detection_count / expected, 1.5),
            self.tracker.count_deficit_fraction,
            self.tracker.occluded_fraction,
            min(ambiguous_fraction, 1.0),
            near_threat_fraction,
            min(mean_uncertainty, 1.0),
        ), np.float32)
        globals_ = np.zeros(spec.global_features, dtype=np.float32)
        width = min(spec.global_features, len(full_globals))
        globals_[:width] = full_globals[:width]
        return objects, mask, globals_


def _init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if layer.weight.is_meta:
        return layer
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return layer


class _ActionCrossAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            _init(nn.Linear(width, 2 * width), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(2 * width, width)),
        )
        self.norm2 = nn.LayerNorm(width)

    def forward(
        self, queries: torch.Tensor, objects: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            queries, objects, objects, key_padding_mask=~mask, need_weights=False
        )
        queries = self.norm1(queries + attended)
        return self.norm2(queries + self.feed_forward(queries))


@dataclass(frozen=True)
class SharedActionGeometry:
    """Geometry decoded once and reused by model and arbitration."""

    masks: torch.Tensor
    clearance_by_object: torch.Tensor
    normalized_minimum_clearance: torch.Tensor


@dataclass(frozen=True)
class ActionQueryLatents:
    """Shared image-policy latents consumed by the recurrent student."""

    object_tokens: torch.Tensor
    action_queries: torch.Tensor


class ActionQueryPolicy(nn.Module):
    """Nine action queries retain counterfactual geometry until prediction."""

    model_version = 12

    def __init__(
        self,
        spec: TrackedPolicySpec = TrackedPolicySpec(),
        *,
        action_count: int = 9,
        width: int = 192,
        attention_layers: int = 2,
        attention_heads: int = 4,
        safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
        geometry_statistics: Sequence[str] = (
            "minimum",
            "danger_mass",
            "blocked_fraction",
        ),
        continuation_horizons: Sequence[float] = (),
        continuation_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if width % attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        self.spec = spec
        self.action_count = int(action_count)
        self.width = int(width)
        self.attention_layers = int(attention_layers)
        self.attention_heads = int(attention_heads)
        self.safety_horizons = tuple(float(value) for value in safety_horizons)
        self.geometry_statistics = tuple(str(value) for value in geometry_statistics)
        self.continuation_horizons = tuple(float(x) for x in continuation_horizons)
        self.continuation_weight = float(continuation_weight)
        if self.continuation_weight < 0 or any(x <= 0 for x in self.continuation_horizons):
            raise ValueError("invalid continuation configuration")
        supported_statistics = {"minimum", "danger_mass", "blocked_fraction"}
        if not self.geometry_statistics or not set(
            self.geometry_statistics
        ).issubset(supported_statistics):
            raise ValueError(
                "unsupported or empty action geometry statistics: "
                f"{self.geometry_statistics!r}"
            )
        self.object_encoder = nn.Sequential(
            _init(nn.Linear(spec.object_features, width), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width, width), 2 ** 0.5),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            _init(nn.Linear(spec.global_features, width), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width, width)),
        )
        self.action_embedding = nn.Embedding(action_count, width)
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        self.action_vector_encoder = nn.Sequential(
            _init(nn.Linear(2, width), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width, width)),
        )
        self.geometry_encoder = nn.Sequential(
            _init(nn.Linear(
                len(self.safety_horizons) * len(self.geometry_statistics),
                width,
            ), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width, width)),
        )
        self.register_buffer(
            "action_vectors",
            torch.as_tensor(ACTION_VECTORS, dtype=torch.float32),
        )
        self.cross_attention = nn.ModuleList([
            _ActionCrossAttention(width, attention_heads)
            for _ in range(attention_layers)
        ])
        self.policy_head = _init(nn.Linear(width, 1), 0.01)
        self.teacher_cost_head = _init(nn.Linear(width, 1), 0.01)
        self.continuation_head = (nn.Sequential(nn.Linear(width + 11, 64), nn.GELU(),
            nn.Linear(64, len(self.continuation_horizons))) if self.continuation_horizons else None)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Discard only the retired output layer; validate all active weights."""
        state = {name: value for name, value in state_dict.items()
                 if name not in {"collision_head.weight", "collision_head.bias"}}
        return super().load_state_dict(state, strict=strict, assign=assign)

    def _clearance_by_object(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        """Return clearance for every image-derived object/action/horizon."""
        return self.prepare_action_geometry(objects, mask, globals_).clearance_by_object

    def prepare_action_geometry(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> SharedActionGeometry:
        """Decode and calculate the geometry shared by all action consumers."""

        belief = build_image_geometry_belief(
            self.spec,
            objects,
            mask,
            globals_,
        )
        horizons = torch.as_tensor(
            self.safety_horizons, device=objects.device, dtype=objects.dtype
        )
        window_clearance = getattr(self, '_window_clearance', None)
        clearance = None
        if window_clearance is not None and not self.training:
            clearance = window_clearance(
                belief, self.action_vectors, horizons,
                bullet_speed=self.spec.bullet_speed,
                collision_radius=self.spec.collision_radius,
            )
        if clearance is None:
            clearance = constant_action_clearance_by_object(
                belief, self.action_vectors, horizons,
                bullet_speed=self.spec.bullet_speed,
                collision_radius=self.spec.collision_radius,
            )
        minimum = clearance.amin(dim=2)
        empty = ~mask.any(dim=1)
        normalized_minimum = torch.where(
            empty[:, None, None], torch.zeros_like(minimum), minimum
        )
        normalized_minimum = torch.clamp(
            normalized_minimum / 100.0, -1.0, 8.0
        )
        return SharedActionGeometry(
            masks=mask,
            clearance_by_object=clearance,
            normalized_minimum_clearance=normalized_minimum,
        )

    def action_geometry(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        """Return image-derived minimum clearance for every action/horizon."""
        return self.prepare_action_geometry(
            objects, mask, globals_
        ).normalized_minimum_clearance

    def action_geometry_from_shared(
        self, geometry: SharedActionGeometry
    ) -> torch.Tensor:
        """Return minimum clearance without recomputing tracked geometry."""

        return geometry.normalized_minimum_clearance

    def action_geometry_features(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        """Encode nearest danger and simultaneous route blockage per action."""
        geometry = self.prepare_action_geometry(objects, mask, globals_)
        return self.action_geometry_features_from_shared(geometry)

    def action_geometry_features_from_shared(
        self, geometry: SharedActionGeometry
    ) -> torch.Tensor:
        """Encode model geometry features from the shared calculation."""

        clearance = geometry.clearance_by_object
        mask = geometry.masks
        features: list[torch.Tensor] = []
        for statistic in self.geometry_statistics:
            if statistic == "minimum":
                value = geometry.normalized_minimum_clearance
            elif statistic == "danger_mass":
                danger = torch.sigmoid(-clearance / 28.0)
                danger = danger.masked_fill(
                    ~mask[:, None, :, None], 0.0
                ).sum(dim=2)
                normalizer = np.log1p(max(self.spec.expected_bullet_count, 1))
                value = torch.log1p(danger) / float(normalizer)
            else:
                blocked = (
                    (clearance < 40.0)
                    & mask[:, None, :, None]
                ).sum(dim=2)
                value = blocked.to(clearance.dtype) / float(
                    max(self.spec.expected_bullet_count, 1)
                )
            features.append(value)
        return torch.cat(features, dim=-1)

    def forward(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, teacher_cost, collision, _ = self.forward_with_geometry(
            objects,
            mask,
            globals_,
        )
        return policy, teacher_cost, collision

    def forward_policy(self, objects, mask, globals_):
        """Window logits without compatibility outputs or unused latent wrappers."""
        return self.forward_with_arbiter_latents(objects, mask, globals_, policy_only=True)

    def forward_with_geometry(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
        *,
        compute_teacher_cost: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        SharedActionGeometry,
    ]:
        """Run the model and return the exact geometry used by its encoder."""

        policy, teacher_cost, collision, geometry, _ = (
            self.forward_with_arbiter_latents(
                objects,
                mask,
                globals_,
                compute_teacher_cost=compute_teacher_cost,
            )
        )
        return policy, teacher_cost, collision, geometry

    def forward_with_arbiter_latents(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
        *,
        compute_teacher_cost: bool = True,
        policy_only: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        SharedActionGeometry,
        ActionQueryLatents,
    ]:
        """Run one backbone pass and expose its object/action representations."""

        if objects.ndim != 3:
            raise ValueError("objects must have shape [batch, objects, features]")
        geometry = self.prepare_action_geometry(
            objects,
            mask.bool(),
            globals_,
        )
        safe_mask = mask.bool().clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[:, 0] |= empty
        encoded_objects = self.object_encoder(objects) * mask.unsqueeze(-1)
        batch = len(objects)
        query_cache = getattr(self, "_window_action_query_cache", None)
        if query_cache is None:
            action_ids = torch.arange(self.action_count, device=objects.device)
            action_query_base = (
                self.action_embedding(action_ids)[None, :, :]
                + self.action_vector_encoder(self.action_vectors)[None, :, :]
            )
        else:
            action_query_base = query_cache.get(self)
        queries = (
            action_query_base
            + self.global_encoder(globals_)[:, None, :]
            + self.geometry_encoder(
                self.action_geometry_features_from_shared(geometry)
            )
        ).expand(batch, -1, -1)
        for block in self.cross_attention:
            queries = block(queries, encoded_objects, safe_mask)
        policy = self.policy_head(queries).squeeze(-1)
        if self.continuation_head is not None and self.continuation_weight > 0:
            context = getattr(self, '_image_continuation_context', None)
            if context is None:
                context = queries.new_zeros((len(queries), 11))
            context = context[:, None, :].expand(-1, self.action_count, -1)
            value = self.continuation_head(torch.cat((queries, context), dim=-1)).sigmoid().mean(dim=-1)
            policy = policy + self.continuation_weight * (value - value.mean(dim=1, keepdim=True))
        if policy_only:
            return policy
        teacher_cost = (self.teacher_cost_head(queries).squeeze(-1)
                        if compute_teacher_cost or self.training or torch.is_grad_enabled()
                        else torch.zeros_like(policy))
        # Retain the legacy tuple layout for old replay/student adapters.
        # This constant has no parameters, gradient, calibration or decision role.
        collision = policy.new_full(
            (len(policy), len(self.safety_horizons), self.action_count), -1000.0
        )
        return policy, teacher_cost, collision, geometry, ActionQueryLatents(
            object_tokens=encoded_objects,
            action_queries=queries,
        )


class TrackedPolicyAgent:
    """Centralized batched inference over image-derived tracked features."""

    def __init__(
        self,
        model: ActionQueryPolicy,
        device: torch.device,
        *,
        analytic_shield: bool = False,
        analytic_guard_horizon_seconds: float = 0.10,
        analytic_clearance_margin: float = 0.0,
        analytic_shield_gate: str = "always",
        teacher_cost_ranking_weight: float = 0.0,
        action_hysteresis_bonus: float = 0.0,
        **legacy_options,
    ) -> None:
        retired = {"use_safety_filter", "safety_threshold", "analytic_min_tracked_count_ratio",
                   "analytic_max_model_risk_increase", "analytic_max_selected_violation",
                   "long_horizon_risk_weight"}
        if set(legacy_options) - retired:
            raise TypeError(f"Unknown policy options: {set(legacy_options) - retired}")
        self.model = model
        self.device = device
        self.use_safety_filter = False  # Legacy diagnostic attribute, never enabled.
        self.safety_threshold = 0.5  # Compatibility for neutral-risk pixel adapters.
        self.analytic_shield = bool(analytic_shield and analytic_shield_gate == "always")
        self.analytic_guard_horizon_seconds = float(
            analytic_guard_horizon_seconds
        )
        self.analytic_clearance_margin = float(analytic_clearance_margin)
        self.analytic_shield_gate = "always"
        self.teacher_cost_ranking_weight = float(teacher_cost_ranking_weight)
        self.action_hysteresis_bonus = float(action_hysteresis_bonus)
        if self.action_hysteresis_bonus < 0.0:
            raise ValueError("action_hysteresis_bonus must be non-negative")
        self._previous_actions: dict[int, int] = {}
        self._analytic_horizon_index: int | None = None
        if self.analytic_shield:
            horizons = np.asarray(self.model.safety_horizons, dtype=np.float64)
            matches = np.flatnonzero(
                np.isclose(
                    horizons,
                    self.analytic_guard_horizon_seconds,
                    rtol=0.0,
                    atol=1e-9,
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    "analytic shield horizon must exactly match one model horizon: "
                    f"requested={self.analytic_guard_horizon_seconds} "
                    f"available={tuple(float(value) for value in horizons)}"
                )
            self._analytic_horizon_index = int(matches[0])
        self._action_selector = UnifiedActionSelector(
            analytic_shield=self.analytic_shield,
            analytic_horizon_index=self._analytic_horizon_index,
            analytic_clearance_margin=self.analytic_clearance_margin,
            teacher_cost_ranking_weight=self.teacher_cost_ranking_weight,
        )
        self.filtered_action_count = 0
        self.all_unsafe_count = 0
        self.overridden_decision_count = 0
        self.analytic_gate_decision_count = 0
        self.analytic_conflict_veto_count = 0
        self.decision_count = 0

    @property
    def action_selector_mode(self) -> str:
        return "unified"

    @property
    def safety_filter_mode(self) -> str:
        return "analytic_geometry" if self.analytic_shield else "none"

    def _select_actions(
        self,
        objects: torch.Tensor,
        masks: torch.Tensor,
        globals_: torch.Tensor,
        *,
        deterministic: bool,
    ) -> ActionSelection:
        if hasattr(self.model, "forward_with_geometry"):
            options = {}
            if (getattr(self, "_window_skip_unused_teacher_cost", False)
                    and isinstance(self.model, ActionQueryPolicy)
                    and self._action_selector.teacher_cost_ranking_weight == 0):
                options["compute_teacher_cost"] = False
            policy, teacher_cost, collision, geometry = (
                self.model.forward_with_geometry(
                    objects,
                    masks,
                    globals_,
                    **options,
                )
            )
        else:
            # Compatibility boundary for small external/test policy stubs.  Real
            # ActionQueryPolicy checkpoints always use the shared fast path.
            policy, teacher_cost, collision = self.model(
                objects,
                masks,
                globals_,
            )
            if self.analytic_shield:
                normalized_minimum = self.model.action_geometry(
                    objects, masks, globals_
                )
            else:
                horizon_count = collision.shape[1]
                normalized_minimum = torch.zeros(
                    len(policy),
                    policy.shape[1],
                    horizon_count,
                    device=policy.device,
                    dtype=policy.dtype,
                )
            geometry = SharedActionGeometry(
                masks=masks,
                clearance_by_object=torch.empty(
                    0, device=objects.device, dtype=objects.dtype
                ),
                normalized_minimum_clearance=normalized_minimum,
            )
        return self._action_selector.select(
            policy,
            teacher_cost,
            collision,
            masks,
            globals_,
            geometry,
            deterministic=deterministic,
        )

    def reset_state(self, episode_indices: np.ndarray | None = None) -> None:
        """Reset optional controller state at an episode boundary."""
        if episode_indices is None:
            self._previous_actions.clear()
            return
        for episode_index in np.asarray(episode_indices).reshape(-1):
            self._previous_actions.pop(int(episode_index), None)

    def _apply_action_hysteresis(
        self,
        selection: ActionSelection,
        episode_indices: np.ndarray | None,
    ) -> ActionSelection:
        if self.action_hysteresis_bonus <= 0.0:
            return selection
        if episode_indices is None:
            indices = np.arange(len(selection.actions), dtype=np.int64)
        else:
            indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        if len(indices) != len(selection.actions):
            raise ValueError("episode_indices must match the action batch")
        scores = selection.scores.clone()
        for row, episode_index in enumerate(indices):
            previous = self._previous_actions.get(int(episode_index))
            if previous is not None and torch.isfinite(scores[row, previous]):
                scores[row, previous] += self.action_hysteresis_bonus
        actions = scores.argmax(dim=1)
        for episode_index, action in zip(indices, actions.detach().cpu().tolist()):
            self._previous_actions[int(episode_index)] = int(action)
        counters = selection.counter_values.clone()
        counters[4] = (actions != selection.raw_actions).sum().to(torch.int64)
        return replace(
            selection,
            actions=actions,
            scores=scores,
            counter_values=counters,
        )

    def _export_selection(
        self,
        selection: ActionSelection,
        *,
        extra_counters: dict[str, torch.Tensor] | None = None,
    ) -> np.ndarray:
        """Synchronize actions and all production counters in one transfer."""

        parts = (selection.actions.to(torch.int64), selection.counter_values)
        if extra_counters:
            parts += tuple(value.reshape(1) for value in extra_counters.values())
        payload = torch.cat(parts).cpu().numpy()
        action_count = len(selection.actions)
        counter_count = len(selection.counter_values)
        counts = payload[action_count:action_count + counter_count]
        self.filtered_action_count += int(counts[0])
        self.all_unsafe_count += int(counts[1])
        self.analytic_gate_decision_count += int(counts[2])
        self.analytic_conflict_veto_count += int(counts[3])
        self.overridden_decision_count += int(counts[4])
        self.decision_count += action_count
        if extra_counters:
            extra_values = payload[action_count + counter_count:]
            for name, value in zip(extra_counters, extra_values):
                setattr(self, name, getattr(self, name) + int(value))
        return payload[:action_count]
    @torch.inference_mode()
    def act_features(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        deterministic: bool = True,
        episode_indices: np.ndarray | None = None,
        **_: object,
    ) -> np.ndarray:
        object_tensor = torch.as_tensor(
            objects, device=self.device, dtype=torch.float32
        )
        mask_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
        global_tensor = torch.as_tensor(
            globals_, device=self.device, dtype=torch.float32
        )
        selection = self._select_actions(
            object_tensor,
            mask_tensor,
            global_tensor,
            deterministic=deterministic,
        )
        selection = self._apply_action_hysteresis(selection, episode_indices)
        return self._export_selection(selection)

    @torch.inference_mode()
    def act_features_with_diagnostics(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        episode_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Choose actions and expose model-only safety data for attribution."""
        object_tensor = torch.as_tensor(
            objects, device=self.device, dtype=torch.float32
        )
        mask_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
        global_tensor = torch.as_tensor(
            globals_, device=self.device, dtype=torch.float32
        )
        selection = self._select_actions(
            object_tensor,
            mask_tensor,
            global_tensor,
            deterministic=True,
        )
        selection = self._apply_action_hysteresis(selection, episode_indices)
        actions = selection.actions
        raw_actions = selection.raw_actions
        learned_actions = selection.learned_actions
        immediate_risk = selection.immediate_risk
        collision_risk = selection.collision_risk
        teacher_cost = selection.teacher_cost
        analytic_clearance = selection.analytic_clearance
        row = torch.arange(len(actions), device=actions.device)
        action_output = self._export_selection(selection)
        diagnostics = {
            "raw_policy_actions": raw_actions.cpu().numpy(),
            "learned_filter_actions": learned_actions.cpu().numpy(),
            "raw_immediate_risk": immediate_risk[row, raw_actions].cpu().numpy(),
            "selected_immediate_risk": immediate_risk[row, actions].cpu().numpy(),
            "action_was_filtered": (actions != raw_actions).cpu().numpy(),
            "all_actions_unsafe": selection.all_unsafe.cpu().numpy(),
            "learned_filter_max_risk": collision_risk.amax(dim=1)[
                row, learned_actions
            ].cpu().numpy(),
            "selected_max_risk": collision_risk.amax(dim=1)[
                row, actions
            ].cpu().numpy(),
            "learned_filter_teacher_cost": teacher_cost[
                row, learned_actions
            ].cpu().numpy(),
            "selected_teacher_cost": teacher_cost[row, actions].cpu().numpy(),
        }
        if analytic_clearance is not None:
            diagnostics.update({
                "raw_analytic_clearance": analytic_clearance[
                    row, raw_actions
                ].cpu().numpy(),
                "selected_analytic_clearance": analytic_clearance[
                    row, actions
                ].cpu().numpy(),
                "learned_analytic_clearance": analytic_clearance[
                    row, learned_actions
                ].cpu().numpy(),
            })
        return action_output, diagnostics
