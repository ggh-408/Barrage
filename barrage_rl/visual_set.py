"""Screen-only object extraction and permutation-invariant barrage policy.

The public policy boundary accepts only stacked uint8 frames.  Simulator state
is deliberately absent from this module; privileged state is used only by the
training script to obtain teacher labels.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class VisualSetSpec:
    max_objects: int = 64
    object_features: int = 12
    global_features: int = 8
    plane_intensity: int = 96
    bullet_intensity: int = 255
    bullet_size_intensity_base: int = 160
    source_screen_size: int = 820


def _component_centers(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact white core written once for every visible bullet."""
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return np.empty((0, 2), np.float32), np.empty(0, np.float32)
    yx = np.asarray(coordinates, dtype=np.float32)
    centers = yx[:, ::-1]
    radii = np.zeros(len(centers), dtype=np.float32)
    return centers, radii


class SemanticFrameExtractor:
    """Convert semantic screen pixels to an unordered set of moving objects.

    The current environment renders plane and bullet masks at distinct
    intensities.  A real-screen adapter only needs to map detected sprites to
    the same semantic values; the policy itself remains unchanged.
    """

    def __init__(self, spec: VisualSetSpec = VisualSetSpec()) -> None:
        self.spec = spec

    def _plane_center(self, frame: np.ndarray) -> np.ndarray:
        points = np.argwhere(frame == self.spec.plane_intensity)
        if len(points) == 0:
            return np.asarray([0.5, 0.5], np.float32)
        y, x = points.mean(axis=0)
        height, width = frame.shape
        return np.asarray([x / max(width - 1, 1), y / max(height - 1, 1)], np.float32)

    def _bullet_size(self, frame: np.ndarray) -> float:
        lower = self.spec.bullet_size_intensity_base + 1
        upper = self.spec.bullet_size_intensity_base + 10
        encoded = frame[(frame >= lower) & (frame <= upper)]
        if len(encoded) == 0:
            raise ValueError("semantic frame does not contain bullet-size halos")
        return float(
            np.clip(
                np.median(encoded) - self.spec.bullet_size_intensity_base,
                1.0,
                10.0,
            )
        )

    def _cap_detections(
        self, centers: np.ndarray, radii: np.ndarray, plane_px: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bound matching work when overlapping 3x3 marks form a dense blob."""
        limit = self.spec.max_objects * 2
        if len(centers) <= limit:
            return centers, radii
        distance = np.linalg.norm(centers - plane_px[None, :], axis=1)
        selected = np.argsort(distance)[:limit]
        return centers[selected], radii[selected]

    @staticmethod
    def _match_velocity(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        if len(current) == 0 or len(previous) == 0:
            return np.zeros_like(current)
        squared = ((current[:, None, :] - previous[None, :, :]) ** 2).sum(axis=2)
        matches = squared.argmin(axis=1)
        velocity = current - previous[matches]
        # New/occluded objects can be matched across the whole screen.  Treat
        # those impossible jumps as unknown velocity rather than misinformation.
        velocity[np.sqrt(squared[np.arange(len(current)), matches]) > 0.12] = 0.0
        return velocity

    @staticmethod
    def _one_to_one_matches(
        current: np.ndarray, previous: np.ndarray, maximum_distance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Greedily match detections without assigning one blob twice."""
        if len(current) == 0 or len(previous) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        squared = ((current[:, None, :] - previous[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(squared, axis=None)
        used_current = np.zeros(len(current), dtype=np.bool_)
        used_previous = np.zeros(len(previous), dtype=np.bool_)
        current_indices = []
        previous_indices = []
        limit = maximum_distance * maximum_distance
        previous_count = squared.shape[1]
        for flat_index in order:
            current_index, previous_index = divmod(int(flat_index), previous_count)
            if squared[current_index, previous_index] > limit:
                break
            if used_current[current_index] or used_previous[previous_index]:
                continue
            used_current[current_index] = True
            used_previous[previous_index] = True
            current_indices.append(current_index)
            previous_indices.append(previous_index)
        return np.asarray(current_indices, np.int64), np.asarray(previous_indices, np.int64)

    def _temporal_velocity(
        self, current: np.ndarray, history: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        history_count = len(history)
        estimates = np.empty(
            (len(current), history_count, 2), dtype=np.float32
        )
        estimate_counts = np.zeros(len(current), dtype=np.int64)
        for lag, previous in enumerate(reversed(history), start=1):
            current_indices, previous_indices = self._one_to_one_matches(
                current, previous, maximum_distance=min(0.025 * lag + 0.015, 0.10)
            )
            if len(current_indices):
                slots = estimate_counts[current_indices]
                estimates[current_indices, slots] = (
                    current[current_indices] - previous[previous_indices]
                ) / float(lag)
                estimate_counts[current_indices] += 1
        velocity = np.zeros_like(current)
        confidence = np.zeros(len(current), dtype=np.float32)
        consistency = np.zeros(len(current), dtype=np.float32)
        confidence_denominator = max(history_count, 1)
        for count in range(1, history_count + 1):
            indices = np.flatnonzero(estimate_counts == count)
            if len(indices) == 0:
                continue
            stacked = estimates[indices, :count]
            grouped_velocity = np.median(stacked, axis=1)
            velocity[indices] = grouped_velocity
            confidence[indices] = count / confidence_denominator
            # Keep the original per-track reduction order.  Vectorizing this
            # norm across tracks changes float32 rounding by roughly one ULP,
            # which would make checkpoint evaluation features non-identical.
            for group_index, index in enumerate(indices):
                deviation = np.linalg.norm(
                    stacked[group_index] - velocity[index], axis=1
                ).mean()
                consistency[index] = np.exp(-40.0 * deviation)
        return velocity, confidence, consistency

    def extract(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if observation.ndim != 3 or observation.shape[0] < 2:
            raise ValueError("observation must have shape [frames, height, width]")
        frames = np.asarray(observation, dtype=np.uint8)
        height, width = frames.shape[-2:]
        current_frame = frames[-1]
        previous_frame = frames[-2]

        plane = self._plane_center(current_frame)
        plane_history = [self._plane_center(frame) for frame in frames[:-1]]
        previous_plane = plane_history[-1]
        plane_velocity = plane - previous_plane
        if len(plane_history) > 1:
            plane_velocity = (plane - plane_history[0]) / float(len(plane_history))

        current_px, radii_px = _component_centers(
            current_frame == self.spec.bullet_intensity
        )
        scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], np.float32)
        current_px, radii_px = self._cap_detections(
            current_px, radii_px, plane * scale
        )
        current = current_px / scale if len(current_px) else current_px
        history = []
        for frame in frames[:-1]:
            previous_px, previous_radii = _component_centers(
                frame == self.spec.bullet_intensity
            )
            previous_px, _ = self._cap_detections(
                previous_px, previous_radii, plane * scale
            )
            history.append(previous_px / scale if len(previous_px) else previous_px)
        if self.spec.object_features >= 11:
            bullet_velocity, track_confidence, velocity_consistency = (
                self._temporal_velocity(current, tuple(history))
            )
        else:
            previous = history[-1]
            bullet_velocity = self._match_velocity(current, previous)
            track_confidence = np.ones(len(current), dtype=np.float32)
            velocity_consistency = np.ones(len(current), dtype=np.float32)
        relative_velocity = bullet_velocity - plane_velocity[None, :]
        relative_position = current - plane[None, :]
        distance = np.linalg.norm(relative_position, axis=1)
        speed_squared = np.sum(relative_velocity ** 2, axis=1)
        time_to_closest = np.clip(
            -np.sum(relative_position * relative_velocity, axis=1)
            / np.maximum(speed_squared, 1e-8),
            0.0,
            60.0,
        )
        closest = relative_position + relative_velocity * time_to_closest[:, None]
        bullet_size = self._bullet_size(current_frame)
        radius_value = 0.5 * bullet_size / float(self.spec.source_screen_size)
        radius = np.full(len(current), radius_value, dtype=np.float32)
        clearance = np.linalg.norm(closest, axis=1) - radius
        threat = np.exp(-np.maximum(clearance, 0.0) / 0.05) * np.exp(
            -time_to_closest / 12.0
        )

        features = np.zeros(
            (self.spec.max_objects, self.spec.object_features), dtype=np.float32
        )
        valid = np.zeros(self.spec.max_objects, dtype=np.bool_)
        if len(current):
            # Imminent trajectories matter more than merely nearby receding
            # bullets.  This also protects against duplicate detections in dense
            # overlaps when the object budget is exceeded.
            order = np.lexsort((distance, -threat))[: self.spec.max_objects]
            count = len(order)
            columns = [
                    relative_position[order, 0],
                    relative_position[order, 1],
                    relative_velocity[order, 0],
                    relative_velocity[order, 1],
                    distance[order],
                    time_to_closest[order] / 60.0,
                    clearance[order],
                    radius[order],
                    np.full(count, bullet_size / 10.0, np.float32),
                    track_confidence[order],
                    velocity_consistency[order],
                    threat[order],
                ]
            features[:count, : len(columns)] = np.column_stack(columns)
            valid[:count] = True

        global_features = np.asarray(
            [
                plane[0],
                plane[1],
                plane[0],
                1.0 - plane[0],
                plane[1],
                1.0 - plane[1],
                plane_velocity[0],
                plane_velocity[1],
            ],
            dtype=np.float32,
        )
        return features, valid, global_features

    def extract_batch(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        extracted = [self.extract(observation) for observation in observations]
        objects, masks, globals_ = zip(*extracted)
        return np.stack(objects), np.stack(masks), np.stack(globals_)


def _init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return layer


def calibrate_safety_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    maximum_false_negative_rate: float = 0.01,
    maximum_all_unsafe_rate: float = 0.02,
    default: float = 0.50,
) -> Tuple[float, ...]:
    """Choose the largest per-horizon threshold meeting a target unsafe FNR."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.bool_)
    if probabilities.shape != targets.shape or probabilities.ndim != 3:
        raise ValueError("safety calibration expects matching [N,H,A] arrays")
    fraction = float(np.clip(maximum_false_negative_rate, 0.0, 1.0))
    thresholds = []
    for horizon in range(probabilities.shape[1]):
        positive = probabilities[:, horizon, :][targets[:, horizon, :]]
        if len(positive) == 0:
            thresholds.append(float(default))
            continue
        try:
            threshold = np.quantile(positive, fraction, method="lower")
        except TypeError:  # NumPy <1.22 compatibility
            threshold = np.quantile(positive, fraction, interpolation="lower")
        thresholds.append(float(np.clip(threshold, 1e-4, 1.0 - 1e-4)))
    thresholds_array = np.asarray(thresholds, dtype=np.float64)
    # Only the nearest horizon is a hard action veto.  Longer horizons are
    # auxiliary forecasts: the controller will make many more decisions before
    # they elapse, so treating them as hard vetoes makes every action appear
    # unsafe in ordinary recoverable states.
    allowed = float(np.clip(maximum_all_unsafe_rate, 0.0, 1.0))
    normalized_risk = probabilities[:, 0, :] / max(thresholds_array[0], 1e-8)
    minimum_action_risk = normalized_risk.min(axis=1)
    current_all_unsafe = float(np.mean(minimum_action_risk >= 1.0))
    if current_all_unsafe > allowed:
        scale = float(np.quantile(minimum_action_risk, 1.0 - allowed))
        thresholds_array[0] = float(np.clip(
            thresholds_array[0] * max(scale, 1.0) * (1.0 + 1e-6),
            1e-4, 0.95,
        ))
    return tuple(float(value) for value in thresholds_array)


def safety_filter_is_usable(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Sequence[float],
    maximum_false_negative_rate: float = 0.05,
    maximum_all_unsafe_rate: float = 0.025,
    minimum_positive_labels: int = 100,
) -> bool:
    """Gate deployment filtering when the risk head is not yet calibrated."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.bool_)
    threshold_array = np.asarray(tuple(thresholds), dtype=np.float64)
    if probabilities.shape != targets.shape or probabilities.ndim != 3:
        raise ValueError("safety calibration expects matching [N,H,A] arrays")
    if threshold_array.shape != (probabilities.shape[1],):
        raise ValueError("one safety threshold is required per horizon")
    # Deployment filtering uses only the imminent-collision horizon.  Validate
    # exactly that contract rather than rejecting a useful short-horizon head
    # because an auxiliary long-horizon forecast is poorly calibrated.
    prediction = probabilities[:, 0, :] >= threshold_array[0]
    immediate_targets = targets[:, 0, :]
    positive_count = int(immediate_targets.sum())
    if positive_count < int(minimum_positive_labels):
        return False
    false_negative_rate = float(
        (immediate_targets & ~prediction).sum() / positive_count
    )
    all_unsafe_rate = float(prediction.all(axis=1).mean())
    return (
        false_negative_rate <= float(maximum_false_negative_rate)
        and all_unsafe_rate <= float(maximum_all_unsafe_rate)
    )


class _SetAttentionBlock(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            _init(nn.Linear(width, width * 2), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width * 2, width), 1.0),
        )
        self.norm2 = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            values, values, values, key_padding_mask=~mask, need_weights=False
        )
        values = self.norm1(values + attended)
        values = self.norm2(values + self.feed_forward(values))
        return values * mask.unsqueeze(-1)


class VisualSetRecurrentQNetwork(nn.Module):
    """Set-attention encoder with recurrent policy, risk, and Q heads."""

    model_version = 9

    def __init__(
        self,
        spec: VisualSetSpec,
        action_count: int = 9,
        width: int = 192,
        attention_layers: int = 2,
        attention_heads: int = 4,
        safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
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
        if not self.safety_horizons or any(value <= 0 for value in self.safety_horizons):
            raise ValueError("safety_horizons must contain positive values")
        self.object_encoder = nn.Sequential(
            _init(nn.Linear(spec.object_features, width), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width, width), 2 ** 0.5),
            nn.GELU(),
        )
        self.attention_blocks = nn.ModuleList(
            [_SetAttentionBlock(width, attention_heads) for _ in range(attention_layers)]
        )
        self.trunk = nn.Sequential(
            _init(nn.Linear(width * 2 + spec.global_features, width * 2), 2 ** 0.5),
            nn.GELU(),
            _init(nn.Linear(width * 2, width), 2 ** 0.5),
            nn.GELU(),
        )
        self.memory = nn.GRU(width, width, batch_first=True)
        for name, parameter in self.memory.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        self.policy_head = _init(nn.Linear(width, action_count), 0.01)
        self.q_head = _init(nn.Linear(width, action_count), 0.01)
        self.collision_head = _init(
            nn.Linear(width, action_count * len(self.safety_horizons)), 0.01
        )

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.width, device=device)

    def encode_set(
        self, objects: torch.Tensor, mask: torch.Tensor, globals_: torch.Tensor
    ) -> torch.Tensor:
        safe_mask = mask.bool().clone()
        empty = ~safe_mask.any(dim=1)
        if torch.any(empty):
            safe_mask[empty, 0] = True
        encoded = self.object_encoder(objects) * mask.unsqueeze(-1)
        for block in self.attention_blocks:
            encoded = block(encoded, safe_mask)
        encoded = encoded * mask.unsqueeze(-1)
        valid = mask.unsqueeze(-1)
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~valid, negative).amax(dim=1)
        maximum = torch.where(
            empty[:, None], torch.zeros_like(maximum), maximum
        )
        mean = encoded.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.trunk(torch.cat((maximum, mean, globals_), dim=1))

    def forward_sequence(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if objects.ndim != 4:
            raise ValueError("objects must have shape [batch, time, objects, features]")
        batch, steps, object_count, feature_count = objects.shape
        encoded = self.encode_set(
            objects.reshape(batch * steps, object_count, feature_count),
            mask.reshape(batch * steps, object_count),
            globals_.reshape(batch * steps, self.spec.global_features),
        ).reshape(batch, steps, self.width)
        # load_state_dict(), deepcopy(), and target-network synchronization can
        # invalidate cuDNN's packed GRU storage. Re-flattening is a cheap no-op
        # when the weights are already contiguous and prevents per-call packing.
        self.memory.flatten_parameters()
        recurrent, hidden = self.memory(encoded, hidden)
        safety = self.collision_head(recurrent).reshape(
            batch, steps, len(self.safety_horizons), self.action_count
        )
        return (
            self.policy_head(recurrent),
            self.q_head(recurrent),
            safety,
            hidden,
        )

    def forward_step(
        self,
        objects: torch.Tensor,
        mask: torch.Tensor,
        globals_: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, q_values, collision, hidden = self.forward_sequence(
            objects[:, None], mask[:, None], globals_[:, None], hidden
        )
        return policy[:, 0], q_values[:, 0], collision[:, 0], hidden


class ScreenOnlyAgent:
    """Inference firewall: its only observation input is a frame tensor."""

    def __init__(
        self,
        model: VisualSetRecurrentQNetwork,
        extractor: SemanticFrameExtractor,
        device: torch.device,
        inference_head: str = "policy",
        safety_thresholds: Optional[Sequence[float]] = None,
        use_safety_filter: bool = True,
    ) -> None:
        self.model = model
        self.extractor = extractor
        self.device = device
        if inference_head not in ("policy", "q"):
            raise ValueError("inference_head must be 'policy' or 'q'")
        self.inference_head = inference_head
        if safety_thresholds is None:
            safety_thresholds = (0.5,) * len(model.safety_horizons)
        if len(tuple(safety_thresholds)) != len(model.safety_horizons):
            raise ValueError("one safety threshold is required per horizon")
        self.safety_thresholds = torch.as_tensor(
            tuple(float(value) for value in safety_thresholds),
            device=device,
            dtype=torch.float32,
        )
        self.use_safety_filter = bool(use_safety_filter)
        self.filtered_action_count = 0
        self.all_unsafe_count = 0
        self.decision_count = 0
        self.hidden: Optional[torch.Tensor] = None

    def reset(self, batch_size: int = 1) -> None:
        self.hidden = self.model.initial_state(batch_size, self.device)

    @torch.inference_mode()
    def reset_indices(self, done: np.ndarray) -> None:
        if self.hidden is None:
            return
        done_tensor = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self.hidden[:, done_tensor] = 0.0

    @torch.inference_mode()
    def act_features(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        deterministic: bool = True,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        batch_size = len(objects)
        if self.hidden is None or self.hidden.shape[1] != batch_size:
            self.reset(batch_size)
        policy, q_values, safety_logits, self.hidden = self.model.forward_step(
            torch.as_tensor(objects, device=self.device, dtype=torch.float32),
            torch.as_tensor(masks, device=self.device),
            torch.as_tensor(globals_, device=self.device, dtype=torch.float32),
            self.hidden,
        )
        scores = q_values if self.inference_head == "q" else policy
        safe = torch.ones_like(scores, dtype=torch.bool)
        aggregate_risk = torch.sigmoid(safety_logits).amax(dim=1)
        if self.use_safety_filter:
            risks = torch.sigmoid(safety_logits)
            # A long-horizon warning is not an immediate collision.  The actor
            # gets another decision every decision_dt, therefore only horizon
            # zero is allowed to hard-mask an action.
            safe = risks[:, 0, :] < self.safety_thresholds[0]
            has_safe = safe.any(dim=-1)
            self.filtered_action_count += int((~safe).sum().item())
            self.all_unsafe_count += int((~has_safe).sum().item())
            masked_scores = scores.masked_fill(~safe, -torch.inf)
            fallback = -aggregate_risk + 1e-6 * scores
            scores = torch.where(has_safe[:, None], masked_scores, fallback)
        self.decision_count += batch_size
        if deterministic:
            action = scores.argmax(dim=-1)
        else:
            action = Categorical(logits=scores).sample()
        result = action.cpu().numpy()
        if epsilon > 0.0:
            rng = np.random.default_rng() if rng is None else rng
            explore = rng.random(batch_size) < epsilon
            for index in np.flatnonzero(explore):
                candidates = np.flatnonzero(safe[index].cpu().numpy())
                if len(candidates) == 0:
                    candidates = np.asarray([int(result[index])], dtype=np.int64)
                result[index] = int(rng.choice(candidates))
        return result

    @torch.inference_mode()
    def act(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        objects, masks, globals_ = self.extractor.extract_batch(observations)
        return self.act_features(objects, masks, globals_, deterministic)
