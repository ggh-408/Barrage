"""Strict image-input tracking and model-based control diagnostics.

The controller in this module never reads the live environment state.  It
reconstructs a belief state from semantic pixels and runs the existing exact
planner on that reconstructed state.  This is an observation-ceiling probe,
not the final learned policy.
"""

from __future__ import annotations

from .timing import DECISION_DT

from collections import deque
from dataclasses import dataclass, field
from itertools import chain
from operator import itemgetter
from typing import Sequence

import numpy as np

_FLOAT32_DTYPE = np.dtype(np.float32)

from .runtime_core import OPENING_BATCH_COUNT
from .task_spec import (
    TARGET_TASK,
    TARGET_TRACKING_CAPACITY,
    tracking_capacity_for,
)


@dataclass
class BulletTrack:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    velocity_known: bool = False
    missed: int = 0
    confidence: float = 0.15
    age: int = 1
    position_uncertainty: float = 24.0
    occluded_steps: int = 0
    association_group_size: int = 1


class PersistentImageTracker:
    """Track one-pixel semantic bullet centres with persistent identities."""

    def __init__(
        self,
        *,
        source_size: float = 820.0,
        bullet_speed: float = 240.0,
        decision_dt: float = DECISION_DT,
        target_track_count: int = TARGET_TRACKING_CAPACITY,
        expected_bullet_count: int = TARGET_TASK.bullet_count,
        maximum_missed: int = 8,
        occlusion_gate: float = 7.5,
        refit_known_velocity: bool = True,
    ) -> None:
        self.source_size = float(source_size)
        self.bullet_speed = float(bullet_speed)
        self.decision_dt = float(decision_dt)
        self.target_track_count = int(target_track_count)
        self.expected_bullet_count = int(expected_bullet_count)
        self.maximum_missed = int(maximum_missed)
        self.occlusion_gate = float(occlusion_gate)
        self.refit_known_velocity = bool(refit_known_velocity)
        self.history_limit = None
        self.tracks: list[BulletTrack] = []
        self.plane_position = np.asarray([source_size / 2.0] * 2, np.float32)
        self.plane_velocity = np.zeros(2, np.float32)
        self._previous_plane: np.ndarray | None = None
        self._next_track_id = 0
        self._step = 0
        self.last_detection_count = 0
        self.last_ambiguous_track_count = 0

    def clear(self) -> None:
        self.__dict__.pop("_pending_predictions", None)
        self.tracks.clear()
        self._previous_plane = None
        self._next_track_id = 0
        self._step = 0
        self.last_detection_count = 0
        self.last_ambiguous_track_count = 0
        self.plane_position[:] = self.source_size / 2.0
        self.plane_velocity.fill(0.0)

    def initialize(self, observation: np.ndarray) -> None:
        frames = np.asarray(observation, dtype=np.uint8)
        if frames.ndim != 3 or len(frames) < 1:
            raise ValueError("observation must have shape [frames, height, width]")
        self.clear()
        # Deployment reset repeats one cold frame.  Treat it as one timestamp,
        # not as a stationary four-step trajectory.
        start = 0
        while start + 1 < len(frames) and np.array_equal(frames[start], frames[start + 1]):
            start += 1
        for frame in frames[start:]:
            self._update_frame(frame)

    def update(self, observation: np.ndarray) -> None:
        frames = np.asarray(observation, dtype=np.uint8)
        if frames.ndim != 3 or len(frames) < 1:
            raise ValueError("observation must have shape [frames, height, width]")
        if not self.tracks:
            self.initialize(frames)
        else:
            self._update_frame(frames[-1])

    def initialize_detections(
        self,
        bullet_positions: np.ndarray,
        plane_position: np.ndarray | None,
        *,
        normalized: bool = True,
    ) -> None:
        """Initialize directly from image-derived full-resolution detections."""
        self.clear()
        self._update_detections(
            bullet_positions, plane_position, normalized=normalized
        )

    def update_detections(
        self,
        bullet_positions: np.ndarray,
        plane_position: np.ndarray | None,
        *,
        normalized: bool = True,
        decision_steps: int = 1,
    ) -> None:
        """Update from detections without rasterizing and detecting a second time."""
        self._update_detections(
            bullet_positions,
            plane_position,
            normalized=normalized,
            decision_steps=decision_steps,
        )

    def _pixel_to_source(self, xy: np.ndarray, frame: np.ndarray) -> np.ndarray:
        scale = np.asarray(
            [
                self.source_size / max(frame.shape[1] - 1, 1),
                self.source_size / max(frame.shape[0] - 1, 1),
            ],
            dtype=np.float32,
        )
        return xy.astype(np.float32, copy=False) * scale

    def _detections(self, frame: np.ndarray) -> np.ndarray:
        from .env import BarrageVisionEnv
        yx = np.argwhere(frame == BarrageVisionEnv.BULLET_INTENSITY)
        if len(yx) == 0:
            return np.empty((0, 2), np.float32)
        return self._pixel_to_source(yx[:, ::-1], frame)

    def _plane(self, frame: np.ndarray) -> np.ndarray:
        from .env import BarrageVisionEnv
        yx = np.argwhere(frame == BarrageVisionEnv.PLANE_INTENSITY)
        if len(yx) == 0:
            return self.plane_position.copy()
        xy = np.asarray([[yx[:, 1].mean(), yx[:, 0].mean()]], np.float32)
        return self._pixel_to_source(xy, frame)[0]

    def _new_track(
        self, position: np.ndarray, *, observation_step: int | None = None
    ) -> BulletTrack:
        step = self._step if observation_step is None else int(observation_step)
        track = BulletTrack(
            track_id=self._next_track_id,
            position=position.copy(),
            velocity=np.zeros(2, np.float32),
            history=[(step, position.copy())],
        )
        self._next_track_id += 1
        return track

    def _fit_velocity(
        self,
        track: BulletTrack,
        *,
        regression_cache: dict[
            tuple[int, ...], tuple[np.ndarray, float]
        ] | None = None,
    ) -> None:
        # Barrage bullets have constant velocity until they respawn, at which
        # point the gated association creates a new track.  Re-fitting the same
        # eight-point regression for every known track on every frame was a
        # large dense-barrage CPU cost with no new motion information.
        if track.velocity_known and not self.refit_known_velocity:
            return
        history = track.history[-8:]
        # Two quantized centres are enough to produce a direction, but in a
        # dense edge spawn they are not enough to establish identity.
        # Waiting for four timestamps removes the noisy cold-start velocities
        # that can hijack a minimum-clearance controller.
        if len(history) < 4:
            return
        timestamps = tuple(item[0] for item in history)
        regression = (
            None if regression_cache is None else regression_cache.get(timestamps)
        )
        if regression is None:
            steps = np.asarray(timestamps, np.float32)
            centered = steps - float(steps.mean())
            denominator = float(np.dot(centered, centered))
            if regression_cache is not None:
                regression_cache[timestamps] = (centered, denominator)
        else:
            centered, denominator = regression
        positions = np.asarray([item[1] for item in history], dtype=np.float32)
        if denominator <= 0.0:
            return
        displacement_per_decision = (centered[:, None] * positions).sum(axis=0) / denominator
        estimate = displacement_per_decision / self.decision_dt
        magnitude = float(np.linalg.norm(estimate))
        if magnitude < 0.20 * self.bullet_speed:
            return
        candidate = (
            estimate * (self.bullet_speed / max(magnitude, 1e-6))
        ).astype(np.float32)
        track.velocity = candidate
        track.velocity_known = True

    @staticmethod
    def _track_vectors(values, dtype=None):
        return np.asarray(values, dtype=dtype)

    def _history_groups(self, tracks):
        groups: dict[tuple[int, ...], list[tuple[BulletTrack, list]]] = {}
        for track in tracks:
            if track.velocity_known and not self.refit_known_velocity:
                continue
            history = track.history[-8:]
            if len(history) >= 4:
                timestamps = tuple(map(itemgetter(0), history))
                groups.setdefault(timestamps, []).append((track, history))
        return groups

    def _association_arrays(self, predictions, detections):
        delta_x = predictions[:, None, 0] - detections[None, :, 0]
        delta_y = predictions[:, None, 1] - detections[None, :, 1]
        distance_squared = delta_x * delta_x + delta_y * delta_y
        gates = np.asarray([15.0 if track.velocity_known else 24.0 for track in self.tracks], np.float32)
        valid = distance_squared <= np.square(gates[:, None])
        nearest_detection = np.argmin(distance_squared, axis=1)
        nearest_track = np.argmin(distance_squared, axis=0)
        association_group_sizes = np.count_nonzero(
            distance_squared <= self.occlusion_gate * self.occlusion_gate, axis=0).tolist()
        return distance_squared, valid, nearest_detection, nearest_track, association_group_sizes

    @staticmethod
    def _assignment_pairs(mutual_pairs, pair_order, valid, detection_count):
        assignments = list(mutual_pairs)
        for flat_index in pair_order:
            track_index, detection_index = divmod(int(flat_index), detection_count)
            if not valid[track_index, detection_index]:
                break
            assignments.append((track_index, detection_index))
        return assignments

    def _fit_velocities(self, tracks: list[BulletTrack]) -> None:
        """Batch identical timestamp bases without changing the regression."""
        groups = self._history_groups(tracks)
        for timestamps, group in groups.items():
            steps = np.asarray(timestamps, np.float32)
            centered = steps - float(steps.mean())
            denominator = float(np.dot(centered, centered))
            if denominator <= 0.0:
                continue
            points = list(map(
                itemgetter(1), chain.from_iterable(history for _, history in group)
            ))
            if all(point.dtype == _FLOAT32_DTYPE for point in points):
                try:
                    # Join the native two-float buffers without NumPy's nested
                    # sequence discovery. The read-only view is only regressed.
                    positions = np.frombuffer(b''.join(points), dtype=np.float32)
                except (TypeError, BufferError, ValueError):
                    positions = np.asarray(points, dtype=np.float32)
            else:
                positions = np.asarray(points, dtype=np.float32)
            positions = positions.reshape(len(group), len(timestamps), 2)
            displacement = (centered[None, :, None] * positions).sum(axis=1) / denominator
            estimates = displacement / self.decision_dt
            magnitudes = np.sqrt((estimates * estimates).sum(axis=1))
            # Preserve the scalar path's float64 division then float32 factor.
            # Other speeds and nonfinite data retain NumPy scalar-promotion rules.
            if self.bullet_speed == 240.0 and np.isfinite(estimates).all():
                wide = magnitudes.astype(np.float64)
                valid = wide >= 0.20 * self.bullet_speed
                factors = (self.bullet_speed / np.maximum(wide[valid], 1e-6)).astype(np.float32)
                velocities = estimates[valid] * factors[:, None]
                for index, velocity in zip(np.flatnonzero(valid), velocities):
                    track = group[index][0]
                    track.velocity = velocity.copy()
                    track.velocity_known = True
                continue
            for (track, _), estimate, magnitude in zip(group, estimates, magnitudes):
                magnitude = float(magnitude)
                if magnitude < 0.20 * self.bullet_speed:
                    continue
                track.velocity = (
                    estimate * (self.bullet_speed / max(magnitude, 1e-6))
                ).astype(np.float32)
                track.velocity_known = True

    def _update_frame(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        self._update_measurements(self._detections(frame), self._plane(frame))

    def _update_detections(
        self,
        bullet_positions: np.ndarray,
        plane_position: np.ndarray | None,
        *,
        normalized: bool,
        decision_steps: int = 1,
    ) -> None:
        detections = np.asarray(bullet_positions, dtype=np.float32).reshape(-1, 2)
        if normalized:
            detections = detections * self.source_size
        if plane_position is None:
            current_plane = self.plane_position.copy()
        else:
            current_plane = np.asarray(plane_position, dtype=np.float32).reshape(2)
            if normalized:
                current_plane = current_plane * self.source_size
        self._update_measurements(
            detections, current_plane, decision_steps=decision_steps
        )

    def _retention_threat(self, track: BulletTrack) -> float:
        relative = track.position - self.plane_position
        if not track.velocity_known:
            distance = float(np.linalg.norm(relative))
            return float(np.exp(-distance / 180.0) * track.confidence)
        relative_velocity = track.velocity - self.plane_velocity
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        ttc = -np.dot(relative, relative_velocity) / max(speed_squared, 1e-6)
        # Preserve scalar clip's inclusive bounds, positive zero and NaN
        # passthrough without dispatching an array operation for every track.
        if ttc <= 0.0:
            ttc = 0.0
        elif ttc >= 3.0:
            ttc = 3.0
        else:
            ttc = float(ttc)
        closest = relative + relative_velocity * ttc
        clearance = max(float(np.linalg.norm(closest)) - 11.5, 0.0)
        return float(
            np.exp(-clearance / 42.0)
            * np.exp(-ttc / 1.5)
            * max(track.confidence, 0.15)
        )

    def _trim_tracks(self, tracks: list[BulletTrack]) -> list[BulletTrack]:
        scores = self._retention_threats(tracks)
        # Keep the original stable ordering and in-place list mutation.
        ordered = sorted(zip(tracks, scores), key=lambda item: (
            item[0].missed > 0 and item[0].occluded_steps == 0,
            -item[1], item[0].missed, -item[0].confidence,
        ))
        tracks[:] = [track for track, _ in ordered]
        return tracks[: self.target_track_count]

    def _retention_threats(self, tracks: list[BulletTrack]) -> np.ndarray:
        """Batch the scalar float32 geometry with float64 scoring intermediates."""
        if not tracks:
            return np.empty(0, np.float64)
        positions = self._track_vectors([track.position for track in tracks])
        velocities = self._track_vectors([track.velocity for track in tracks])
        confidence = np.asarray([track.confidence for track in tracks])
        if (type(self)._retention_threat is not PersistentImageTracker._retention_threat
                or positions.dtype != np.float32 or velocities.dtype != np.float32
                or self.plane_position.dtype != np.float32
                or self.plane_velocity.dtype != np.float32
                or not all(np.isfinite(a).all() for a in (
                    positions, velocities, confidence, self.plane_position,
                    self.plane_velocity))):
            return np.asarray([self._retention_threat(track) for track in tracks])
        relative = positions - self.plane_position
        relative_velocity = velocities - self.plane_velocity
        speed_squared = (relative_velocity * relative_velocity).sum(axis=1).astype(np.float64)
        dot = (relative * relative_velocity).sum(axis=1)
        ttc = -dot / np.maximum(speed_squared, 1e-6)
        ttc = np.where(ttc <= 0.0, 0.0, np.minimum(ttc, 3.0))
        closest = relative + relative_velocity * ttc.astype(np.float32)[:, None]
        clearance = np.maximum(
            np.sqrt((closest * closest).sum(axis=1)).astype(np.float64) - 11.5, 0.0)
        known = (np.exp(-clearance / 42.0) * np.exp(-ttc / 1.5)
                 * np.maximum(confidence, 0.15))
        unknown = (np.exp(-np.sqrt((relative * relative).sum(axis=1)).astype(np.float64)
                          / 180.0) * confidence)
        return np.where([track.velocity_known for track in tracks], known, unknown)

    def _trim_tracks_scalar(self, tracks: list[BulletTrack]) -> list[BulletTrack]:
        """Reference implementation retained for differential checks and fallback."""
        tracks.sort(
            key=lambda track: (
                track.missed > 0 and track.occluded_steps == 0,
                -self._retention_threat(track),
                track.missed,
                -track.confidence,
            )
        )
        return tracks[: self.target_track_count]

    def predict_positions(self, decision_steps=1):
        """Predict prior track positions once in source coordinates."""
        if not self.tracks:
            return np.empty((0, 2), np.float32)
        elapsed = self.decision_dt * max(1, int(decision_steps))
        positions = np.asarray([track.position for track in self.tracks])
        velocities = np.asarray([track.velocity for track in self.tracks])
        if all(t.position.dtype == _FLOAT32_DTYPE and t.velocity.dtype == _FLOAT32_DTYPE for t in self.tracks):
            known = np.asarray([track.velocity_known for track in self.tracks])
            displacement = np.zeros_like(positions)  # Preserve unknown-velocity +0.0.
            displacement[known] = velocities[known] * elapsed
            predictions = positions + displacement
        else:
            predictions = np.stack([
                track.position
                + (track.velocity * elapsed if track.velocity_known else 0.0)
                for track in self.tracks
            ])
        return predictions

    def _update_measurements(
        self,
        detections: np.ndarray,
        current_plane: np.ndarray,
        *,
        decision_steps: int = 1,
    ) -> None:
        decision_steps = max(1, int(decision_steps))
        elapsed = self.decision_dt * decision_steps
        observation_step = self._step + decision_steps - 1
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 2)
        current_plane = np.asarray(current_plane, dtype=np.float32).reshape(2)
        self.last_detection_count = int(len(detections))
        if self._previous_plane is None:
            self.plane_velocity.fill(0.0)
        else:
            raw_plane_velocity = (
                current_plane - self._previous_plane
            ) / elapsed
            magnitude = float(np.linalg.norm(raw_plane_velocity))
            if magnitude > 0.20 * self.bullet_speed:
                self.plane_velocity = (
                    raw_plane_velocity
                    * (self.bullet_speed / max(magnitude, 1e-6))
                ).astype(np.float32)
            else:
                self.plane_velocity.fill(0.0)
        self._previous_plane = current_plane.copy()
        self.plane_position = current_plane

        if not self.tracks:
            self.tracks = self._trim_tracks([
                self._new_track(position, observation_step=observation_step)
                for position in detections
            ])
            self._step += decision_steps
            return

        cached = self.__dict__.pop('_pending_predictions', None)
        if cached is not None and cached[:2] == (self._step, decision_steps):
            predictions = cached[2]
        else:
            predictions = self.predict_positions(decision_steps)
        occlusion_distance_squared = self.occlusion_gate * self.occlusion_gate
        if len(detections):
            (distance_squared, valid, nearest_detection, nearest_track,
             association_group_sizes) = self._association_arrays(predictions, detections)
            track_indices = np.arange(len(self.tracks))
            mutual_indices = np.flatnonzero(
                (nearest_track[nearest_detection] == track_indices)
                & valid[track_indices, nearest_detection]
            )
            mutual_pairs = list(zip(mutual_indices.tolist(), nearest_detection[mutual_indices].tolist()))
            # Only a few local pairs survive the 15/24 px gates.  Sorting the
            # complete track×detection matrix spent milliseconds ordering tens
        # of thousands of identical ``inf`` entries in dense barrages.  The
            # sparse order is equivalent for every pair the assignment loop can
            # consume; flat index is an explicit deterministic tie-breaker.
            valid_flat = np.flatnonzero(valid)
            pair_order = valid_flat[np.lexsort((
                valid_flat,
                distance_squared.reshape(-1)[valid_flat],
            ))]
        else:
            distance_squared = np.empty((len(self.tracks), 0), np.float32)
            valid = np.zeros_like(distance_squared, dtype=np.bool_)
            mutual_pairs = []
            pair_order = np.empty(0, np.int64)
        # Assignment is an ordered greedy loop: native booleans avoid creating
        # NumPy scalar objects on every duplicate-pair check.
        used_tracks = [False] * len(self.tracks)
        used_detections = [False] * len(detections)
        assignments = self._assignment_pairs(mutual_pairs, pair_order, valid, len(detections))
        # Association uses precomputed predictions; matched histories can be
        # fitted together before retention uses the updated velocities.
        matched_tracks: list[BulletTrack] = []
        for track_index, detection_index in assignments:
            if used_tracks[track_index] or used_detections[detection_index]:
                continue
            track = self.tracks[track_index]
            used_tracks[track_index] = True
            used_detections[detection_index] = True
            position = detections[detection_index]
            track.position = position.copy()
            track.history.append((observation_step, position.copy()))
            if self.history_limit is not None:
                del track.history[:-self.history_limit]
            track.missed = 0
            track.occluded_steps = 0
            confidence = track.confidence + 0.20
            track.confidence = confidence if confidence < 1.0 else 1.0
            uncertainty = track.position_uncertainty * 0.55
            track.position_uncertainty = uncertainty if uncertainty > 1.5 else 1.5
            group_size = association_group_sizes[detection_index]
            track.association_group_size = group_size if group_size > 1 else 1
            matched_tracks.append(track)

        self._fit_velocities(matched_tracks)

        retained: list[BulletTrack] = []
        ambiguous_tracks = 0
        for index, track in enumerate(self.tracks):
            track.age += 1
            if not used_tracks[index]:
                if track.velocity_known:
                    track.position = predictions[index].copy()
                overlaps_detection = bool(
                    len(detections)
                    and distance_squared[index, nearest_detection[index]]
                    <= occlusion_distance_squared
                )
                if overlaps_detection:
                    # One RGB component can represent several crossing bullets.
                    # Keep every pre-existing identity through the occlusion and
                    # expose the ambiguity to the policy instead of deleting all
                    # but one trajectory.
                    track.occluded_steps += 1
                    track.association_group_size = max(
                        2,
                        int(association_group_sizes[nearest_detection[index]]),
                    )
                    track.confidence *= 0.94
                    ambiguous_tracks += 1
                else:
                    track.occluded_steps = 0
                    track.association_group_size = 1
                    track.missed += 1
                    track.confidence *= 0.72
                track.position_uncertainty = min(
                    96.0,
                    track.position_uncertainty
                    + decision_steps * (3.0 if overlaps_detection else 8.0),
                )
            if track.missed <= self.maximum_missed:
                retained.append(track)
        for detection_index in np.flatnonzero(np.logical_not(used_detections)):
            retained.append(self._new_track(
                detections[detection_index], observation_step=observation_step
            ))
        self.last_ambiguous_track_count = ambiguous_tracks
        self.tracks = self._trim_tracks(retained)
        self._step += decision_steps

    @property
    def known_velocity_fraction(self) -> float:
        if not self.tracks:
            return 0.0
        return float(np.mean([track.velocity_known for track in self.tracks]))

    @property
    def count_deficit_fraction(self) -> float:
        if self.expected_bullet_count <= 0:
            return 0.0
        return float(np.clip(
            (self.expected_bullet_count - self.last_detection_count)
            / self.expected_bullet_count,
            0.0,
            1.0,
        ))

    @property
    def occluded_fraction(self) -> float:
        if not self.tracks:
            return 0.0
        return float(np.mean([track.occluded_steps > 0 for track in self.tracks]))


class ImageOnlyPlannerAgent:
    """Run the exact planner on a belief reconstructed only from pixels."""

    def __init__(
        self,
        *,
        observation_size: int = 192,
        bullet_count: int = TARGET_TASK.bullet_count,
        bullet_size: int = 5,
        bullet_speed: float = 240.0,
        horizon_seconds: float = 1.5,
        reaction_seconds: float = 0.30,
        safety_horizons: Sequence[float] = (0.10,),
        minimum_known_velocity_fraction: float = 0.50,
        planner_kind: str = "recovery",
    ) -> None:
        self.tracker = PersistentImageTracker(
            bullet_speed=bullet_speed,
            target_track_count=tracking_capacity_for(bullet_count),
            expected_bullet_count=bullet_count,
        )
        self.horizon_seconds = float(horizon_seconds)
        self.reaction_seconds = float(reaction_seconds)
        self.safety_horizons = tuple(float(value) for value in safety_horizons)
        self.minimum_known_velocity_fraction = float(minimum_known_velocity_fraction)
        if planner_kind not in ("recovery", "exact"):
            raise ValueError("planner_kind must be 'recovery' or 'exact'")
        self.planner_kind = planner_kind
        self.bullet_speed = float(bullet_speed)
        from .env import BarrageVisionEnv
        self._belief = BarrageVisionEnv(
            bullet_count=bullet_count,
            bullet_size=bullet_size,
            bullet_speed_min=bullet_speed,
            bullet_speed_max=bullet_speed,
            targeted_bullet_probability=0.10,
            observation_size=observation_size,
            max_episode_seconds=120.0,
            randomize_initial_phase=False,
        )
        self._belief.reset(seed=0)
        self._initialized = False

    def close(self) -> None:
        self._belief.close()

    def reset(self, observation: np.ndarray | None = None) -> None:
        self.tracker.clear()
        self._initialized = False
        if observation is not None:
            self.tracker.initialize(observation)
            self._initialized = True

    def _belief_velocities(self) -> np.ndarray:
        velocities = []
        for track in self.tracker.tracks:
            if track.velocity_known:
                velocities.append(track.velocity)
                continue
            inward = self.tracker.plane_position - track.position
            magnitude = float(np.linalg.norm(inward))
            velocities.append(
                inward * (self.bullet_speed / max(magnitude, 1e-6))
            )
        return np.asarray(velocities, dtype=np.float32)

    def _synchronize_belief(self, observation: np.ndarray) -> None:
        tracks = self.tracker.tracks
        positions = np.asarray([track.position for track in tracks], np.float32)
        self._belief.bullet_count = len(tracks)
        self._belief._pending_bullet_count = len(tracks)
        self._belief.bullet_positions = positions.copy()
        self._belief.bullet_velocities = self._belief_velocities()
        self._belief.bullet_is_targeted = np.zeros(len(tracks), dtype=np.bool_)
        self._belief.bullet_speed = self.bullet_speed
        # The belief contains only bullets recovered from the current image.
        # Mark the deployment opening complete so counterfactual planning cannot
        # append hidden synthetic batches after every synchronization.
        self._belief.opening_spawned_batches = OPENING_BATCH_COUNT
        self._belief.plane_position[:] = self.tracker.plane_position
        self._belief.plane_velocity[:] = self.tracker.plane_velocity
        self._belief.frames = deque(
            (frame.copy() for frame in np.asarray(observation, dtype=np.uint8)),
            maxlen=self._belief.frame_stack,
        )

    def act(self, observation: np.ndarray) -> int:
        if not self._initialized:
            self.tracker.initialize(observation)
            self._initialized = True
        else:
            self.tracker.update(observation)
        if (
            not self.tracker.tracks
            or self.tracker.known_velocity_fraction
            < self.minimum_known_velocity_fraction
        ):
            return 0
        self._synchronize_belief(observation)
        if self.planner_kind == "exact":
            from .baselines import privileged_planner_supervision as planner
        else:
            from .recovery_planner import vectorized_recovery_supervision as planner
        supervision = planner(
            self._belief,
            horizon_seconds=self.horizon_seconds,
            reaction_seconds=self.reaction_seconds,
            wall_margin=80.0,
            wall_penalty_weight=0.35,
            safety_horizons=self.safety_horizons,
        )
        return int(supervision.action)
