"""Pygame screen adapter for the trained visual-set policy.

This module does not import or inspect ``Bullet.LIST`` or plane state.  It sees
only RGB pixels from the already-rendered Pygame surface and the actions it
previously emitted.
"""

from .timing import PHYSICS_FPS

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch

from .checkpoint_loader import (
    checkpoint_action_delay_steps,
    load_tracked_agent,
)
from .runtime_core import snapshot_surface_rgb, tracker_prediction_hints
from .task_spec import (
    PRODUCTION_ANALYTIC_SHIELD,
)
from .tracked_policy import TrackedFeatureExtractor, TrackedPolicySpec


@dataclass(frozen=True)
class RGBDetections:
    """Image-only detections before semantic downsampling."""

    semantic: np.ndarray
    bullet_positions: np.ndarray
    plane_position: np.ndarray | None


class DominantBackgroundSemanticizer:
    """Turn a rendered RGB frame into the semantic frames used for training.

    The dominant/median color is treated as background, so uniform background
    color changes do not require retraining.  Sprite color is not hard-coded;
    the player is identified by temporal proximity to its previous center.
    """

    _BULLET_OFFSETS = tuple(
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if not (abs(dx) == 2 and abs(dy) == 2)
    )

    def __init__(
        self,
        output_size: int = 96,
        color_threshold: float = 28.0,
        bullet_size: int = 5,
        recovery_edge_width: int = 16,
        recovery_stripe_width: int = 64,
    ) -> None:
        self.output_size = int(output_size)
        self.color_threshold = float(color_threshold)
        self.bullet_size = int(bullet_size)
        self.recovery_edge_width = max(1, int(recovery_edge_width))
        self.recovery_stripe_width = max(1, int(recovery_stripe_width))
        self.previous_plane: Optional[np.ndarray] = None
        self._recovery_stripe_index = 0
        self.last_detection_mode = "full"
        self.last_predicted_match_count = 0
        self.last_recovery_detection_count = 0

    def reset(self) -> None:
        self.previous_plane = None
        self._recovery_stripe_index = 0
        self.last_detection_mode = "full"
        self.last_predicted_match_count = 0
        self.last_recovery_detection_count = 0

    @staticmethod
    def _candidate_hypotheses(bullet_mask: np.ndarray) -> np.ndarray:
        """Return sparse centre hypotheses in y/x order, including clipped edges."""
        height, width = bullet_mask.shape
        # Find the sparse flat indices before deriving coordinates.  NumPy's
        # multidimensional nonzero path maintains coordinates across the frame;
        # divmod visits only the visible pixels and keeps C-order coordinates.
        visible = np.column_stack(np.divmod(np.flatnonzero(bullet_mask), width))
        if len(visible) == 0:
            return np.empty((0, 2), dtype=np.int64)
        hypotheses = [visible]
        right_rows = np.unique(visible[visible[:, 1] >= width - 2, 0])
        if len(right_rows):
            hypotheses.append(np.column_stack((right_rows, np.full_like(right_rows, width))))
        bottom_columns = np.unique(visible[visible[:, 0] >= height - 2, 1])
        if len(bottom_columns):
            hypotheses.append(np.column_stack((np.full_like(bottom_columns, height), bottom_columns)))
        if len(right_rows) and len(bottom_columns):
            hypotheses.append(np.asarray([[height, width]], dtype=np.int64))
        return visible if len(hypotheses) == 1 else np.concatenate(hypotheses, axis=0)

    def _validated_candidates(
        self, bullet_mask: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        from .rgb_sprite_kernel import validate_candidates

        if validate_candidates is None:
            return self._validated_candidates_numpy(bullet_mask, candidates)
        return validate_candidates(
            bullet_mask, np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
        )

    def _validated_candidates_numpy(
        self, bullet_mask: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """Keep hypotheses whose complete visible sprite template is white."""
        height, width = bullet_mask.shape
        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
        if len(candidates) == 0:
            return candidates
        expected = np.zeros(len(candidates), dtype=np.uint8)
        for dy, dx in self._BULLET_OFFSETS:
            source_y = candidates[:, 0] + dy
            source_x = candidates[:, 1] + dx
            valid = (
                (source_y >= 0)
                & (source_y < height)
                & (source_x >= 0)
                & (source_x < width)
            )
            expected += valid
            # A single missing in-bounds template pixel permanently rejects a
            # hypothesis.  Keep only survivors for subsequent offsets, retaining
            # their original order and the same clipped-sprite visibility count.
            matched = ~valid
            matched[valid] = bullet_mask[source_y[valid], source_x[valid]]
            if not np.all(matched):
                candidates = candidates[matched]
                expected = expected[matched]
                if len(candidates) == 0:
                    return candidates
        return candidates[expected >= 8]

    def _candidate_centers(self, bullet_mask: np.ndarray) -> np.ndarray:
        """Recover 5x5 sprite instances from their possibly touching union.

        A true sprite centre explains every in-bounds pixel of the fixed bullet
        mask.  Greedy residual coverage separates most touching sprites while
        retaining clipped centres on the four screen edges.  Exact coincident
        sprites are intentionally represented once because a single RGB frame
        contains no information that can distinguish them.
        """
        candidates = self._candidate_hypotheses(bullet_mask)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.float32)
        # Every non-clipped centre is itself a white sprite pixel.  Generate
        # hypotheses only from those ~2k sparse pixels instead of convolving
        # 21 offsets over all 672k source pixels.  Right/bottom centres lie one
        # pixel beyond the array under Pygame's Rect convention, so infer those
        # boundary hypotheses from nearby visible pixels as well.
        candidates = self._validated_candidates(bullet_mask, candidates)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.float32)

        return self._select_cover(bullet_mask, candidates)

    def _select_cover(
        self,
        bullet_mask: np.ndarray,
        candidates: np.ndarray,
        *,
        uncovered_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        from .rgb_sprite_kernel import select_cover

        if select_cover is None:
            return self._select_cover_numpy(
                bullet_mask, candidates, uncovered_mask=uncovered_mask
            )
        return select_cover(
            np.asarray(bullet_mask if uncovered_mask is None else uncovered_mask, dtype=np.bool_),
            np.asarray(candidates, dtype=np.int64).reshape(-1, 2),
        )

    def _select_cover_numpy(
        self,
        bullet_mask: np.ndarray,
        candidates: np.ndarray,
        *,
        uncovered_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select sprite centres with the existing deterministic residual cover."""
        height, width = bullet_mask.shape
        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.float32)
        residual = np.asarray(
            bullet_mask if uncovered_mask is None else uncovered_mask,
            dtype=np.bool_,
        )
        remaining = int(np.count_nonzero(residual))
        if remaining == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Candidate count is normally close to the number of bullets.  Keep the
        # exact deterministic greedy cover, but index each covered pixel back to
        # the candidates that contain it.  Selecting one sprite then updates only
        # overlapping candidates instead of rescanning the complete candidate
        # matrix after every selection.
        sentinel = height * width
        coverage = np.full(
            (len(candidates), len(self._BULLET_OFFSETS)),
            sentinel,
            dtype=np.int32,
        )
        candidate_y = candidates[:, 0]
        candidate_x = candidates[:, 1]
        for offset_index, (dy, dx) in enumerate(self._BULLET_OFFSETS):
            source_y = candidate_y + dy
            source_x = candidate_x + dx
            valid = (
                (source_y >= 0)
                & (source_y < height)
                & (source_x >= 0)
                & (source_x < width)
            )
            coverage[valid, offset_index] = (
                source_y[valid] * width + source_x[valid]
            )
        uncovered = np.concatenate((
            residual.reshape(-1),
            np.asarray([False], dtype=np.bool_),
        ))
        gains = np.count_nonzero(uncovered[coverage], axis=1).astype(
            np.int32, copy=False
        )
        flattened = coverage.reshape(-1)
        valid_flattened = flattened != sentinel
        pixel_values = flattened[valid_flattened]
        candidate_values = np.repeat(
            np.arange(len(candidates), dtype=np.int32), coverage.shape[1]
        )[valid_flattened]
        pixel_order = np.argsort(pixel_values, kind="stable")
        sorted_pixels = pixel_values[pixel_order]
        sorted_candidates = candidate_values[pixel_order]
        selected: list[int] = []
        while remaining > 0:
            best = int(np.argmax(gains))
            if gains[best] < 2:
                break
            selected.append(best)
            newly_covered = coverage[best][uncovered[coverage[best]]]
            remaining -= len(newly_covered)
            uncovered[newly_covered] = False
            left = np.searchsorted(sorted_pixels, newly_covered, side="left")
            right = np.searchsorted(sorted_pixels, newly_covered, side="right")
            affected = np.concatenate(tuple(
                sorted_candidates[start:stop]
                for start, stop in zip(left, right)
                if stop > start
            ))
            np.add.at(gains, affected, -1)
            # Gains only decrease.  Selected candidates stay below the >=2
            # threshold without rescanning every previously selected entry.
            gains[best] = -1
        if not selected:
            return np.empty((0, 2), dtype=np.float32)
        # Return x/y order because all downstream normalization is Cartesian.
        return candidates[np.asarray(selected, dtype=np.int32), ::-1].astype(np.float32)

    def _erase_sprite_coverage(
        self, residual: np.ndarray, centers_yx: np.ndarray
    ) -> None:
        from .rgb_sprite_kernel import erase_coverage

        if erase_coverage is None:
            self._erase_sprite_coverage_numpy(residual, centers_yx)
            return
        erase_coverage(residual, np.asarray(centers_yx, dtype=np.int64).reshape(-1, 2))

    def _erase_sprite_coverage_numpy(
        self, residual: np.ndarray, centers_yx: np.ndarray
    ) -> None:
        """Remove pixels explained by already selected centres in place."""
        height, width = residual.shape
        centers = np.asarray(centers_yx, dtype=np.int64).reshape(-1, 2)
        if len(centers) == 0:
            return
        for dy, dx in self._BULLET_OFFSETS:
            source_y = centers[:, 0] + dy
            source_x = centers[:, 1] + dx
            valid = (
                (source_y >= 0)
                & (source_y < height)
                & (source_x >= 0)
                & (source_x < width)
            )
            residual[source_y[valid], source_x[valid]] = False

    def _predictive_centers(
        self,
        bullet_mask: np.ndarray,
        predicted_positions: np.ndarray,
        search_radii: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Match prior image tracks locally, then discover births in bounded scans."""
        height, width = bullet_mask.shape
        hypotheses = self._candidate_hypotheses(bullet_mask)
        valid = self._validated_candidates(bullet_mask, hypotheses)
        predicted_xy = np.asarray(predicted_positions, dtype=np.float32).reshape(-1, 2)
        radii = np.asarray(search_radii, dtype=np.float32).reshape(-1)
        if len(radii) != len(predicted_xy):
            raise ValueError("one search radius is required for each predicted position")

        selected_yx = np.empty((0, 2), dtype=np.int64)
        if len(valid):
            predicted_pixels = predicted_xy * np.asarray([width, height], np.float32)
            distance_squared = self._squared_distance_2d(
                valid[None, :, ::-1].astype(np.float32),
                predicted_pixels[:, None, :],
            )
            nearest = np.argmin(distance_squared, axis=1)
            accepted = distance_squared[np.arange(len(predicted_xy)), nearest] <= np.square(radii)
            if np.any(accepted):
                proposed = np.unique(valid[nearest[accepted]], axis=0)
                # A touching-sprite union can contain several locally valid
                # centres.  Do not let motion prediction choose among those
                # visually ambiguous alternatives; leave that small component
                # to the canonical residual cover so prediction cannot invent
                # extra bullets relative to the full-frame detector.
                local_distance_squared = self._squared_distance_2d(
                    proposed[:, None, :].astype(np.float32),
                    valid[None, :, :].astype(np.float32),
                )
                local_neighbors = np.count_nonzero(
                    local_distance_squared <= 4.0,
                    axis=1,
                )
                selected_yx = proposed[local_neighbors == 1]
        predicted_count = int(len(selected_yx))

        # Every frame covers all spawn edges and one rotating vertical shard.
        # The two-pixel halo preserves complete 5x5 templates at shard borders.
        edge = min(self.recovery_edge_width, max(height, width))
        stripe_count = max(1, int(np.ceil(width / self.recovery_stripe_width)))
        stripe_index = self._recovery_stripe_index % stripe_count
        stripe_start = stripe_index * self.recovery_stripe_width
        stripe_stop = min(width, stripe_start + self.recovery_stripe_width)
        self._recovery_stripe_index = (stripe_index + 1) % stripe_count
        halo = max(2, int(np.ceil(self.bullet_size / 2.0)))
        scan = np.zeros_like(bullet_mask)
        scan[: min(height, edge + halo), :] = True
        scan[max(0, height - edge - halo) :, :] = True
        scan[:, : min(width, edge + halo)] = True
        scan[:, max(0, width - edge - halo) :] = True
        scan[:, max(0, stripe_start - halo) : min(width, stripe_stop + halo)] = True
        residual = bullet_mask.copy()
        self._erase_sprite_coverage(residual, selected_yx)
        bounded_yx = np.empty((0, 2), dtype=np.int64)
        if len(valid):
            core = (
                (valid[:, 0] <= edge)
                | (valid[:, 0] >= height - edge)
                | (valid[:, 1] <= edge)
                | (valid[:, 1] >= width - edge)
                | (
                    (valid[:, 1] >= stripe_start)
                    & (valid[:, 1] < stripe_stop)
                )
            )
            recovery_candidates = valid[core]
            recovered_xy = self._select_cover(
                bullet_mask,
                recovery_candidates,
                uncovered_mask=residual & scan,
            )
            bounded_yx = recovered_xy[:, ::-1].astype(np.int64)
            self._erase_sprite_coverage(residual, bounded_yx)

        # Finish only the still-unexplained residue.  In the steady state this
        # selects a handful of births/crossing corrections rather than running
        # the full set cover over all configured sprites, while avoiding temporary
        # blind spots between rotating recovery shards.
        residual_xy = self._select_cover(
            bullet_mask, valid, uncovered_mask=residual
        )
        residual_yx = residual_xy[:, ::-1].astype(np.int64)
        recovered_yx = (
            np.unique(np.concatenate((bounded_yx, residual_yx), axis=0), axis=0)
            if len(bounded_yx) or len(residual_yx)
            else np.empty((0, 2), dtype=np.int64)
        )
        recovery_count = int(len(recovered_yx))
        combined = (
            np.unique(np.concatenate((selected_yx, recovered_yx), axis=0), axis=0)
            if predicted_count or recovery_count
            else np.empty((0, 2), dtype=np.int64)
        )
        return combined[:, ::-1].astype(np.float32), predicted_count, recovery_count

    @staticmethod
    def _squared_distance_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Keep the two-term float32 sum without a pairwise coordinate tensor."""
        delta_first = first[..., 0] - second[..., 0]
        delta_second = first[..., 1] - second[..., 1]
        return delta_first * delta_first + delta_second * delta_second

    @staticmethod
    def _classify_foreground(
        image: np.ndarray, foreground: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from .rgb_sprite_kernel import classify_sparse

        if classify_sparse is not None and image.dtype == np.uint8:
            return classify_sparse(image, np.flatnonzero(foreground))
        return DominantBackgroundSemanticizer._classify_foreground_numpy(image, foreground)

    @staticmethod
    def _classify_foreground_numpy(
        image: np.ndarray, foreground: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify only visible pixels, preserving row-major detection order."""
        visible_y, visible_x = np.divmod(
            np.flatnonzero(foreground), foreground.shape[1]
        )
        # The full-size color comparisons repeatedly scanned the uniform
        # background.  Those pixels can never enter either sprite mask.
        red = image[visible_y, visible_x, 0]
        green = image[visible_y, visible_x, 1]
        blue = image[visible_y, visible_x, 2]
        is_bullet = (
            (np.maximum(red, green) - np.minimum(red, green) <= 16)
            & (np.maximum(red, blue) - np.minimum(red, blue) <= 16)
        )
        bullet_mask = np.zeros(foreground.shape, dtype=np.bool_)
        bullet_mask[visible_y[is_bullet], visible_x[is_bullet]] = True
        is_plane = ~is_bullet
        plane_yx = np.column_stack((visible_y[is_plane], visible_x[is_plane]))
        return bullet_mask, plane_yx

    @staticmethod
    def _draw_box(
        semantic: np.ndarray,
        position: np.ndarray,
        value: int,
        half_width: int,
        half_height: int,
    ) -> tuple[int, int]:
        size = semantic.shape[0]
        x = int(round(float(position[0]) * (size - 1)))
        y = int(round(float(position[1]) * (size - 1)))
        semantic[
            max(0, y - half_height) : min(size, y + half_height + 1),
            max(0, x - half_width) : min(size, x + half_width + 1),
        ] = value
        return x, y

    @staticmethod
    def _foreground_mask(
        image: np.ndarray, background: np.ndarray, color_threshold: float
    ) -> np.ndarray:
        from .foreground_kernel import fused_foreground

        if fused_foreground is None:
            return DominantBackgroundSemanticizer._foreground_mask_numpy(
                image, background, color_threshold)
        lower = np.full(3, -1, np.int64)
        upper = np.full(3, 256, np.int64)
        for index in range(3):
            lo = background[index] - color_threshold
            hi = background[index] + color_threshold
            if lo >= 255 or hi <= 0:
                return np.ones(image.shape[:2], dtype=np.bool_)
            if lo >= 0:
                lower[index] = int(np.floor(lo))
            if hi <= 255:
                upper[index] = int(np.ceil(hi))
        return fused_foreground(image, lower, upper)

    @staticmethod
    def _foreground_mask_numpy(
        image: np.ndarray, background: np.ndarray, color_threshold: float
    ) -> np.ndarray:
        """Apply the original inclusive thresholds directly to uint8 channels."""
        foreground = np.zeros(image.shape[:2], dtype=np.bool_)
        for index in range(3):
            lower = background[index] - color_threshold
            upper = background[index] + color_threshold
            if lower >= 255 or upper <= 0:
                foreground.fill(True)
                return foreground
            channel = image[:, :, index]
            # Integer pixels let us round each finite bound without changing
            # the inclusive comparison.  Bounds outside uint8 contribute no
            # pixels; NaN also contributes none, matching the float comparison.
            if lower >= 0:
                foreground |= channel <= int(np.floor(lower))
            if upper <= 255:
                foreground |= channel >= int(np.ceil(upper))
        return foreground

    def detect(
        self,
        rgb: np.ndarray,
        *,
        include_semantic: bool = True,
        predicted_bullet_positions: np.ndarray | None = None,
        prediction_search_radii: np.ndarray | None = None,
    ) -> RGBDetections:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("rgb frame must have shape [height, width, channels]")
        height, width = image.shape[:2]
        # A sparse grid is sufficient for a uniform/dominant background and is
        # over 2,000x smaller than sorting every source pixel for a full median.
        stride_y = max(1, height // 32)
        stride_x = max(1, width // 32)
        samples = image[::stride_y, ::stride_x, :3].reshape(-1, 3)
        background = np.median(samples, axis=0).astype(np.int16)
        foreground = self._foreground_mask(
            image, background, self.color_threshold
        )
        # The deployed bullet sprite is achromatic white.  Using its observable
        # color, rather than simulator positions, preserves the image-only
        # boundary and separates the red player even when sprites touch.
        bullet_mask, plane_yx = self._classify_foreground(image, foreground)
        semantic = (
            np.zeros((self.output_size, self.output_size), dtype=np.uint8)
            if include_semantic
            else np.empty((0, 0), dtype=np.uint8)
        )
        if predicted_bullet_positions is None:
            bullet_xy = self._candidate_centers(bullet_mask)
            self.last_detection_mode = "full"
            self.last_predicted_match_count = 0
            self.last_recovery_detection_count = 0
        else:
            predictions = np.asarray(
                predicted_bullet_positions, dtype=np.float32
            ).reshape(-1, 2)
            if len(predictions) == 0:
                bullet_xy = self._candidate_centers(bullet_mask)
                self.last_detection_mode = "full_recovery"
                self.last_predicted_match_count = 0
                self.last_recovery_detection_count = int(len(bullet_xy))
            else:
                if prediction_search_radii is None:
                    radii = np.full(len(predictions), 12.0, dtype=np.float32)
                else:
                    radii = np.asarray(
                        prediction_search_radii, dtype=np.float32
                    ).reshape(-1)
                bullet_xy, predicted_count, recovery_count = self._predictive_centers(
                    bullet_mask, predictions, radii
                )
                self.last_detection_mode = "predictive"
                self.last_predicted_match_count = predicted_count
                self.last_recovery_detection_count = recovery_count
        if len(bullet_xy) == 0 and len(plane_yx) == 0:
            return RGBDetections(
                semantic,
                np.empty((0, 2), dtype=np.float32),
                None if self.previous_plane is None else self.previous_plane.copy(),
            )

        bullet_centers: list[tuple[int, int]] = []
        # NumPy 1.x scalar float32/int division promotes to float64.  Preserve
        # that rounding before the existing float32 output while batching it.
        normalized_bullets = (
            bullet_xy.astype(np.float64)
            / np.asarray([max(width, 1), max(height, 1)], dtype=np.float64)
        ).astype(np.float32)
        if include_semantic:
            bullet_radius = max(1, int(np.ceil(self.bullet_size / 2.0)))
            for position in normalized_bullets:
                bullet_centers.append(self._draw_box(
                    semantic,
                    position,
                    160 + self.bullet_size,
                    bullet_radius,
                    bullet_radius,
                ))
            for x, y in bullet_centers:
                semantic[y, x] = 255
        if len(plane_yx):
            plane = np.asarray(
                [
                    float(plane_yx[:, 1].mean()) / max(width, 1),
                    float(plane_yx[:, 0].mean()) / max(height, 1),
                ],
                np.float32,
            )
            self.previous_plane = plane
        elif self.previous_plane is not None:
            plane = self.previous_plane.copy()
        else:
            return RGBDetections(semantic, normalized_bullets, None)
        if include_semantic:
            plane_half_width = max(2, int(round(18 * self.output_size / width)) // 2)
            plane_half_height = max(2, int(round(12 * self.output_size / height)) // 2)
            self._draw_box(
                semantic, plane, 96, plane_half_width, plane_half_height
            )
        return RGBDetections(semantic, normalized_bullets, plane.copy())

    def convert(self, rgb: np.ndarray) -> np.ndarray:
        """Compatibility semantic frame for diagnostics and legacy checkpoints."""
        return self.detect(rgb, include_semantic=True).semantic


class LiveVisualController:
    """Load ``best.pt`` and choose an action from rendered pixels every 4 frames."""

    def __init__(
        self,
        checkpoint: str,
        device_name: str = "cpu",
        decision_interval: int = 4,
        analytic_shield: bool = PRODUCTION_ANALYTIC_SHIELD,
        analytic_guard_horizon_seconds: float = 0.10,
        analytic_clearance_margin: float = 0.0,
        record_timings: bool = False,
        rgb_workers: int | None = None,
        analytic_shield_gate: str = "always",
        experimental_controller: str | None = None,
    ) -> None:
        device = torch.device(
            device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.record_timings = bool(record_timings)
        from .rgb_parallel import DEFAULT_RGB_WORKERS
        self.rgb_workers = DEFAULT_RGB_WORKERS if rgb_workers is None else max(0, int(rgb_workers))
        checkpoint_path = Path(checkpoint)
        metadata = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if int(metadata.get("model_version", 0)) != 12:
            raise ValueError(
                "the window requires a model_version 12 tracked checkpoint"
            )
        self.tracked_spec: TrackedPolicySpec
        self.tracked_extractor: TrackedFeatureExtractor
        self.tracked_initialized = False
        self.agent, self.tracked_spec, checkpoint_data = load_tracked_agent(
            str(checkpoint_path),
            device,
            checkpoint_data=metadata,
            materialize_state=True,
            analytic_shield=analytic_shield,
            analytic_guard_horizon_seconds=analytic_guard_horizon_seconds,
            analytic_clearance_margin=analytic_clearance_margin,
            analytic_shield_gate=analytic_shield_gate,
            experimental_controller=experimental_controller,
        )
        from .window_inference import enable_window_inference

        enable_window_inference(self.agent.model)
        self.agent._window_skip_unused_teacher_cost = True
        from .rgb_sprite_kernel import warmup_sprite_kernels

        warmup_sprite_kernels()
        from .window_tracker import WindowImageTracker
        self.action_delay_steps = checkpoint_action_delay_steps(checkpoint_data)
        self.tracked_extractor = TrackedFeatureExtractor(
            self.tracked_spec, refit_known_velocity=True,
            decision_dt=max(1, int(decision_interval)) / PHYSICS_FPS,
            tracker_class=WindowImageTracker,
        )
        self.tracked_extractor.tracker.history_limit = 8
        config = checkpoint_data["config"]
        self.semanticizer = DominantBackgroundSemanticizer(
            output_size=int(checkpoint_data.get("observation_size", 96)),
            bullet_size=int(config.get("bullet_size", 5)),
        )
        expected_interval = int(config.get("action_repeat", 4))
        self.decision_interval = max(1, int(decision_interval))
        if self.decision_interval != expected_interval:
            raise ValueError(
                "live decision interval does not match checkpoint action_repeat: "
                f"{self.decision_interval} != {expected_interval}"
            )
        self.frame_counter = 0
        self.action = 0
        self._last_capture_boundary: int | None = None
        self._stage_detection_ms: list[float] = []
        self._stage_tracking_ms: list[float] = []
        self._stage_model_ms: list[float] = []

    def reset(self) -> None:
        self.semanticizer.reset()
        from .window_tracker import WindowImageTracker
        self.tracked_extractor = TrackedFeatureExtractor(
            self.tracked_spec, refit_known_velocity=True,
            decision_dt=self.decision_interval / PHYSICS_FPS,
            tracker_class=WindowImageTracker,
        )
        self.tracked_extractor.tracker.history_limit = 8
        self.tracked_initialized = False
        self.frame_counter = 0
        self.action = 0
        self._last_capture_boundary = None
        self.begin_measurement()

    def begin_measurement(self) -> None:
        self._stage_detection_ms = []
        self._stage_tracking_ms = []
        self._stage_model_ms = []

    @staticmethod
    def _stage_statistics(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        return {
            "count": int(len(array)),
            "mean_ms": float(array.mean()),
            "p95_ms": float(np.percentile(array, 95.0)),
            "p99_ms": float(np.percentile(array, 99.0)),
        }

    def runtime_stage_report(self) -> dict[str, dict[str, float | int]]:
        return {
            "rgb_detection": self._stage_statistics(self._stage_detection_ms),
            "tracking_and_features": self._stage_statistics(self._stage_tracking_ms),
            "model_forward": self._stage_statistics(self._stage_model_ms),
        }

    def _act_detections(
        self, detections: RGBDetections, *, decision_steps: int = 1
    ) -> int:
        tracking_started = time.perf_counter() if self.record_timings else 0.0
        features = (
            self.tracked_extractor.step_detections(
                detections.bullet_positions,
                detections.plane_position,
                decision_steps=decision_steps,
            )
            if self.tracked_initialized
            else self.tracked_extractor.reset_detections(
                detections.bullet_positions, detections.plane_position
            )
        )
        self.tracked_initialized = True
        objects, mask, globals_ = features
        model_started = time.perf_counter() if self.record_timings else 0.0
        if self.record_timings:
            self._stage_tracking_ms.append((model_started - tracking_started) * 1000.0)
        action = int(self.agent.act_features(
            objects[None], mask[None], globals_[None], deterministic=True,
            decision_steps=decision_steps,
        )[0])
        if self.record_timings:
            self._stage_model_ms.append((time.perf_counter() - model_started) * 1000.0)
        return action

    def prime_rgb(self, rgb: np.ndarray) -> int:
        """Initialize the image tracker and choose the first action at t=0."""
        self.agent.reset_state()
        detections = self.semanticizer.detect(rgb, include_semantic=False)
        self.action = self._act_detections(detections)
        self.frame_counter = 0
        # Model loading and JIT warmup leave cyclic compiler objects pending.
        # Drain them at the priming boundary, before realtime frame timing,
        # so an ordinary tracking allocation cannot trigger that full scan.
        # Keep automatic collection enabled during play.
        import gc

        gc.collect(2)
        return self.action

    def _prediction_hints(
        self, rgb: np.ndarray, decision_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict search windows exclusively from prior image-derived tracks."""
        return tracker_prediction_hints(
            self.tracked_extractor.tracker,
            tuple(np.asarray(rgb).shape),
            decision_steps,
            reuse_for_update=True,
        )

    def act_rgb_frame(self, rgb: np.ndarray) -> int:
        """Act on one externally scheduled decision frame.

        The causal deployment pipeline owns the fixed-decision schedule, so this method
        deliberately bypasses ``frame_counter`` while retaining the image-only
        detector and persistent tracker state.
        """
        if not self.tracked_initialized:
            return self.prime_rgb(rgb)
        self.action = self._observe_due_rgb(rgb)
        self.frame_counter = 0
        return self.action

    def act_rgb_frame_at_boundary(self, rgb: np.ndarray, capture_boundary: int) -> int:
        """Act with the actual elapsed decision count after latest-only drops."""
        boundary = int(capture_boundary)
        decision_steps = (
            1
            if self._last_capture_boundary is None
            else max(1, boundary - self._last_capture_boundary)
        )
        self._last_capture_boundary = boundary
        if not self.tracked_initialized:
            return self.prime_rgb(rgb)
        self.action = self._observe_due_rgb(rgb, decision_steps=decision_steps)
        self.frame_counter = 0
        return self.action

    def observe_rgb(self, rgb: np.ndarray, physics_steps: int = 1) -> int:
        if not self.decision_due(physics_steps):
            return self.action
        return self._observe_due_rgb(rgb)

    def decision_due(self, physics_steps: int = 1) -> bool:
        """Advance the fixed-step scheduler without reading another input."""
        self.frame_counter += max(0, int(physics_steps))
        if self.frame_counter < self.decision_interval:
            return False
        self.frame_counter %= self.decision_interval
        return True

    def _observe_due_rgb(self, rgb: np.ndarray, *, decision_steps: int = 1) -> int:
        detection_started = time.perf_counter() if self.record_timings else 0.0
        predictions, radii = self._prediction_hints(rgb, decision_steps)
        try:
            detections = self.semanticizer.detect(
                rgb,
                include_semantic=False,
                predicted_bullet_positions=predictions,
                prediction_search_radii=radii,
            )
            if self.record_timings:
                self._stage_detection_ms.append(
                    (time.perf_counter() - detection_started) * 1000.0
                )
            self.action = self._act_detections(
                detections, decision_steps=decision_steps
            )
            return self.action
        finally:
            self.tracked_extractor.tracker.__dict__.pop("_pending_predictions", None)

    def observe_surface(self, surface: "object", physics_steps: int = 1) -> int:
        if not self.decision_due(physics_steps):
            return self.action
        return self.observe_due_surface(surface)

    def observe_due_surface(self, surface: "object") -> int:
        """Choose an action from the current RGB surface at an exact due step."""
        from .rgb_parallel import rgb_workers
        with rgb_workers(self.rgb_workers):
            return self._observe_due_rgb(snapshot_surface_rgb(surface))

    def prime_surface(self, surface: "object") -> int:
        from .rgb_parallel import rgb_workers
        with rgb_workers(self.rgb_workers):
            return self.prime_rgb(snapshot_surface_rgb(surface))
