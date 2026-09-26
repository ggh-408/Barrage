"""Exact numerical and ordering checks for low-latency image processing."""

import copy
import unittest

import numpy as np

from barrage_rl.image_oracle import BulletTrack, PersistentImageTracker
from barrage_rl.live_screen import DominantBackgroundSemanticizer


def _reference_fit_velocity(tracker: PersistentImageTracker, track: BulletTrack) -> None:
    """The uncached per-track regression used before the latency changes."""
    if track.velocity_known and not tracker.refit_known_velocity:
        return
    history = track.history[-8:]
    if len(history) < 4:
        return
    steps = np.asarray([item[0] for item in history], np.float32)
    positions = np.stack([item[1] for item in history]).astype(np.float32)
    centered = steps - float(steps.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return
    displacement = (centered[:, None] * positions).sum(axis=0) / denominator
    estimate = displacement / tracker.decision_dt
    magnitude = float(np.linalg.norm(estimate))
    if magnitude < 0.20 * tracker.bullet_speed:
        return
    track.velocity = (
        estimate * (tracker.bullet_speed / max(magnitude, 1e-6))
    ).astype(np.float32)
    track.velocity_known = True


class _UncachedRegressionTracker(PersistentImageTracker):
    def _fit_velocities(self, tracks: list[BulletTrack]) -> None:
        for track in tracks:
            _reference_fit_velocity(self, track)

    def _fit_velocity(self, track: BulletTrack, *, regression_cache=None) -> None:
        _reference_fit_velocity(self, track)


def _reference_cover(
    detector: DominantBackgroundSemanticizer,
    bullet_mask: np.ndarray,
    candidates: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Full recomputation oracle, independent of incremental gain updates."""
    height, width = bullet_mask.shape
    candidates = np.asarray(candidates, np.int64).reshape(-1, 2)
    if not len(candidates):
        return np.empty((0, 2), np.float32)
    sentinel = height * width
    coverage = np.full((len(candidates), 21), sentinel, np.int32)
    for index, (dy, dx) in enumerate(detector._BULLET_OFFSETS):
        y = candidates[:, 0] + dy
        x = candidates[:, 1] + dx
        valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        coverage[valid, index] = y[valid] * width + x[valid]
    uncovered = np.concatenate((residual.reshape(-1).copy(), [False]))
    available = np.ones(len(candidates), np.bool_)
    selected = []
    while np.count_nonzero(uncovered):
        gains = np.count_nonzero(uncovered[coverage], axis=1)
        gains[~available] = -1
        best = int(np.argmax(gains))
        if gains[best] < 2:
            break
        selected.append(best)
        uncovered[coverage[best]] = False
        available[best] = False
    return candidates[np.asarray(selected, np.int64), ::-1].astype(np.float32)


class CoreLatencyEquivalenceTests(unittest.TestCase):
    def assert_tracker_equal(self, actual, expected) -> None:
        for attribute in (
            "_step", "_next_track_id", "last_detection_count",
            "last_ambiguous_track_count",
        ):
            self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
        np.testing.assert_array_equal(actual.plane_position, expected.plane_position)
        np.testing.assert_array_equal(actual.plane_velocity, expected.plane_velocity)
        self.assertEqual(len(actual.tracks), len(expected.tracks))
        for first, second in zip(actual.tracks, expected.tracks):
            for key, value in vars(first).items():
                other = getattr(second, key)
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, other, err_msg=key)
                elif key == "history":
                    self.assertEqual(len(value), len(other))
                    for (step, position), (other_step, other_position) in zip(value, other):
                        self.assertEqual(step, other_step)
                        np.testing.assert_array_equal(position, other_position)
                else:
                    self.assertEqual(value, other, key)

    def test_foreground_integer_bounds_exhaust_uint8_pixels_and_backgrounds(self) -> None:
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = np.repeat(pixels[:, :, None], 3, axis=2)
        layouts = (image, image.transpose(1, 0, 2), image[::-1, ::-1])
        thresholds = (
            28.0, 28.25, 0.0, 0.25, 0.5, 1.0, 1.25, 254.5, 255.0,
            255.5, 256.0, -0.25, -28.0, -255.0,
            float("inf"), float("-inf"), float("nan"),
        )
        for background_value in range(256):
            background = np.full(3, background_value, np.int16)
            for threshold in thresholds:
                for layout in layouts:
                    values = layout[:, :, 0]
                    expected = (
                        (values <= background[0] - threshold)
                        | (values >= background[0] + threshold)
                    )
                    actual = DominantBackgroundSemanticizer._foreground_mask(
                        layout, background, threshold
                    )
                    np.testing.assert_array_equal(
                        actual, expected,
                        err_msg=f"background={background_value}, threshold={threshold}",
                    )

    def test_foreground_channel_union_matches_float_reference(self) -> None:
        rng = np.random.default_rng(48907)
        image = rng.integers(0, 256, size=(19, 27, 4), dtype=np.uint8)
        for layout in (image, image.transpose(1, 0, 2), image[::2, ::-2]):
            before = layout.copy()
            for background in (
                np.asarray([0, 127, 255], np.int16),
                np.asarray([255, 0, 128], np.int16),
                np.asarray([31, 67, 11], np.int16),
            ):
                for threshold in (28.0, 17.25, 0.0, -17.25, 256.0, float("nan")):
                    expected = np.zeros(layout.shape[:2], dtype=np.bool_)
                    for index in range(3):
                        channel = layout[:, :, index]
                        expected |= (
                            (channel <= background[index] - threshold)
                            | (channel >= background[index] + threshold)
                        )
                    actual = DominantBackgroundSemanticizer._foreground_mask(
                        layout, background, threshold
                    )
                    np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(layout, before)

    def test_cached_regression_preserves_exact_float32_velocity(self) -> None:
        rng = np.random.default_rng(73109)
        tracker = PersistentImageTracker()
        cache = {}
        schedules = (
            (0, 1, 2, 3), (0, 1, 3, 6), (7, 8, 9, 10, 11, 12, 13, 14),
            (7, 8, 10, 11, 12, 14, 15, 18), (5, 5, 5, 5),
            (16_777_216, 16_777_217, 16_777_218, 16_777_219),
        )
        for schedule in schedules:
            for velocity_scale in (0.0, 0.1, 8.0, 30.0):
                positions = rng.normal(size=(len(schedule), 2)).astype(np.float32)
                positions *= velocity_scale
                positions += np.float32(400.0)
                track = BulletTrack(
                    track_id=1,
                    position=positions[-1].copy(),
                    velocity=np.asarray([1.25, -7.5], np.float32),
                    history=list(zip(schedule, positions)),
                )
                expected = copy.deepcopy(track)
                _reference_fit_velocity(tracker, expected)
                tracker._fit_velocity(track, regression_cache=cache)
                np.testing.assert_array_equal(track.velocity, expected.velocity)
                self.assertEqual(track.velocity_known, expected.velocity_known)
        self.assertEqual(set(cache), set(schedules))

    def test_tracker_regression_cache_handles_occlusion_and_skipped_frames(self) -> None:
        rng = np.random.default_rng(43021)
        positions = rng.uniform(50.0, 770.0, size=(300, 2)).astype(np.float32)
        angles = rng.uniform(-np.pi, np.pi, size=300).astype(np.float32)
        velocity = np.column_stack((np.cos(angles), np.sin(angles))) * np.float32(240.0)
        plane = np.asarray([410.0, 410.0], np.float32)
        actual = PersistentImageTracker()
        expected = _UncachedRegressionTracker()
        for frame in range(24):
            elapsed = (1, 1, 2, 1, 3)[frame % 5]
            positions = positions + velocity * (actual.decision_dt * elapsed)
            detections = np.rint(positions).astype(np.float32)
            if frame % 4 == 1:
                detections = detections[np.arange(300) % 7 != 0]
            elif frame % 4 == 2:
                detections[1:4] = detections[0]
            if frame == 17:
                detections = np.empty((0, 2), np.float32)
            for tracker in (actual, expected):
                tracker.update_detections(
                    detections, plane, normalized=False, decision_steps=elapsed
                )
            self.assert_tracker_equal(actual, expected)

    def test_association_counts_preserve_crossing_identities_and_empty_updates(self) -> None:
        tracker = PersistentImageTracker()
        plane = np.asarray([410.0, 410.0], np.float32)
        tracker.initialize_detections(
            np.asarray([[400.0, 410.0], [404.0, 410.0], [408.0, 410.0]], np.float32),
            plane, normalized=False,
        )
        tracker.update_detections(
            np.asarray([[404.0, 410.0]], np.float32), plane, normalized=False
        )
        self.assertEqual(tracker.last_ambiguous_track_count, 2)
        self.assertEqual({track.association_group_size for track in tracker.tracks}, {3})
        self.assertEqual({track.track_id for track in tracker.tracks}, {0, 1, 2})
        tracker.update_detections(np.empty((0, 2), np.float32), plane, normalized=False)
        self.assertEqual(tracker.last_ambiguous_track_count, 0)
        self.assertEqual({track.association_group_size for track in tracker.tracks}, {1})
        self.assertEqual({track.missed for track in tracker.tracks}, {1})

    def test_cover_preserves_ties_clipped_edges_and_empty_residual(self) -> None:
        rng = np.random.default_rng(7921)
        detector = DominantBackgroundSemanticizer()
        for case in range(8):
            mask = np.zeros((53, 71), np.bool_)
            centers = np.concatenate((
                rng.integers([0, 0], [54, 72], size=(55, 2)),
                np.asarray([[0, 0], [53, 71], [0, 71], [53, 0], [22, 30], [22, 31]]),
            ))
            for y, x in centers:
                for dy, dx in detector._BULLET_OFFSETS:
                    if 0 <= y + dy < 53 and 0 <= x + dx < 71:
                        mask[y + dy, x + dx] = True
            candidates = detector._validated_candidates(
                mask, detector._candidate_hypotheses(mask)
            )
            residual = mask.copy()
            if case == 0:
                residual.fill(False)
            elif case % 2:
                residual[:, :case * 5] = False
            before = residual.copy()
            expected = _reference_cover(detector, mask, candidates, residual)
            actual = detector._select_cover(mask, candidates, uncovered_mask=residual)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(residual, before)

    def test_batched_normalization_matches_scalar_rounding_and_semantics(self) -> None:
        for height, width in ((120, 193), (193, 120), (820, 820)):
            mask = np.zeros((height, width), np.bool_)
            detector = DominantBackgroundSemanticizer(output_size=96)
            for y, x in ((7, 19), (31, 42), (height - 1, width - 1), (0, width)):
                for dy, dx in detector._BULLET_OFFSETS:
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        mask[y + dy, x + dx] = True
            for background in ((0, 0, 0), (17, 31, 63)):
                image = np.full((height, width, 3), background, np.uint8)
                image[mask] = 255
                candidates = detector._validated_candidates(
                    mask, detector._candidate_hypotheses(mask)
                )
                centers = _reference_cover(detector, mask, candidates, mask)
                normalized = np.asarray([
                    np.asarray([x / width, y / height], np.float32)
                    for x, y in centers
                ], np.float32)
                expected_semantic = np.zeros((96, 96), np.uint8)
                semantic_centers = [
                    detector._draw_box(expected_semantic, position, 165, 3, 3)
                    for position in normalized
                ]
                for x, y in semantic_centers:
                    expected_semantic[y, x] = 255
                for include_semantic in (False, True):
                    result = detector.detect(image, include_semantic=include_semantic)
                    np.testing.assert_array_equal(result.bullet_positions, normalized)
                    self.assertIsNone(result.plane_position)
                    if include_semantic:
                        np.testing.assert_array_equal(result.semantic, expected_semantic)


if __name__ == "__main__":
    unittest.main()
