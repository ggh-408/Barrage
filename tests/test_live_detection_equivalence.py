"""Exact regressions for sparse RGB classification and pairwise distances."""

import unittest

import numpy as np

from barrage_rl.live_screen import DominantBackgroundSemanticizer


class _DenseReferenceSemanticizer(DominantBackgroundSemanticizer):
    """Retain the original dense operations as an independent numeric oracle."""

    def _validated_candidates(self, bullet_mask, candidates):
        height, width = bullet_mask.shape
        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
        matched = np.zeros(len(candidates), dtype=np.uint8)
        expected = np.zeros(len(candidates), dtype=np.uint8)
        for dy, dx in self._BULLET_OFFSETS:
            source_y = candidates[:, 0] + dy
            source_x = candidates[:, 1] + dx
            valid = (
                (source_y >= 0) & (source_y < height)
                & (source_x >= 0) & (source_x < width)
            )
            expected += valid
            matched[valid] += bullet_mask[source_y[valid], source_x[valid]]
        return candidates[(matched == expected) & (expected >= 8)]

    @staticmethod
    def _classify_foreground(image, foreground):
        red, green, blue = (image[:, :, index] for index in range(3))
        bullets = (
            foreground
            & (np.maximum(red, green) - np.minimum(red, green) <= 16)
            & (np.maximum(red, blue) - np.minimum(red, blue) <= 16)
        )
        return bullets, np.argwhere(foreground & ~bullets)

    @staticmethod
    def _squared_distance_2d(first, second):
        delta = first - second
        return np.sum(delta * delta, axis=-1)


class LiveDetectionEquivalenceTests(unittest.TestCase):
    def test_early_template_rejection_preserves_clipped_candidates_and_duplicates(self):
        rng = np.random.default_rng(820384)
        detector = DominantBackgroundSemanticizer()
        reference = _DenseReferenceSemanticizer()
        for height, width in ((0, 0), (1, 1), (3, 5), (31, 47), (820, 820)):
            for density in (0.0, 0.01, 0.5, 1.0):
                mask = rng.random((height, width)) < density
                centers = rng.integers([-2, -2], [height + 3, width + 3], size=(300, 2))
                centers[:6] = [
                    [0, 0], [height, width], [0, width], [height, 0],
                    [height // 2, width // 2], [height // 2, width // 2],
                ]
                if density == 0.01:
                    for y, x in centers:
                        for dy, dx in detector._BULLET_OFFSETS:
                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                mask[y + dy, x + dx] = True
                for layout in (mask, mask.T, mask[::-2, ::-1]):
                    hypotheses = detector._candidate_hypotheses(layout)
                    if len(hypotheses) > 8_192:
                        hypotheses = hypotheses[:: (len(hypotheses) // 4_096)]
                    candidates = np.concatenate((centers, hypotheses), axis=0)
                    for ordered in (candidates, candidates[::-1], candidates[:0]):
                        before_mask = layout.copy()
                        before_candidates = ordered.copy()
                        actual = detector._validated_candidates(layout, ordered)
                        expected = reference._validated_candidates(layout, ordered)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.dtype, expected.dtype)
                        np.testing.assert_array_equal(layout, before_mask)
                        np.testing.assert_array_equal(ordered, before_candidates)

    def test_sparse_hypotheses_preserve_edge_candidates_and_row_order(self):
        rng = np.random.default_rng(250820)
        for height, width in ((0, 0), (0, 7), (3, 0), (1, 1), (31, 47), (820, 820)):
            for density in (0.0, 0.01, 0.5, 1.0):
                mask = rng.random((height, width)) < density
                for layout in (mask, mask.T, mask[::-2, ::-1]):
                    rows, columns = layout.shape
                    visible = np.argwhere(layout)
                    hypotheses = [visible]
                    right_rows = np.unique(visible[visible[:, 1] >= columns - 2, 0])
                    bottom_columns = np.unique(visible[visible[:, 0] >= rows - 2, 1])
                    if len(right_rows):
                        hypotheses.append(np.column_stack((
                            right_rows, np.full_like(right_rows, columns)
                        )))
                    if len(bottom_columns):
                        hypotheses.append(np.column_stack((
                            np.full_like(bottom_columns, rows), bottom_columns
                        )))
                    if len(right_rows) and len(bottom_columns):
                        hypotheses.append(np.asarray([[rows, columns]], np.int64))
                    expected = np.concatenate(hypotheses, axis=0)
                    before = layout.copy()
                    actual = DominantBackgroundSemanticizer._candidate_hypotheses(layout)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(layout, before)

    def test_sparse_classification_preserves_thresholds_and_pixel_order(self):
        rng = np.random.default_rng(820250)
        image = rng.integers(0, 256, size=(31, 47, 4), dtype=np.uint8)
        # Exercise the inclusive 16-channel difference and uint8 extremes.
        image[0, :8, :3] = np.asarray([
            [0, 0, 0], [255, 255, 255], [16, 0, 0], [17, 0, 0],
            [239, 255, 255], [238, 255, 255], [16, 0, 32], [0, 16, 17],
        ], np.uint8)
        for layout in (image, image.transpose(1, 0, 2), image[::-2, ::-1]):
            for density in (0.0, 0.01, 0.5, 1.0):
                foreground = rng.random(layout.shape[:2]) < density
                before_image = layout.copy()
                before_foreground = foreground.copy()
                actual = DominantBackgroundSemanticizer._classify_foreground(
                    layout, foreground
                )
                expected = _DenseReferenceSemanticizer._classify_foreground(
                    layout, foreground
                )
                for first, second in zip(actual, expected):
                    np.testing.assert_array_equal(first, second)
                    self.assertEqual(first.dtype, second.dtype)
                np.testing.assert_array_equal(layout, before_image)
                np.testing.assert_array_equal(foreground, before_foreground)

    def test_pairwise_distances_preserve_float32_bits(self):
        rng = np.random.default_rng(120100)
        for row_count, column_count in ((0, 5), (5, 0), (1, 1), (17, 23), (300, 384)):
            first = rng.normal(size=(row_count, 1, 2)).astype(np.float32) * 820
            second = rng.normal(size=(1, column_count, 2)).astype(np.float32) * 820
            for scale in (1.0, 1e-20, 1e10):
                left, right = first * scale, second * scale
                for left, right in ((left, right), (left[..., ::-1], right[..., ::-1])):
                    actual = DominantBackgroundSemanticizer._squared_distance_2d(
                        left, right
                    )
                    expected = _DenseReferenceSemanticizer._squared_distance_2d(
                        left, right
                    )
                    np.testing.assert_array_equal(
                        actual.view(np.uint32), expected.view(np.uint32)
                    )
        signed_zero = np.asarray([[[-0.0, 0.0]], [[0.0, -0.0]]], np.float32)
        expected = _DenseReferenceSemanticizer._squared_distance_2d(
            signed_zero, signed_zero.transpose(1, 0, 2)
        )
        actual = DominantBackgroundSemanticizer._squared_distance_2d(
            signed_zero, signed_zero.transpose(1, 0, 2)
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_full_and_predictive_detection_preserve_outputs_and_state(self):
        rng = np.random.default_rng(250384)
        for height, width, count in ((53, 71, 25), (820, 820, 300)):
            actual = DominantBackgroundSemanticizer()
            expected = _DenseReferenceSemanticizer()
            centers = rng.integers([0, 0], [height + 1, width + 1], size=(count, 2))
            centers[:6] = np.asarray([
                [0, 0], [height, width], [0, width], [height, 0],
                [height // 2, width // 2], [height // 2, width // 2 + 1],
            ])
            previous_detections = np.empty((0, 2), np.float32)
            for frame in range(12):
                background = (0, 0, 0) if frame < 6 else (17, 31, 63)
                image = np.full((height, width, 4), (*background, 255), np.uint8)
                if frame != 8:
                    for y, x in centers:
                        for dy, dx in actual._BULLET_OFFSETS:
                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                image[y + dy, x + dx, :3] = 255
                    if frame % 3:
                        image[height // 3:height // 3 + 9, width // 3:width // 3 + 9, :3] = [255, 0, 0]
                layout = image if frame % 2 else image[:, ::-1]
                before = layout.copy()
                predictions = None if frame in (0, 6) else previous_detections
                options = {
                    "include_semantic": bool(frame % 2),
                    "predicted_bullet_positions": predictions,
                    "prediction_search_radii": (
                        None if predictions is None
                        else np.full(len(predictions), 12.0, np.float32)
                    ),
                }
                first = actual.detect(layout, **options)
                second = expected.detect(layout, **options)
                np.testing.assert_array_equal(first.semantic, second.semantic)
                np.testing.assert_array_equal(
                    first.bullet_positions.view(np.uint32),
                    second.bullet_positions.view(np.uint32),
                )
                if second.plane_position is None:
                    self.assertIsNone(first.plane_position)
                else:
                    np.testing.assert_array_equal(
                        first.plane_position.view(np.uint32),
                        second.plane_position.view(np.uint32),
                    )
                for attribute in (
                    "last_detection_mode", "last_predicted_match_count",
                    "last_recovery_detection_count", "_recovery_stripe_index",
                ):
                    self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
                if expected.previous_plane is None:
                    self.assertIsNone(actual.previous_plane)
                else:
                    np.testing.assert_array_equal(actual.previous_plane, expected.previous_plane)
                np.testing.assert_array_equal(layout, before)
                previous_detections = second.bullet_positions
                centers = (centers + [1, 2]) % [height + 1, width + 1]


if __name__ == "__main__":
    unittest.main()
