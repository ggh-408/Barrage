"""Retention score and track-order regressions against scalar NumPy clipping."""

import copy
import unittest

import numpy as np

from barrage_rl.image_oracle import BulletTrack, PersistentImageTracker


class _ClipReferenceTracker(PersistentImageTracker):
    def _retention_threat(self, track):
        relative = track.position - self.plane_position
        if not track.velocity_known:
            distance = float(np.linalg.norm(relative))
            return float(np.exp(-distance / 180.0) * track.confidence)
        relative_velocity = track.velocity - self.plane_velocity
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        ttc = float(np.clip(
            -np.dot(relative, relative_velocity) / max(speed_squared, 1e-6),
            0.0,
            3.0,
        ))
        closest = relative + relative_velocity * ttc
        clearance = max(float(np.linalg.norm(closest)) - 11.5, 0.0)
        return float(
            np.exp(-clearance / 42.0)
            * np.exp(-ttc / 1.5)
            * max(track.confidence, 0.15)
        )


class RetentionThreatEquivalenceTests(unittest.TestCase):
    def assert_float_bits_equal(self, actual, expected):
        self.assertEqual(
            np.float64(actual).view(np.uint64),
            np.float64(expected).view(np.uint64),
        )

    def test_retention_preserves_clip_boundaries_and_nonfinite_inputs(self):
        actual = PersistentImageTracker()
        expected = _ClipReferenceTracker()
        for tracker in (actual, expected):
            tracker.plane_position = np.zeros(2, np.float64)
            tracker.plane_velocity = np.zeros(2, np.float64)
        times = [
            -np.inf, -10.0, -0.0, 0.0, 1.0, 3.0, 10.0, np.inf,
            np.nextafter(0.0, -np.inf), np.nextafter(0.0, np.inf),
            np.nextafter(3.0, -np.inf), np.nextafter(3.0, np.inf),
            np.asarray([0x7ff8000000000000, 0xfff8000000000000,
                        0x7ff0000000000001, 0xffffffffffffffff], np.uint64).view(np.float64),
        ]
        values = np.concatenate((np.asarray(times[:-1]), times[-1]))
        with np.errstate(all="ignore"):
            for index, value in enumerate(values):
                track = BulletTrack(
                    track_id=index,
                    position=np.asarray([-value, 0.0], np.float64),
                    velocity=np.asarray([1.0, 0.0], np.float64),
                    velocity_known=True,
                    confidence=0.7,
                )
                self.assert_float_bits_equal(
                    actual._retention_threat(track),
                    expected._retention_threat(track),
                )

    def test_retention_preserves_random_scores_and_trim_order(self):
        rng = np.random.default_rng(384250100)
        for case in range(4):
            actual = PersistentImageTracker()
            expected = _ClipReferenceTracker()
            for tracker in (actual, expected):
                tracker.plane_position = np.asarray([410.0, 410.0], np.float32)
                tracker.plane_velocity = np.asarray([120.0, -180.0], np.float32)
            tracks = []
            for index in range(600):
                position = rng.uniform(-50.0, 870.0, size=2).astype(np.float32)
                velocity = rng.uniform(-240.0, 240.0, size=2).astype(np.float32)
                if index % 7 == 0:
                    velocity = actual.plane_velocity.copy()
                if index % 11 == 0:
                    position = actual.plane_position.copy()
                track = BulletTrack(
                    track_id=index,
                    position=position,
                    velocity=velocity,
                    velocity_known=bool(index % 3),
                    confidence=float(rng.uniform(0.0, 1.0)),
                    missed=index % 5,
                    occluded_steps=index % 2,
                )
                self.assert_float_bits_equal(
                    actual._retention_threat(track),
                    expected._retention_threat(track),
                )
                tracks.append(track)
            # Exact duplicate scores exercise Python's existing stable tie order.
            for index in range(20):
                duplicate = copy.deepcopy(tracks[index])
                duplicate.track_id = len(tracks)
                tracks.append(duplicate)
            actual_ids = [track.track_id for track in actual._trim_tracks(list(tracks))]
            expected_ids = [track.track_id for track in expected._trim_tracks(list(tracks))]
            self.assertEqual(actual_ids, expected_ids, case)


if __name__ == "__main__":
    unittest.main()
