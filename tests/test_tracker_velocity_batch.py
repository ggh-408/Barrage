"""Scalar/batched regression equivalence, including history and gate edges."""
import copy
import unittest

import numpy as np

from barrage_rl.image_oracle import BulletTrack, PersistentImageTracker


class VelocityBatchTests(unittest.TestCase):
    def test_history_layouts_and_velocity_ownership(self):
        tracker = PersistentImageTracker()
        tracks = []
        for dtype in (np.float16, np.float32, np.float64, np.dtype('>f4')):
            for strided in (False, True):
                positions = np.zeros((8, 4), dtype=dtype)
                positions[:, 0] = np.arange(8) * 8
                points = positions[:, ::2] if strided else positions[:, :2].copy()
                start = len(tracks) * 10
                tracks.append(BulletTrack(len(tracks), np.zeros(2, np.float32),
                    np.zeros(2, np.float32), history=list(zip(range(start, start+8), points))))
        expected = copy.deepcopy(tracks)
        history_before = [[p.tobytes() for _, p in t.history] for t in tracks]
        for t in expected:
            tracker._fit_velocity(t)
        tracker._fit_velocities(tracks)
        for i, (want, got) in enumerate(zip(expected, tracks)):
            self.assertEqual(want.velocity.tobytes(), got.velocity.tobytes())
            self.assertEqual(want.velocity_known, got.velocity_known)
            self.assertTrue(got.velocity.flags.owndata)
            self.assertEqual(history_before[i], [p.tobytes() for _, p in got.history])

    def test_exact_scalar_equivalence(self):
        rng = np.random.default_rng(745)
        for refit in (True, False):
            tracker = PersistentImageTracker(refit_known_velocity=refit)
            for batch in range(20):
                tracks = []
                for i in range(384):
                    length = int(rng.integers(1, 13))
                    times = np.cumsum(rng.integers(1, 4, length)) + batch * 100
                    if i % 17 == 0:
                        times[:] = times[0]
                    positions = rng.uniform(0, 820, (length, 2)).astype(np.float32)
                    if i % 11 == 0:
                        positions[:] = positions[0]
                    track = BulletTrack(i, positions[-1].copy(), rng.normal(size=2).astype(np.float32))
                    track.history = list(zip(map(int, times), positions))
                    track.velocity_known = bool(i % 2)
                    tracks.append(track)
                actual = copy.deepcopy(tracks)
                cache = {}
                for track in tracks:
                    tracker._fit_velocity(track, regression_cache=cache)
                tracker._fit_velocities(actual)
                for expected, got in zip(tracks, actual):
                    np.testing.assert_array_equal(got.velocity, expected.velocity)
                    self.assertEqual(got.velocity_known, expected.velocity_known)


if __name__ == '__main__':
    unittest.main()
