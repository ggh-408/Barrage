import copy
import unittest
import numpy as np
from barrage_rl.image_oracle import BulletTrack, PersistentImageTracker


class BatchedRetentionTests(unittest.TestCase):
    def test_scores_and_stable_inplace_order(self):
        rng = np.random.default_rng(9013)
        for _ in range(30):
            tracker = PersistentImageTracker()
            tracker.plane_velocity = rng.uniform(-240, 240, 2).astype(np.float32)
            tracks = [BulletTrack(i, rng.uniform(0, 820, 2).astype(np.float32),
                                  rng.uniform(-240, 240, 2).astype(np.float32),
                                  velocity_known=bool(i % 3), confidence=float(rng.random()))
                      for i in range(384)]
            tracks[1].position = tracks[0].position.copy()
            tracks[1].velocity = tracks[0].velocity.copy()
            tracks[1].velocity_known = tracks[0].velocity_known
            tracks[1].confidence = tracks[0].confidence
            expected = np.asarray([tracker._retention_threat(t) for t in tracks])
            self.assertEqual(expected.tobytes(), tracker._retention_threats(tracks).tobytes())
            reference = copy.deepcopy(tracks)
            tracker._trim_tracks_scalar(reference)
            tracker._trim_tracks(tracks)
            self.assertEqual([t.track_id for t in reference], [t.track_id for t in tracks])

    def test_override_and_empty(self):
        class CustomTracker(PersistentImageTracker):
            def _retention_threat(self, track):
                return float(track.track_id)
        tracker = CustomTracker()
        self.assertEqual(tracker._trim_tracks([]), [])
        tracks = [BulletTrack(i, np.zeros(2, np.float32), np.zeros(2, np.float32)) for i in range(4)]
        self.assertEqual([t.track_id for t in tracker._trim_tracks(tracks)], [3, 2, 1, 0])
