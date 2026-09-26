import unittest
import copy
import numpy as np
from barrage_rl.image_oracle import BulletTrack, PersistentImageTracker
from barrage_rl.runtime_core import tracker_prediction_hints
from tools.probe_shared_predictions import shared_hints, build_shared_update
from tools.benchmark_tracker_exact import exact


class SharedPredictionTests(unittest.TestCase):
    def test_continuous_updates_keep_complete_state_and_features(self):
        from barrage_rl.tracked_policy import TrackedFeatureExtractor
        from tools.benchmark_tracker_exact import state
        rng = np.random.default_rng(92538)
        old = TrackedFeatureExtractor()
        new = copy.deepcopy(old)
        update = build_shared_update()
        positions = rng.uniform(.05, .95, (300, 2)).astype(np.float32)
        velocities = rng.normal(0, .002, (300, 2)).astype(np.float32)
        for frame in range(240):
            bullets = np.clip(positions + velocities * frame, 0, 1)
            if frame % 23 < 4:
                bullets = bullets[35:]
            if frame % 79 == 78:
                bullets = np.empty((0, 2), np.float32)
            plane = None if frame % 17 == 16 else np.array([.5 + .05*np.sin(frame), .5], np.float32)
            steps = 1 + frame % 3
            hints, predictions = shared_hints(new.tracker, (820, 820, 3), steps)
            exact(tracker_prediction_hints(old.tracker, (820, 820, 3), steps), hints)
            expected = old.step_detections(bullets, plane, decision_steps=steps)
            actual = update(new, bullets, plane, predictions, decision_steps=steps)
            exact(expected, actual)
            exact(state(old), state(new))

    def test_hints_and_source_positions_preserve_dtype_rounding_and_signed_zero(self):
        rng = np.random.default_rng(92537)
        for count in (0, 1, 300, 384):
            for dtype in (np.float16, np.float32, np.float64):
                tracker = PersistentImageTracker()
                for i in range(count):
                    position = rng.uniform(0, 820, 2).astype(dtype)
                    velocity = rng.uniform(-240, 240, 2).astype(dtype)
                    if i % 7 == 0:
                        position[:] = [-0., 0.]
                        velocity[:] = [0., -0.]
                    tracker.tracks.append(BulletTrack(i, position, velocity,
                        velocity_known=bool(i % 3), position_uncertainty=float(i % 97)))
                for steps in (-1, 0, 1, 2, 17):
                    old = tracker_prediction_hints(tracker, (720, 820, 3), steps)
                    hints, source = shared_hints(tracker, (720, 820, 3), steps)
                    exact(old, hints)
                    elapsed = tracker.decision_dt * max(1, steps)
                    expected = (np.stack([t.position + (t.velocity * elapsed if t.velocity_known else 0.)
                        for t in tracker.tracks]) if count else np.empty((0, 2), np.float32))
                    exact(expected, source)


if __name__ == '__main__':
    unittest.main()
