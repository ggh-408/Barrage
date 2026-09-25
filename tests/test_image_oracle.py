"""Image-only state reconstruction tests."""

import unittest

import numpy as np

from barrage_rl.env import BarrageVisionEnv
from barrage_rl.image_oracle import ImageOnlyPlannerAgent, PersistentImageTracker
from barrage_rl.runtime_core import OPENING_BATCH_COUNT


class PersistentImageTrackerTests(unittest.TestCase):
    def test_equal_distance_preserves_identity_and_independent_history(self):
        tracker = PersistentImageTracker()
        plane = np.array([410., 410.], np.float32)
        tracker.initialize_detections(np.array([[99., 100.], [101., 100.]], np.float32),
                                      plane, normalized=False)
        # Retention has already ordered identities by threat. Distance ties
        # must preserve that input order, including when IDs are reordered.
        first, second = [track.track_id for track in tracker.tracks]
        detection = np.array([[100., 100.]], np.float32)
        tracker.update_detections(detection, plane, normalized=False)
        tracks = {track.track_id: track for track in tracker.tracks}
        self.assertEqual(len(tracks[first].history), 2)
        self.assertEqual(len(tracks[second].history), 1)
        self.assertEqual(tracks[second].occluded_steps, 1)
        self.assertFalse(np.shares_memory(tracks[first].position, tracks[first].history[-1][1]))
        detection[:] = -100.
        np.testing.assert_array_equal(tracks[first].position, [100., 100.])
        np.testing.assert_array_equal(tracks[first].history[-1][1], [100., 100.])

    def test_dense_occlusion_preserves_multiple_image_derived_tracks(self) -> None:
        tracker = PersistentImageTracker(
            target_track_count=384, expected_bullet_count=300
        )
        tracker.initialize_detections(
            np.asarray([[0.495, 0.50], [0.505, 0.50]], np.float32),
            np.asarray([0.50, 0.50], np.float32),
        )
        tracker.update_detections(
            np.asarray([[0.50, 0.50]], np.float32),
            np.asarray([0.50, 0.50], np.float32),
        )
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.last_detection_count, 1)
        self.assertEqual(tracker.last_ambiguous_track_count, 1)
        self.assertGreater(tracker.occluded_fraction, 0.0)
        self.assertGreater(tracker.count_deficit_fraction, 0.99)

    def test_default_tracker_has_headroom_for_300_bullets(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            observation_size=384,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=1_800_000)
            for _ in range(108):
                env._move_bullets()
                env.physics_steps += 1
            frame = env._make_frame()
            observation = np.stack([frame] * env.frame_stack)
            tracker = PersistentImageTracker()
            tracker.initialize(observation)
            self.assertGreaterEqual(len(tracker.tracks), 280)
            self.assertGreaterEqual(tracker.target_track_count, 384)
            self.assertLessEqual(len(tracker.tracks), tracker.target_track_count)
        finally:
            env.close()

    def test_tracks_moving_300_bullet_scene_without_state_input(self) -> None:
        known_velocity_fractions: list[float] = []
        for seed in range(1_800_000, 1_800_005):
            env = BarrageVisionEnv(
                bullet_count=300,
                bullet_size=5,
                bullet_speed_min=240.0,
                bullet_speed_max=240.0,
                targeted_bullet_probability=0.10,
                observation_size=192,
                randomize_initial_phase=False,
            )
            try:
                observation, _ = env.reset(seed=seed)
                tracker = PersistentImageTracker(
                    target_track_count=384, expected_bullet_count=300
                )
                tracker.initialize(observation)
                for _ in range(32):
                    for _ in range(env.action_repeat):
                        env._move_bullets()
                        env.physics_steps += 1
                    env.frames.append(env._make_frame())
                    observation = env._get_observation()
                    tracker.update(observation)

                self.assertGreaterEqual(len(tracker.tracks), 288)
                known_velocity_fractions.append(tracker.known_velocity_fraction)
                known_speeds = np.asarray([
                    np.linalg.norm(track.velocity)
                    for track in tracker.tracks
                    if track.velocity_known
                ])
                np.testing.assert_allclose(known_speeds, 240.0, atol=1e-3)

                tracked = np.stack([track.position for track in tracker.tracks])
                nearest = np.linalg.norm(
                    tracked[:, None, :] - env.bullet_positions[None, :, :], axis=2
                ).min(axis=1)
                self.assertLess(float(np.median(nearest)), 5.0)
            finally:
                env.close()
        self.assertGreaterEqual(float(np.mean(known_velocity_fractions)), 0.90)

    def test_detection_updates_preserve_elapsed_decision_timestamps(self) -> None:
        tracker = PersistentImageTracker(source_size=820.0, bullet_speed=240.0)
        plane = np.asarray([0.5, 0.5], np.float32)
        tracker.initialize_detections(
            np.asarray([[100.0 / 820.0, 200.0 / 820.0]], np.float32), plane
        )
        for x in (121.0, 142.0, 163.0):
            tracker.update_detections(
                np.asarray([[x / 820.0, 200.0 / 820.0]], np.float32),
                plane,
                decision_steps=3,
            )
        self.assertEqual(
            [step for step, _ in tracker.tracks[0].history], [0, 3, 6, 9]
        )
        self.assertTrue(tracker.tracks[0].velocity_known)

    def test_image_belief_marks_observed_field_opening_complete(self) -> None:
        observation = np.zeros((4, 96, 96), dtype=np.uint8)
        observation[:, 48, 48] = BarrageVisionEnv.PLANE_INTENSITY
        for index in range(20):
            x = 5 + 4 * (index % 10)
            y = 8 + 10 * (index // 10)
            observation[:, y, x] = BarrageVisionEnv.BULLET_INTENSITY

        agent = ImageOnlyPlannerAgent(
            observation_size=96,
            bullet_count=20,
            planner_kind="recovery",
        )
        try:
            agent.reset(observation)
            agent._synchronize_belief(observation)
            observed_count = len(agent._belief.bullet_positions)
            self.assertEqual(observed_count, len(agent.tracker.tracks))
            self.assertEqual(
                agent._belief.opening_spawned_batches,
                OPENING_BATCH_COUNT,
            )

            agent._belief._append_due_opening_batches(10_000)
            self.assertEqual(len(agent._belief.bullet_positions), observed_count)
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
