"""Shared-memory tracked DAgger pipeline smoke test."""

import unittest

import numpy as np

from barrage_rl.baselines import privileged_planner_supervision
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.tracked_collection import (
    ParallelTrackedDaggerEnv,
    _RenderedRGBObservation,
    _delayed_planner_supervision,
)
from barrage_rl.tracked_policy import TrackedFeatureExtractor, TrackedPolicySpec


class TrackedCollectionTests(unittest.TestCase):
    def test_repeated_initial_seeds_require_an_explicit_seed_list(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "requires initial_episode_seeds"
        ):
            ParallelTrackedDaggerEnv(
                env_count=1,
                workers=1,
                seed=1,
                initial_episode_seeds=(),
                repeat_initial_episode_seeds=True,
                env_kwargs={},
                spec=TrackedPolicySpec(),
            )

    def test_repeated_initial_seed_survives_multiple_worker_resets(self) -> None:
        with ParallelTrackedDaggerEnv(
            env_count=1,
            workers=1,
            seed=7,
            initial_episode_seeds=(12345,),
            repeat_initial_episode_seeds=True,
            env_kwargs={
                "bullet_count": 300,
                "targeted_bullet_probability": 0.10,
                "observation_size": 192,
                "max_episode_seconds": 0.10,
                "randomize_initial_phase": False,
            },
            spec=TrackedPolicySpec(
                max_objects=384,
                tracker_capacity=384,
                expected_bullet_count=300,
            ),
            teacher_kind="exact",
        ) as pipeline:
            for _ in range(10):
                pipeline.step(pipeline.teacher_actions.copy())
            self.assertGreaterEqual(int(pipeline.episode_serial[0]), 2)

    def test_rendered_training_path_uses_live_predictive_detection(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            observation_size=192,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=1771)
            rendered = _RenderedRGBObservation(env, env.observation_size)
            extractor = TrackedFeatureExtractor(TrackedPolicySpec())
            first = rendered.detections(True)
            extractor.reset_detections(
                first.bullet_positions, first.plane_position
            )
            env.step(0)
            rendered.detections(False, extractor=extractor)
            self.assertEqual(rendered.semanticizer.last_detection_mode, "predictive")
            self.assertGreater(
                rendered.semanticizer.last_predicted_match_count, 0
            )
        finally:
            env.close()

    def test_delayed_teacher_labels_next_boundary_without_mutating_env(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=901)
            env.physics_steps = 8
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.bullet_is_targeted.fill(False)
            original = env.capture_state()
            env.simulate_action(
                2,
                include_scheduled_opening=False,
                include_respawns=False,
            )
            expected = privileged_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                wall_margin=80.0,
                wall_penalty_weight=0.35,
                safety_horizons=(0.10,),
            )
            env.restore_state(original)
            actual = _delayed_planner_supervision(
                env,
                2,
                teacher_kind="exact",
                teacher_horizon_seconds=0.10,
                teacher_reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            restored = env.capture_state()
            self.assertEqual(actual.action, expected.action)
            np.testing.assert_allclose(actual.regrets, expected.regrets)
            np.testing.assert_array_equal(
                actual.safety_targets, expected.safety_targets
            )
            np.testing.assert_array_equal(
                restored.plane_position, original.plane_position
            )
            np.testing.assert_array_equal(
                restored.bullet_positions, original.bullet_positions
            )
            self.assertEqual(restored.rng_state, original.rng_state)
        finally:
            env.close()

    def test_one_step_delay_applies_initial_stay_then_previous_command(self) -> None:
        env_kwargs = {
            "bullet_count": 300,
            "bullet_size": 5,
            "bullet_speed_min": 240.0,
            "bullet_speed_max": 240.0,
            "targeted_bullet_probability": 0.10,
            "observation_size": 96,
            "max_episode_seconds": 2.0,
            "randomize_initial_phase": False,
        }
        spec = TrackedPolicySpec()
        with ParallelTrackedDaggerEnv(
            env_count=1,
            workers=1,
            seed=812,
            env_kwargs=env_kwargs,
            spec=spec,
            safety_horizons=(0.10,),
            teacher_kind="exact",
            teacher_horizon_seconds=0.10,
            teacher_reaction_seconds=0.10,
            deployment_rgb_observation=True,
            causal_action_delay_steps=1,
        ) as pipeline:
            initial_x = float(pipeline.globals[0, 0])
            pipeline.step(np.asarray([2], dtype=np.int64))
            after_initial_stay = float(pipeline.globals[0, 0])
            pipeline.step(np.asarray([2], dtype=np.int64))
            after_delayed_right = float(pipeline.globals[0, 0])
        self.assertAlmostEqual(after_initial_stay, initial_x, places=6)
        self.assertGreater(after_delayed_right, after_initial_stay)

    def test_explicit_initial_seed_overrides_base_seed(self) -> None:
        env_kwargs = {
            "bullet_count": 300,
            "observation_size": 96,
            "max_episode_seconds": 1.0,
        }
        snapshots = []
        for base_seed in (10, 20):
            with ParallelTrackedDaggerEnv(
                env_count=1,
                workers=1,
                seed=base_seed,
                initial_episode_seeds=(12345,),
                env_kwargs=env_kwargs,
                spec=TrackedPolicySpec(),
            ) as pipeline:
                snapshots.append(pipeline.objects.copy())
        np.testing.assert_array_equal(snapshots[0], snapshots[1])

    def test_parallel_pipeline_publishes_features_and_teacher_labels(self) -> None:
        env_kwargs = {
            "bullet_count": 300,
            "bullet_size_min": 5,
            "bullet_size_max": 5,
            "bullet_speed_min": 240.0,
            "bullet_speed_max": 240.0,
            "targeted_bullet_probability": 0.10,
            "observation_size": 192,
            "max_episode_seconds": 5.0,
            "randomize_initial_phase": False,
        }
        with ParallelTrackedDaggerEnv(
            env_count=2,
            workers=2,
            seed=2_100_000,
            env_kwargs=env_kwargs,
            spec=TrackedPolicySpec(),
        ) as pipeline:
            spec = TrackedPolicySpec()
            self.assertEqual(
                pipeline.objects.shape,
                (2, spec.max_objects, spec.object_features),
            )
            self.assertEqual(pipeline.collisions.shape, (2, 4, 9))
            self.assertTrue(np.isfinite(pipeline.objects).all())
            self.assertTrue(np.isfinite(pipeline.regrets).all())
            self.assertTrue(np.all((pipeline.teacher_actions >= 0) & (pipeline.teacher_actions < 9)))
            np.testing.assert_array_equal(pipeline.episode_steps, np.zeros(2))
            done, truncated, survival = pipeline.step(pipeline.teacher_actions.copy())
            self.assertEqual(done.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
            self.assertEqual(survival.shape, (2,))
            np.testing.assert_array_equal(pipeline.episode_steps, np.ones(2))

    def test_exact_teacher_pipeline_publishes_real_dynamics_labels(self) -> None:
        env_kwargs = {
            "bullet_count": 300,
            "bullet_size": 5,
            "bullet_speed_min": 240.0,
            "bullet_speed_max": 240.0,
            "targeted_bullet_probability": 0.10,
            "observation_size": 192,
            "max_episode_seconds": 1.0,
            "randomize_initial_phase": False,
        }
        with ParallelTrackedDaggerEnv(
            env_count=1,
            workers=1,
            seed=2_100_000,
            env_kwargs=env_kwargs,
            spec=TrackedPolicySpec(),
            safety_horizons=(0.10,),
            teacher_kind="exact",
            teacher_horizon_seconds=0.10,
            teacher_reaction_seconds=0.10,
        ) as pipeline:
            self.assertEqual(pipeline.collisions.shape, (1, 1, 9))
            self.assertTrue(np.isfinite(pipeline.regrets).all())
            pipeline.step(pipeline.teacher_actions.copy())

    def test_action_repeat_randomization_pipeline_smoke(self) -> None:
        env_kwargs = {
            "bullet_count": 300,
            "observation_size": 96,
            "max_episode_seconds": 1.0,
            "randomize_initial_phase": False,
        }
        with ParallelTrackedDaggerEnv(
            env_count=1,
            workers=1,
            seed=123,
            env_kwargs=env_kwargs,
            spec=TrackedPolicySpec(),
            action_repeat_choices=(3, 4, 5),
        ) as pipeline:
            pipeline.step(pipeline.teacher_actions.copy())
            np.testing.assert_array_equal(pipeline.episode_steps, np.ones(1))

    def test_action_repeat_randomization_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integers"):
            ParallelTrackedDaggerEnv(
                env_count=1,
                workers=1,
                seed=123,
                env_kwargs={"bullet_count": 300, "observation_size": 96},
                spec=TrackedPolicySpec(),
                action_repeat_choices=(0, 4),
            )

    def test_collection_rejects_more_than_one_delay_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or one"):
            ParallelTrackedDaggerEnv(
                env_count=1,
                workers=1,
                seed=123,
                env_kwargs={"bullet_count": 300, "observation_size": 96},
                spec=TrackedPolicySpec(),
                causal_action_delay_steps=2,
            )


if __name__ == "__main__":
    unittest.main()
