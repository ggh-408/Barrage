"""Correctness and performance invariants for vectorized recovery labels."""

import time
import unittest

import numpy as np

from barrage_rl.env import BarrageVisionEnv
from barrage_rl.recovery_planner import vectorized_recovery_supervision
from barrage_rl.runtime_core import OPENING_BATCH_COUNT


class RecoveryPlannerTests(unittest.TestCase):
    def test_sparse_opening_stays_when_safe_and_moves_for_clear_risk(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=1_600_100)
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.bullet_is_targeted.fill(False)
            env.plane_position[:] = (410.0, 410.0)
            env.plane_velocity[:] = (240.0, 0.0)
            env.opening_spawned_batches = 1

            safe = vectorized_recovery_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            self.assertEqual(safe.action, 0)

            env.bullet_positions[0] = (410.0, 380.0)
            env.bullet_velocities[0] = (0.0, 240.0)
            risky = vectorized_recovery_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            self.assertEqual(risky.safety_targets[0, 0], 1.0)
            self.assertNotEqual(risky.action, 0)

            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.opening_spawned_batches = OPENING_BATCH_COUNT
            full = vectorized_recovery_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            self.assertEqual(full.action, 2)
        finally:
            env.close()

    def test_one_bullet_opening_has_finite_recovery_labels(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=1_600_101)
            self.assertEqual(len(env.bullet_positions), 1)
            result = vectorized_recovery_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            self.assertTrue(np.isfinite(result.action_costs).all())
            self.assertTrue(np.isfinite(result.regrets).all())
        finally:
            env.close()

    def test_supervision_is_finite_and_does_not_mutate_environment(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=1_600_000)
            before = env.capture_state()
            started = time.perf_counter()
            result = vectorized_recovery_supervision(
                env, horizon_seconds=1.5, safety_horizons=(0.1, 0.3, 0.6, 1.2)
            )
            elapsed = time.perf_counter() - started
            after = env.capture_state()

            self.assertIn(result.action, range(len(env.ACTIONS)))
            self.assertEqual(result.regrets.shape, (len(env.ACTIONS),))
            self.assertEqual(result.safety_targets.shape, (4, len(env.ACTIONS)))
            self.assertTrue(np.isfinite(result.action_costs).all())
            self.assertLess(elapsed, 1.0)
            np.testing.assert_array_equal(after.plane_position, before.plane_position)
            np.testing.assert_array_equal(after.plane_velocity, before.plane_velocity)
            np.testing.assert_array_equal(after.bullet_positions, before.bullet_positions)
            np.testing.assert_array_equal(after.bullet_velocities, before.bullet_velocities)
            self.assertEqual(after.episode_steps, before.episode_steps)
            self.assertEqual(after.physics_steps, before.physics_steps)
            self.assertEqual(after.rng_state, before.rng_state)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
