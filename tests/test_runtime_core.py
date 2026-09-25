"""Regression coverage for the current tracked-policy runtime core."""

from copy import deepcopy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pygame

from Barrage import Barrage as GameBarrage, FIXED_SCREEN_HEIGHT, FIXED_SCREEN_WIDTH
from barrage_rl.baselines import (
    _action_change_cost,
    fast_planner_supervision,
    privileged_planner_supervision,
)
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.runtime_core import (
    OPENING_BATCH_COUNT,
    OPENING_BATCH_INTERVAL_SECONDS,
    opening_batch_size,
    spawn_bullet,
    spawn_bullets,
)
from barrage_rl.live_screen import DominantBackgroundSemanticizer
from barrage_rl.tracked_collection import _RenderedRGBObservation
from barrage_rl.metrics import (
    bootstrap_confidence_intervals,
    interquartile_mean,
    lower_tail_mean,
    wilson_lower_bound,
)
from barrage_rl.plot import (
    _dagger_best_index,
    _dagger_best_metric,
    _set_middle_80_ylim,
    _set_reliability_ylim,
    save_round_summary_plot,
)


class RuntimeCoreTests(unittest.TestCase):
    def test_runtime_geometry_and_rates_match_deployment(self) -> None:
        self.assertEqual((FIXED_SCREEN_WIDTH, FIXED_SCREEN_HEIGHT), (820, 820))
        self.assertEqual(GameBarrage.PHYSICS_FPS, 120)
        self.assertEqual(GameBarrage.PLANE_SPEED, 240.0)
        self.assertEqual(GameBarrage.BULLET_SPEED, 240.0)
        self.assertEqual(GameBarrage.OPENING_BATCH_COUNT, 10)
        self.assertEqual(GameBarrage.OPENING_BATCH_INTERVAL_SECONDS, 0.1)

    def test_real_game_reset_uses_first_opening_batch(self) -> None:
        pygame.display.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1))
        previous = {
            "quantity": GameBarrage.QUANTITY,
            "bullet_size": GameBarrage.BULLET_SIZE,
            "bullet_speed": GameBarrage.BULLET_SPEED,
            "targeted": GameBarrage.TARGETED_BULLET_PROBABILITY,
            "collision": GameBarrage.COLLISION,
            "rng": GameBarrage.RNG,
        }
        try:
            GameBarrage.QUANTITY = 300
            GameBarrage.BULLET_SIZE = 5
            GameBarrage.BULLET_SPEED = 240.0
            GameBarrage.TARGETED_BULLET_PROBABILITY = 0.10
            GameBarrage.COLLISION = False
            GameBarrage.RNG = np.random.default_rng(321)
            GameBarrage.reset_game()
            positions = np.asarray(
                [bullet[:2] for bullet in GameBarrage.BULLET.LIST],
                dtype=np.float32,
            )
            self.assertEqual(len(positions), 30)
            on_edge = (
                (positions[:, 0] == 0.0)
                | (positions[:, 0] == GameBarrage.SCREEN_WIDTH)
                | (positions[:, 1] == 0.0)
                | (positions[:, 1] == GameBarrage.SCREEN_HEIGHT)
            )
            self.assertTrue(np.all(on_edge))
            self.assertEqual(GameBarrage.OPENING_SPAWNED_BATCHES, 1)
            self.assertEqual(GameBarrage.OPENING_EFFECTIVE_PHASE_SECONDS, 0.9)
        finally:
            GameBarrage.QUANTITY = previous["quantity"]
            GameBarrage.BULLET_SIZE = previous["bullet_size"]
            GameBarrage.BULLET_SPEED = previous["bullet_speed"]
            GameBarrage.TARGETED_BULLET_PROBABILITY = previous["targeted"]
            GameBarrage.COLLISION = previous["collision"]
            GameBarrage.RNG = previous["rng"]

    def test_deployment_opening_is_deterministic_and_batched(self) -> None:
        kwargs = {
            "bullet_count": 300,
            "bullet_size": 5,
            "bullet_speed_min": 240.0,
            "bullet_speed_max": 240.0,
            "targeted_bullet_probability": 0.10,
            "randomize_initial_phase": False,
        }
        first = BarrageVisionEnv(**kwargs)
        second = BarrageVisionEnv(**kwargs)
        try:
            _, info = first.reset(seed=812)
            second.reset(seed=812)
            np.testing.assert_array_equal(first.bullet_positions, second.bullet_positions)
            np.testing.assert_array_equal(first.bullet_velocities, second.bullet_velocities)
            self.assertEqual(len(first.bullet_positions), 30)
            self.assertEqual(info["opening_effective_phase_seconds"], 0.9)
            self.assertEqual(info["opening_spawned_batches"], 1)
        finally:
            first.close()
            second.close()

    def test_counterfactual_can_exclude_unobserved_opening_batches(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            bullet_size_min=5,
            bullet_size_max=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=4112)
            env.physics_steps = 8
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.bullet_is_targeted.fill(False)
            before = env.capture_state()
            initial_count = len(env.bullet_positions)

            collided = env.simulate_action(
                0,
                include_scheduled_opening=False,
            )
            self.assertFalse(collided)
            self.assertEqual(env.physics_steps, 12)
            self.assertEqual(len(env.bullet_positions), initial_count)
            self.assertEqual(env.opening_spawned_batches, 1)
            self.assertEqual(env.np_random.bit_generator.state, before.rng_state)

            env.restore_state(before)
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(len(env.bullet_positions), initial_count + 30)
            self.assertEqual(env.opening_spawned_batches, 2)
        finally:
            env.close()

    def test_counterfactual_can_exclude_unobserved_respawns(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1,
            bullet_size_min=5,
            bullet_size_max=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=4113)
            env.opening_spawned_batches = OPENING_BATCH_COUNT
            env.bullet_positions[0] = (-1.0, 410.0)
            env.bullet_velocities[0] = (-240.0, 0.0)
            env.bullet_is_targeted[0] = False
            before = env.capture_state()

            collided = env.simulate_action(
                0,
                include_scheduled_opening=False,
                include_respawns=False,
            )
            self.assertFalse(collided)
            self.assertLess(float(env.bullet_positions[0, 0]), -1.0)
            self.assertEqual(env.np_random.bit_generator.state, before.rng_state)

            env.restore_state(before)
            collided = env.simulate_action(0)
            self.assertFalse(collided)
            self.assertNotEqual(
                env.np_random.bit_generator.state,
                before.rng_state,
            )
            self.assertGreaterEqual(float(env.bullet_positions[0, 0]), 0.0)
        finally:
            env.close()

    def test_opening_batches_fill_the_last_batch_with_the_remainder(self) -> None:
        production_sizes = [
            opening_batch_size(300, index, OPENING_BATCH_COUNT)
            for index in range(OPENING_BATCH_COUNT)
        ]
        self.assertEqual(production_sizes, [30] * OPENING_BATCH_COUNT)

        sizes = [
            opening_batch_size(303, index, OPENING_BATCH_COUNT)
            for index in range(OPENING_BATCH_COUNT)
        ]
        self.assertEqual(sizes, [30] * 9 + [33])
        self.assertEqual(sum(sizes), 303)

        env = BarrageVisionEnv(
            bullet_count=303,
            bullet_size_min=5,
            bullet_size_max=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=1901)
            self.assertEqual(len(env.bullet_positions), 30)
            for elapsed_step in range(1, 109):
                env._move_bullets()
                env.physics_steps += 1
                if elapsed_step % 12 == 0:
                    completed_batches = min(10, 1 + elapsed_step // 12)
                    expected = sum(sizes[:completed_batches])
                    self.assertEqual(len(env.bullet_positions), expected)
            self.assertEqual(len(env.bullet_positions), 303)
            self.assertEqual(env.opening_spawned_batches, 10)
        finally:
            env.close()

    def test_tiny_openings_spawn_immediately_and_keep_the_total(self) -> None:
        for total_count in range(1, OPENING_BATCH_COUNT):
            sizes = [
                opening_batch_size(total_count, index, OPENING_BATCH_COUNT)
                for index in range(OPENING_BATCH_COUNT)
            ]
            self.assertEqual(sizes[:total_count], [1] * total_count)
            self.assertEqual(
                sizes[total_count:],
                [0] * (OPENING_BATCH_COUNT - total_count),
            )
            self.assertEqual(sum(sizes), total_count)

        for total_count in (1, OPENING_BATCH_COUNT - 1):
            env = BarrageVisionEnv(
                bullet_count=total_count,
                bullet_size_min=5,
                bullet_size_max=5,
                bullet_speed_min=240.0,
                bullet_speed_max=240.0,
                randomize_initial_phase=False,
            )
            try:
                env.reset(seed=9000 + total_count)
                self.assertEqual(len(env.bullet_positions), 1)
                exact = privileged_planner_supervision(
                    env,
                    horizon_seconds=0.10,
                    reaction_seconds=0.10,
                    safety_horizons=(0.10,),
                )
                self.assertTrue(np.isfinite(exact.action_costs).all())
            finally:
                env.close()

    def test_real_game_and_environment_share_seed_and_respawn_order(self) -> None:
        pygame.display.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1))
        previous = {
            "quantity": GameBarrage.QUANTITY,
            "bullet_size": GameBarrage.BULLET_SIZE,
            "bullet_speed": GameBarrage.BULLET_SPEED,
            "targeted": GameBarrage.TARGETED_BULLET_PROBABILITY,
            "collision": GameBarrage.COLLISION,
            "invincible": GameBarrage.INVINCIBLE,
            "key": GameBarrage.KEY,
            "rng": GameBarrage.RNG,
            "controller": GameBarrage.AI_CONTROLLER,
            "window": GameBarrage.window,
        }
        env = BarrageVisionEnv(
            bullet_count=300,
            bullet_size_min=5,
            bullet_size_max=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            GameBarrage.QUANTITY = 300
            GameBarrage.BULLET_SIZE = 5
            GameBarrage.BULLET_SPEED = 240.0
            GameBarrage.TARGETED_BULLET_PROBABILITY = 0.10
            GameBarrage.COLLISION = False
            GameBarrage.INVINCIBLE = False
            GameBarrage.KEY = True
            GameBarrage.AI_CONTROLLER = None
            GameBarrage.RNG = np.random.default_rng(4812)
            GameBarrage.reset_game()
            env.reset(seed=4812)

            game_bullets = np.asarray(GameBarrage.BULLET.LIST)
            np.testing.assert_array_equal(
                game_bullets[:, :2].astype(np.float32), env.bullet_positions
            )
            np.testing.assert_array_equal(
                game_bullets[:, 2:4].astype(np.float32), env.bullet_velocities
            )
            np.testing.assert_array_equal(
                game_bullets[:, 4].astype(np.bool_), env.bullet_is_targeted
            )
            self.assertEqual(
                GameBarrage.RNG.bit_generator.state,
                env.np_random.bit_generator.state,
            )

            rendered = _RenderedRGBObservation(env, env.observation_size)
            rendered._frame()
            GameBarrage.window = pygame.Surface(
                (GameBarrage.SCREEN_WIDTH, GameBarrage.SCREEN_HEIGHT)
            )
            GameBarrage.render_world()
            np.testing.assert_array_equal(
                pygame.surfarray.array3d(GameBarrage.window),
                pygame.surfarray.array3d(rendered.surface),
            )

            for index in (2, 7):
                GameBarrage.BULLET.LIST[index][0] = -1.0
                env.bullet_positions[index, 0] = -1.0
            GameBarrage.BULLET.update(
                GameBarrage.PLANE, 1.0 / GameBarrage.PHYSICS_FPS
            )
            env._move_bullets()
            GameBarrage.ALIVE_PHYSICS_STEPS += 1
            env.physics_steps += 1

            game_bullets = np.asarray(GameBarrage.BULLET.LIST)
            np.testing.assert_allclose(
                game_bullets[:, :2], env.bullet_positions, atol=1e-5, rtol=0.0
            )
            np.testing.assert_array_equal(
                game_bullets[:, 2:4].astype(np.float32), env.bullet_velocities
            )
            np.testing.assert_array_equal(
                game_bullets[:, 4].astype(np.bool_), env.bullet_is_targeted
            )
            self.assertEqual(
                GameBarrage.RNG.bit_generator.state,
                env.np_random.bit_generator.state,
            )

            for physics_step in range(1_200):
                direction = env.ACTIONS[(physics_step // 19) % len(env.ACTIONS)]
                GameBarrage.PLANE.move(
                    direction, 1.0 / GameBarrage.PHYSICS_FPS
                )
                GameBarrage.BULLET.update(
                    GameBarrage.PLANE, 1.0 / GameBarrage.PHYSICS_FPS
                )
                env._move_plane(direction)
                env._move_bullets()
                GameBarrage.ALIVE_PHYSICS_STEPS += 1
                env.physics_steps += 1
                np.testing.assert_array_equal(
                    np.asarray(GameBarrage.PLANE.position, dtype=np.float32),
                    env.plane_position,
                    err_msg=f"plane position at physics step {physics_step}",
                )
                np.testing.assert_array_equal(
                    np.asarray(GameBarrage.PLANE.velocity, dtype=np.float32),
                    env.plane_velocity,
                    err_msg=f"plane velocity at physics step {physics_step}",
                )
                np.testing.assert_array_equal(
                    np.asarray(GameBarrage.BULLET.LIST)[:, 2:4].astype(np.float32),
                    env.bullet_velocities,
                    err_msg=f"bullet velocity at physics step {physics_step}",
                )
                np.testing.assert_array_equal(
                    np.asarray(GameBarrage.BULLET.LIST)[:, :2].astype(np.float32),
                    env.bullet_positions,
                    err_msg=f"physics step {physics_step}",
                )

            game_bullets = np.asarray(GameBarrage.BULLET.LIST)
            np.testing.assert_array_equal(
                np.asarray(GameBarrage.PLANE.position, dtype=np.float32),
                env.plane_position,
            )
            np.testing.assert_array_equal(
                np.asarray(GameBarrage.PLANE.velocity, dtype=np.float32),
                env.plane_velocity,
            )
            np.testing.assert_array_equal(
                game_bullets[:, :2].astype(np.float32), env.bullet_positions
            )
            np.testing.assert_array_equal(
                game_bullets[:, 2:4].astype(np.float32), env.bullet_velocities
            )
            np.testing.assert_array_equal(
                game_bullets[:, 4].astype(np.bool_), env.bullet_is_targeted
            )
            self.assertEqual(
                GameBarrage.RNG.bit_generator.state,
                env.np_random.bit_generator.state,
            )

            terminal_positions = np.column_stack((
                np.linspace(40.0, 700.0, env.bullet_count, dtype=np.float32),
                np.full(env.bullet_count, 40.0, dtype=np.float32),
            ))
            terminal_velocities = np.zeros_like(terminal_positions)
            terminal_positions[0] = GameBarrage.PLANE.rect.center
            terminal_positions[1] = (100.0, 100.0)
            terminal_velocities[1] = (240.0, 0.0)
            env.bullet_positions[:] = terminal_positions
            env.bullet_velocities[:] = terminal_velocities
            from barrage_rl.window_bullets import WindowBulletField
            field = WindowBulletField()
            field.append_batch(terminal_positions, terminal_velocities,
                               np.zeros(env.bullet_count, dtype=np.bool_))
            type(GameBarrage.BULLET).LIST = field
            GameBarrage.INVINCIBLE = True
            GameBarrage.KEY = True
            GameBarrage.BULLET.rect.center = GameBarrage.BULLET.LIST[0][:2]
            self.assertTrue(
                GameBarrage.BULLET.rect.colliderect(GameBarrage.PLANE.rect)
            )
            self.assertIsNotNone(
                pygame.sprite.collide_mask(
                    GameBarrage.BULLET, GameBarrage.PLANE
                )
            )
            GameBarrage.BULLET.update(
                GameBarrage.PLANE, 1.0 / GameBarrage.PHYSICS_FPS
            )
            env._move_bullets()
            self.assertFalse(GameBarrage.KEY)
            self.assertTrue(env._has_collision())
            self.assertEqual(GameBarrage.BULLET.LIST[1][0], 102.0)
            np.testing.assert_array_equal(
                np.asarray(GameBarrage.BULLET.LIST)[:, :2].astype(np.float32),
                env.bullet_positions,
            )
        finally:
            env.close()
            GameBarrage.QUANTITY = previous["quantity"]
            GameBarrage.BULLET_SIZE = previous["bullet_size"]
            GameBarrage.BULLET_SPEED = previous["bullet_speed"]
            GameBarrage.TARGETED_BULLET_PROBABILITY = previous["targeted"]
            GameBarrage.COLLISION = previous["collision"]
            GameBarrage.INVINCIBLE = previous["invincible"]
            GameBarrage.KEY = previous["key"]
            GameBarrage.RNG = previous["rng"]
            GameBarrage.AI_CONTROLLER = previous["controller"]
            GameBarrage.window = previous["window"]

    def test_observation_shape_and_seed(self) -> None:
        first = BarrageVisionEnv(bullet_count=300)
        second = BarrageVisionEnv(bullet_count=300)
        try:
            first_observation, _ = first.reset(seed=123)
            second_observation, _ = second.reset(seed=123)
            self.assertEqual(first_observation.shape, (4, 96, 96))
            self.assertEqual(first_observation.dtype, np.uint8)
            np.testing.assert_array_equal(first_observation, second_observation)
        finally:
            first.close()
            second.close()

    def test_environment_respawns_bullets_sequentially_in_index_order(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            randomize_initial_phase=False,
            targeted_bullet_probability=0.10,
        )
        try:
            env.reset(seed=9127)
            reference_rng = np.random.default_rng()
            reference_rng.bit_generator.state = deepcopy(env.np_random.bit_generator.state)
            expected: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for index in (2, 7):
                expected[index] = spawn_bullets(
                    1,
                    env.screen_width,
                    env.screen_height,
                    env.bullet_speed,
                    env.plane_position,
                    env.plane_velocity,
                    env.targeted_bullet_probability,
                    env.targeted_prediction_scale_min,
                    env.targeted_prediction_scale_max,
                    env.targeted_angular_noise,
                    reference_rng,
                )

            env._spawn_bullets(np.asarray([7, 2], dtype=np.int64))

            for index in (2, 7):
                positions, velocities, targeted = expected[index]
                np.testing.assert_array_equal(env.bullet_positions[index], positions[0])
                np.testing.assert_array_equal(env.bullet_velocities[index], velocities[0])
                self.assertEqual(env.bullet_is_targeted[index], targeted[0])
            self.assertEqual(
                env.np_random.bit_generator.state,
                reference_rng.bit_generator.state,
            )
        finally:
            env.close()

    def test_scalar_respawn_matches_single_item_batch_exactly(self) -> None:
        for seed in range(32):
            for targeted_probability in (0.0, 0.10, 1.0):
                batch_rng = np.random.default_rng(seed)
                scalar_rng = np.random.default_rng(seed)
                positions, velocities, targeted = spawn_bullets(
                    1,
                    820,
                    820,
                    240.0,
                    np.asarray([410.25, 409.75], dtype=np.float32),
                    np.asarray([-240.0, 0.0], dtype=np.float32),
                    targeted_probability,
                    0.65,
                    1.0,
                    0.08,
                    batch_rng,
                )
                position, velocity, is_targeted = spawn_bullet(
                    820,
                    820,
                    240.0,
                    np.asarray([410.25, 409.75], dtype=np.float32),
                    np.asarray([-240.0, 0.0], dtype=np.float32),
                    targeted_probability,
                    0.65,
                    1.0,
                    0.08,
                    scalar_rng,
                )
                np.testing.assert_array_equal(position, positions[0])
                np.testing.assert_array_equal(velocity, velocities[0])
                self.assertEqual(is_targeted, targeted[0])
                self.assertEqual(
                    scalar_rng.bit_generator.state,
                    batch_rng.bit_generator.state,
                )

    def test_privileged_collision_attribution_returns_exact_bullet_index(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            randomize_initial_phase=False,
            targeted_bullet_probability=0.10,
        )
        try:
            env.reset(seed=456)
            env.bullet_positions[:] = (-100.0, -100.0)
            env.bullet_positions[3] = env.plane_position
            self.assertTrue(env._has_collision())
            np.testing.assert_array_equal(
                env.colliding_bullet_indices(), np.asarray([3], dtype=np.int64)
            )
        finally:
            env.close()

    def test_teacher_is_deterministic_and_does_not_mutate_environment(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            randomize_initial_phase=False,
            targeted_bullet_probability=0.5,
        )
        try:
            env.reset(seed=10)
            before = env.capture_state()
            first = privileged_planner_supervision(
                env, horizon_seconds=0.3, safety_horizons=(0.1, 0.3)
            )
            after = env.capture_state()
            second = privileged_planner_supervision(
                env, horizon_seconds=0.3, safety_horizons=(0.1, 0.3)
            )
            self.assertEqual(first.action, second.action)
            np.testing.assert_allclose(first.action_costs, second.action_costs)
            np.testing.assert_array_equal(before.plane_position, after.plane_position)
            np.testing.assert_array_equal(before.bullet_positions, after.bullet_positions)
            self.assertEqual(before.rng_state, after.rng_state)
            self.assertEqual(first.sequence_viable.shape, (len(env.ACTIONS),))
            self.assertEqual(
                first.terminal_viable_action_count.shape, (len(env.ACTIONS),)
            )

            searched = privileged_planner_supervision(
                env,
                horizon_seconds=0.1,
                safety_horizons=(0.1,),
                strong_sequence_beam_width=9,
                strong_sequence_search_seconds=0.1,
                terminal_reserve_seconds=0.1,
                force_sequence_search=True,
            )
            final = env.capture_state()
            self.assertTrue(searched.used_sequence_search)
            self.assertEqual(searched.sequence_beam_width, len(env.ACTIONS))
            self.assertGreater(searched.sequence_nodes_expanded, 0)
            np.testing.assert_array_equal(before.plane_position, final.plane_position)
            np.testing.assert_array_equal(before.bullet_positions, final.bullet_positions)
            self.assertEqual(before.rng_state, final.rng_state)
        finally:
            env.close()

    def test_sparse_opening_teacher_ignores_hidden_future_rng(self) -> None:
        kwargs = {
            "bullet_count": 300,
            "bullet_size_min": 5,
            "bullet_size_max": 5,
            "bullet_speed_min": 240.0,
            "bullet_speed_max": 240.0,
            "targeted_bullet_probability": 0.10,
            "randomize_initial_phase": False,
        }
        first = BarrageVisionEnv(**kwargs)
        second = BarrageVisionEnv(**kwargs)
        try:
            for env, seed in ((first, 1001), (second, 1002)):
                env.reset(seed=seed)
                env.bullet_positions[:] = (700.0, 700.0)
                env.bullet_velocities.fill(0.0)
                env.bullet_is_targeted.fill(False)
                env.plane_position[:] = (100.0, 100.0)
                env.plane_velocity.fill(0.0)
                env.physics_steps = 8
                env.opening_spawned_batches = 1

            first_before = first.capture_state()
            second_before = second.capture_state()
            first_label = privileged_planner_supervision(
                first,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            second_label = privileged_planner_supervision(
                second,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )

            self.assertEqual(first_label.action, second_label.action)
            np.testing.assert_array_equal(
                first_label.action_costs, second_label.action_costs
            )
            np.testing.assert_array_equal(
                first_label.regrets, second_label.regrets
            )
            np.testing.assert_array_equal(
                first_label.safety_targets, second_label.safety_targets
            )
            self.assertEqual(
                first.np_random.bit_generator.state, first_before.rng_state
            )
            self.assertEqual(
                second.np_random.bit_generator.state, second_before.rng_state
            )
        finally:
            first.close()
            second.close()

    def test_mid_episode_teacher_ignores_hidden_respawn_rng(self) -> None:
        kwargs = {
            "bullet_count": 300,
            "bullet_size_min": 5,
            "bullet_size_max": 5,
            "bullet_speed_min": 240.0,
            "bullet_speed_max": 240.0,
            "targeted_bullet_probability": 0.10,
            "randomize_initial_phase": False,
        }
        first = BarrageVisionEnv(**kwargs)
        second = BarrageVisionEnv(**kwargs)
        try:
            for env, seed in ((first, 8101), (second, 8102)):
                env.reset(seed=seed)
                env.opening_spawned_batches = OPENING_BATCH_COUNT
                env.physics_steps = 1200
                env.plane_position[:] = (410.0, 410.0)
                env.plane_velocity.fill(0.0)
                env.bullet_positions = np.column_stack((
                    np.ones(300, dtype=np.float32),
                    np.linspace(2.0, 818.0, 300, dtype=np.float32),
                ))
                env.bullet_velocities = np.tile(
                    np.asarray((-240.0, 0.0), dtype=np.float32),
                    (300, 1),
                )
                env.bullet_is_targeted = np.zeros(300, dtype=np.bool_)
                env.bullet_positions[0] = (410.0, 100.0)
                env.bullet_velocities[0] = (0.0, 240.0)

            first_before = first.capture_state()
            second_before = second.capture_state()
            first_label = privileged_planner_supervision(first)
            second_label = privileged_planner_supervision(second)

            self.assertEqual(first_label.action, second_label.action)
            np.testing.assert_array_equal(
                first_label.action_costs,
                second_label.action_costs,
            )
            np.testing.assert_array_equal(first_label.regrets, second_label.regrets)
            np.testing.assert_array_equal(
                first_label.safety_targets,
                second_label.safety_targets,
            )
            self.assertEqual(
                first.np_random.bit_generator.state,
                first_before.rng_state,
            )
            self.assertEqual(
                second.np_random.bit_generator.state,
                second_before.rng_state,
            )
        finally:
            first.close()
            second.close()

    def test_sparse_safe_opening_prefers_stay_over_previous_velocity(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            bullet_size_min=5,
            bullet_size_max=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=7712)
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.bullet_is_targeted.fill(False)
            env.plane_position[:] = (410.0, 410.0)
            env.plane_velocity[:] = (240.0, 0.0)
            env.opening_spawned_batches = 1
            sparse = env.capture_state()

            label = privileged_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            self.assertEqual(label.action, 0)
            self.assertLess(
                _action_change_cost(env, sparse, 0),
                _action_change_cost(env, sparse, 2),
            )

            env.restore_state(sparse)
            env.bullet_positions[0] = (410.0, 380.0)
            env.bullet_velocities[0] = (0.0, 240.0)
            risky = privileged_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                safety_horizons=(0.10,),
            )
            self.assertEqual(risky.safety_targets[0, 0], 1.0)
            self.assertNotEqual(risky.action, 0)

            env.opening_spawned_batches = OPENING_BATCH_COUNT
            full = env.capture_state()
            self.assertLess(
                _action_change_cost(env, full, 2),
                _action_change_cost(env, full, 0),
            )
        finally:
            env.close()

    def test_exact_teacher_prefers_inward_action_near_wall(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=7713)
            half_plane = env.plane_size / 2.0
            env.plane_position[:] = (half_plane[0] + 1.0, 410.0)
            env.plane_velocity.fill(0.0)
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.bullet_is_targeted.fill(False)
            env.opening_spawned_batches = 1

            no_margin = privileged_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                wall_margin=0.0,
                wall_penalty_weight=0.0,
                safety_horizons=(0.10,),
            )
            wall_margin = privileged_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                wall_margin=80.0,
                wall_penalty_weight=0.35,
                safety_horizons=(0.10,),
            )

            self.assertEqual(no_margin.action, 0)
            self.assertNotEqual(wall_margin.action, no_margin.action)
            self.assertGreater(
                float(np.max(np.abs(wall_margin.action_costs - no_margin.action_costs))),
                0.0,
            )
        finally:
            env.close()

    def test_fast_teacher_prefers_inward_action_near_wall(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=7714)
            half_plane = env.plane_size / 2.0
            env.plane_position[:] = (half_plane[0] + 1.0, 410.0)
            env.plane_velocity.fill(0.0)
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities.fill(0.0)
            env.bullet_is_targeted.fill(False)

            no_margin = fast_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                wall_margin=0.0,
                wall_penalty_weight=0.0,
                safety_horizons=(0.10,),
            )
            wall_margin = fast_planner_supervision(
                env,
                horizon_seconds=0.10,
                reaction_seconds=0.10,
                wall_margin=80.0,
                wall_penalty_weight=0.35,
                safety_horizons=(0.10,),
            )

            self.assertEqual(no_margin.action, 0)
            self.assertNotEqual(wall_margin.action, no_margin.action)
            self.assertGreater(
                float(np.max(np.abs(wall_margin.action_costs - no_margin.action_costs))),
                0.0,
            )
        finally:
            env.close()

    def test_live_semanticizer_ignores_background_color(self) -> None:
        first = np.full((120, 120, 3), (10, 40, 90), dtype=np.uint8)
        second = np.full((120, 120, 3), (90, 20, 10), dtype=np.uint8)
        for image in (first, second):
            image[56:64, 56:64] = (255, 0, 0)
            image[18:23, 28:33] = (255, 255, 255)
            image[18, 28] = image[18, 32] = image[22, 28] = image[22, 32] = image[0, 0]
        extractor = DominantBackgroundSemanticizer(output_size=96)
        semantic_first = extractor.convert(first)
        extractor.reset()
        semantic_second = extractor.convert(second)
        np.testing.assert_array_equal(semantic_first, semantic_second)
        self.assertGreater(np.count_nonzero(semantic_first == 255), 0)
        self.assertGreater(np.count_nonzero(semantic_first == 96), 0)

    def test_predictive_semanticizer_tracks_locally_and_recovers_residual(self) -> None:
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[56:64, 56:64] = (255, 0, 0)

        def draw_bullet(x: int, y: int) -> None:
            for dy, dx in DominantBackgroundSemanticizer._BULLET_OFFSETS:
                image[y + dy, x + dx] = (255, 255, 255)

        draw_bullet(80, 60)
        draw_bullet(50, 75)
        extractor = DominantBackgroundSemanticizer(
            output_size=96,
            recovery_edge_width=8,
            recovery_stripe_width=20,
        )
        prediction = np.asarray([[79.0 / 120.0, 60.0 / 120.0]], np.float32)
        radius = np.asarray([4.0], np.float32)
        first = extractor.detect(
            image,
            include_semantic=False,
            predicted_bullet_positions=prediction,
            prediction_search_radii=radius,
        )
        self.assertEqual(extractor.last_detection_mode, "predictive")
        self.assertEqual(extractor.last_predicted_match_count, 1)
        self.assertEqual(extractor.last_recovery_detection_count, 1)
        self.assertEqual(len(first.bullet_positions), 2)
        np.testing.assert_allclose(
            first.bullet_positions,
            np.asarray([[80.0 / 120.0, 60.0 / 120.0], [50.0 / 120.0, 75.0 / 120.0]]),
            atol=1e-6,
        )

    def test_incremental_cover_matches_reference_greedy_order(self) -> None:
        extractor = DominantBackgroundSemanticizer(output_size=64)

        def reference(
            bullet_mask: np.ndarray,
            candidates: np.ndarray,
            uncovered_mask: np.ndarray | None = None,
        ) -> np.ndarray:
            height, width = bullet_mask.shape
            candidates = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
            sentinel = height * width
            coverage = np.full(
                (len(candidates), len(extractor._BULLET_OFFSETS)),
                sentinel,
                dtype=np.int32,
            )
            for offset_index, (dy, dx) in enumerate(extractor._BULLET_OFFSETS):
                source_y = candidates[:, 0] + dy
                source_x = candidates[:, 1] + dx
                valid = (
                    (source_y >= 0)
                    & (source_y < height)
                    & (source_x >= 0)
                    & (source_x < width)
                )
                coverage[valid, offset_index] = source_y[valid] * width + source_x[valid]
            residual = bullet_mask if uncovered_mask is None else uncovered_mask
            uncovered = np.concatenate((
                np.asarray(residual, dtype=np.bool_).reshape(-1).copy(),
                np.asarray([False], dtype=np.bool_),
            ))
            available = np.ones(len(candidates), dtype=np.bool_)
            selected: list[int] = []
            remaining = int(np.count_nonzero(uncovered))
            while remaining > 0:
                gains = np.count_nonzero(uncovered[coverage], axis=1).astype(
                    np.int32, copy=False
                )
                gains[~available] = -1
                best = int(np.argmax(gains))
                if gains[best] < 2:
                    break
                selected.append(best)
                remaining -= int(gains[best])
                uncovered[coverage[best]] = False
                available[best] = False
            return candidates[np.asarray(selected, dtype=np.int32), ::-1].astype(
                np.float32
            )

        rng = np.random.default_rng(90210)
        for case in range(12):
            mask = np.zeros((64, 64), dtype=np.bool_)
            centers = np.concatenate((
                rng.integers(18, 46, size=(35, 2)),
                rng.integers(0, 64, size=(20, 2)),
            ))
            for center_y, center_x in centers:
                for dy, dx in extractor._BULLET_OFFSETS:
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < 64 and 0 <= x < 64:
                        mask[y, x] = True
            candidates = extractor._validated_candidates(
                mask, extractor._candidate_hypotheses(mask)
            )
            residual = mask.copy()
            if case % 2:
                residual[:, : 20 + case] = False
            np.testing.assert_array_equal(
                extractor._select_cover(mask, candidates, uncovered_mask=residual),
                reference(mask, candidates, residual),
            )

    def test_current_evaluation_statistics(self) -> None:
        times = np.asarray([8.0, 12.0, 18.0, 30.0], dtype=np.float32)
        confidence = bootstrap_confidence_intervals(times, 123, 2_000)
        self.assertLess(confidence["model_mean_ci95_low"], float(times.mean()))
        self.assertGreater(confidence["model_mean_ci95_high"], float(times.mean()))
        self.assertEqual(interquartile_mean(times), 15.0)
        self.assertEqual(lower_tail_mean(np.arange(1, 11), 0.5), 3.0)
        self.assertAlmostEqual(wilson_lower_bound(68, 100), 0.583372, places=5)

    def test_dashboard_uses_current_selection_and_axis_rules(self) -> None:
        class Axis:
            limits = None

            def set_ylim(self, lower: float, upper: float) -> None:
                self.limits = (lower, upper)

        axis = Axis()
        _set_middle_80_ylim(axis, np.asarray([2.0, 10.0]))
        self.assertEqual(axis.limits, (1.0, 11.0))

        class ReliabilityAxis(Axis):
            def get_ylim(self) -> tuple[float, float]:
                assert self.limits is not None
                return self.limits

        reliability_axis = ReliabilityAxis()
        _set_reliability_ylim(
            reliability_axis,
            np.asarray([96.0, 98.0]),
            np.asarray([92.0, 94.0]),
        )
        self.assertEqual(reliability_axis.limits, (91.25, 100.0))
        data = {
            "model_iqm": [120.0, 120.0, 120.0],
            "model_mean": [119.0, 119.4, 119.4],
            "model_rmst": [119.0, 119.4, 119.4],
            "model_p10": [120.0, 120.0, 120.0],
            "model_cvar5": [94.0, 108.0, 108.0],
            "success_at_limit": [0.98, 0.995, 0.995],
        }
        self.assertEqual(_dagger_best_index(data, "success_then_cvar5", 110.0), 1)
        self.assertEqual(
            _dagger_best_index(data, "iqm_then_success_then_mean", 120.0), 2
        )
        self.assertEqual(
            _dagger_best_metric(
                data, "iqm_then_success_then_mean", 120.0, 1
            ),
            "success_at_limit",
        )
        below_limit = dict(data, model_iqm=[110.0, 119.0, 118.0])
        self.assertEqual(
            _dagger_best_metric(
                below_limit, "iqm_then_success_then_mean", 120.0, 1
            ),
            "success_at_limit",
        )

    def test_current_round_dashboard_renders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "round_summaries.csv"
            summary.write_text(
                "round,episodes,bullets,bullet_size_min,bullet_speed_min,"
                "targeted_bullet_probability,model_mean,model_median,model_iqm,"
                "model_p1,model_p5,model_cvar5,success_at_limit,"
                "success_at_limit_ci95_low,wall_episode_fraction,"
                "median_min_wall_distance\n"
                "1,200,200,5,240,0.10,40,35,38,3,2,1.5,0.10,0.04,0.50,30\n"
                "2,200,200,5,240,0.10,45,41,43,5,3,2.0,0.16,0.08,0.40,45\n",
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                '{"output_dir":"runs/visual_set_v30",'
                '"selection_mode":"success_then_cvar5"}',
                encoding="utf-8",
            )
            output = save_round_summary_plot(summary, root / "results.png", config)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
