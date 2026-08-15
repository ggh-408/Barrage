"""不依赖pytest的基础回归测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from Barrage import Barrage as GameBarrage, Plane as GamePlane
from barrage_rl.env import BarrageVisionEnv, BatchedBarrageEnv
from barrage_rl.scenarios import ScenarioSampler
from barrage_rl.evaluate_visual_set import _bootstrap_confidence_intervals
from barrage_rl.baselines import privileged_planner_action, privileged_planner_supervision
from barrage_rl.model import ActorCritic
from barrage_rl.plot import save_results_plot, save_round_summary_plot
from barrage_rl.visual_set import (
    ScreenOnlyAgent,
    SemanticFrameExtractor,
    VisualSetRecurrentQNetwork,
    VisualSetSpec,
    calibrate_safety_thresholds,
    safety_filter_is_usable,
)
from barrage_rl.train_visual_set import (
    DAggerConfig,
    _cost_sensitive_loss,
    _episode_split,
    _sequence_windows,
)
from barrage_rl.train_qdagger import (
    EpisodeTransitionReplay,
    QDaggerConfig,
    _n_step_double_q_targets,
    _stack_sequences,
)
from barrage_rl.live_screen import DominantBackgroundSemanticizer
from barrage_rl.planner_pool import (
    _initialize_worker,
    _planner_task,
    _supervise_task,
    _worker_environment_kwargs,
)


class BarrageEnvironmentTests(unittest.TestCase):
    def test_teacher_wall_penalty_is_fixed(self) -> None:
        config = DAggerConfig()
        self.assertEqual(config.teacher_wall_penalty_weight, 0.35)
        self.assertEqual(config.rounds, 10)
        self.assertEqual(config.replay_capacity, 350_000)
        self.assertEqual(config.evaluation_episodes, 50)

    def test_qdagger_evaluates_100_episodes_by_default(self) -> None:
        config = QDaggerConfig(checkpoint="unused.pt")
        self.assertEqual(config.evaluation_episodes, 100)
        self.assertEqual(config.evaluation_workers, 8)

    def test_live_plane_accepts_shared_numpy_action_vector(self) -> None:
        import pygame

        plane = object.__new__(GamePlane)
        plane.rect = pygame.Rect(0, 0, 10, 10)
        plane.rect.center = (100, 100)
        plane.position = pygame.Vector2(100.0, 100.0)
        plane.velocity = pygame.Vector2(0.0, 0.0)
        old_width = GameBarrage.SCREEN_WIDTH
        old_height = GameBarrage.SCREEN_HEIGHT
        old_speed = GameBarrage.PLANE_SPEED
        try:
            GameBarrage.SCREEN_WIDTH = 820
            GameBarrage.SCREEN_HEIGHT = 820
            GameBarrage.PLANE_SPEED = 240.0
            plane.move(np.asarray([1.0, 0.0], np.float32), 1.0 / 120.0)
            self.assertAlmostEqual(plane.position.x, 102.0)
            self.assertAlmostEqual(plane.position.y, 100.0)
        finally:
            GameBarrage.SCREEN_WIDTH = old_width
            GameBarrage.SCREEN_HEIGHT = old_height
            GameBarrage.PLANE_SPEED = old_speed

    def test_live_game_uses_fixed_physics_steps(self) -> None:
        class FakePlane:
            def __init__(self) -> None:
                self.steps = []

            def move(self, direction, delta_time) -> None:
                self.steps.append((direction, delta_time))

        class FakeBullet:
            def __init__(self) -> None:
                self.steps = []

            def update(self, plane, delta_time) -> None:
                self.steps.append((plane, delta_time))

        old_plane = GameBarrage.PLANE
        old_bullet = GameBarrage.BULLET
        old_key = GameBarrage.KEY
        old_time = GameBarrage.TimeNow
        plane = FakePlane()
        bullet = FakeBullet()
        try:
            GameBarrage.PLANE = plane
            GameBarrage.BULLET = bullet
            GameBarrage.KEY = True
            GameBarrage.TimeNow = 0.0
            step = 1.0 / GameBarrage.PHYSICS_FPS
            for _ in range(6):
                GameBarrage.advance_physics((False, True, False, False), step)
            self.assertEqual(len(plane.steps), 6)
            self.assertEqual(len(bullet.steps), 6)
            self.assertTrue(all(value[1] == step for value in plane.steps))
            self.assertAlmostEqual(GameBarrage.TimeNow, 6 * step)
        finally:
            GameBarrage.PLANE = old_plane
            GameBarrage.BULLET = old_bullet
            GameBarrage.KEY = old_key
            GameBarrage.TimeNow = old_time

    def test_wall_bounce_is_disabled_by_default(self) -> None:
        env = BarrageVisionEnv(bullet_count=1, randomize_initial_phase=False)
        try:
            self.assertFalse(env.wall_collision)
            self.assertEqual(
                env.max_episode_steps, 60 * env.metadata["render_fps"]
            )
        finally:
            env.close()

    def test_observation_shape_and_seed(self) -> None:
        first = BarrageVisionEnv(bullet_count=10)
        second = BarrageVisionEnv(bullet_count=10)
        try:
            first_observation, _ = first.reset(seed=123)
            second_observation, _ = second.reset(seed=123)
            self.assertEqual(first_observation.shape, (4, 96, 96))
            self.assertEqual(first_observation.dtype, np.uint8)
            np.testing.assert_array_equal(first_observation, second_observation)
        finally:
            first.close()
            second.close()

    def test_step_and_batch(self) -> None:
        batch = BatchedBarrageEnv(
            [BarrageVisionEnv(bullet_count=5) for _ in range(4)]
        )
        try:
            observation = batch.reset(seed=20)
            next_observation, rewards, dones, _ = batch.step(
                np.zeros(4, dtype=np.int64)
            )
            self.assertEqual(observation.shape, (4, 4, 96, 96))
            self.assertEqual(next_observation.shape, observation.shape)
            self.assertEqual(rewards.shape, (4,))
            self.assertEqual(dones.shape, (4,))
        finally:
            batch.close()

    def test_batch_uses_one_global_stratified_scenario_stream(self) -> None:
        sampler = ScenarioSampler(321)
        batch = BatchedBarrageEnv(
            [
                BarrageVisionEnv(
                    bullet_count=1,
                    max_episode_seconds=1.0 / 30.0,
                    scenario_mix=False,
                )
                for _ in range(8)
            ],
            reset_options_factory=lambda _index: {
                "scenario": sampler.next(), "reset_mode": "deployment"
            },
        )
        sources = []
        try:
            batch.reset(90)
            while len(sources) < 100:
                for env in batch.envs:
                    if len(sources) < 100:
                        sources.append(env.current_scenario.source)
                batch.step(np.zeros(8, dtype=np.int64))
        finally:
            batch.close()
        self.assertEqual(sources.count("core"), 60)
        self.assertEqual(sources.count("broad"), 25)
        self.assertEqual(sources.count("stress"), 15)

    def test_random_phase_has_motion_and_distinct_plane(self) -> None:
        env = BarrageVisionEnv(bullet_count=50)
        try:
            observation, info = env.reset(seed=123)
            self.assertEqual(info["survival_seconds"], 0.0)
            self.assertFalse(np.array_equal(observation[0], observation[-1]))
            self.assertEqual(
                int(np.count_nonzero(observation[-1] == env.PLANE_INTENSITY)),
                env.MIN_PLANE_OBSERVATION_SIZE ** 2,
            )
            self.assertGreater(
                int(np.count_nonzero(observation[-1] == env.BULLET_INTENSITY)),
                0,
            )
            distance_to_wall = np.minimum(
                env.bullet_positions,
                np.asarray([env.screen_width, env.screen_height])
                - env.bullet_positions,
            )
            self.assertTrue(np.any(np.all(distance_to_wall > 50.0, axis=1)))
        finally:
            env.close()

    def test_danger_potential_detects_approaching_trajectory(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=9)
            env.plane_position[:] = (410.0, 410.0)
            env.bullet_positions[:] = (510.0, 410.0)
            env.bullet_velocities[:] = (-240.0, 0.0)
            approaching_danger = env._danger_potential()
            env.bullet_velocities[:] = (240.0, 0.0)
            receding_danger = env._danger_potential()
            self.assertGreater(approaching_danger, 0.1)
            self.assertEqual(receding_danger, 0.0)
        finally:
            env.close()

    def test_diagonal_actions_have_unit_length(self) -> None:
        lengths = np.linalg.norm(BarrageVisionEnv.ACTIONS[1:], axis=1)
        np.testing.assert_allclose(lengths, np.ones_like(lengths), atol=1e-6)

    def test_bullet_observation_encodes_visible_size(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1,
            bullet_size_min=7,
            bullet_size_max=7,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=4)
            env.bullet_positions[:] = (300.0, 300.0)
            frame = env._make_frame()
            self.assertEqual(int(np.count_nonzero(frame == env.BULLET_INTENSITY)), 1)
            self.assertGreater(
                int(np.count_nonzero(frame == env.BULLET_SIZE_INTENSITY_BASE + 7)),
                env.BULLET_OBSERVATION_SIZE ** 2,
            )
        finally:
            env.close()

    def test_random_bullet_dynamics_stay_visible_and_in_range(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1,
            bullet_size_min=1,
            bullet_size_max=7,
            bullet_speed_min=60.0,
            bullet_speed_max=300.0,
            randomize_initial_phase=False,
        )
        sizes = set()
        speeds = []
        try:
            for seed in range(64):
                observation, info = env.reset(seed=seed)
                sizes.add(info["bullet_size"])
                speeds.append(info["bullet_speed"])
                self.assertGreaterEqual(
                    int(np.count_nonzero(observation[-1] == env.BULLET_INTENSITY)),
                    1,
                )
            self.assertEqual(sizes, set(range(1, 8)))
            self.assertGreaterEqual(min(speeds), 60.0)
            self.assertLessEqual(max(speeds), 300.0)
            self.assertGreater(np.std(speeds), 10.0)
        finally:
            env.close()

    def test_maximum_bullet_speed_does_not_tunnel_through_plane(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1,
            bullet_size=1,
            bullet_speed_min=300.0,
            bullet_speed_max=300.0,
            randomize_initial_phase=False,
        )
        try:
            env.reset(seed=77)
            env.plane_position[:] = (410.0, 410.0)
            env.bullet_positions[:] = (430.0, 410.0)
            env.bullet_velocities[:] = (-300.0, 0.0)
            detected = False
            for _ in range(12):
                env._move_bullets()
                detected = detected or env._has_collision()
            self.assertTrue(detected)
        finally:
            env.close()

    def test_dense_reward_prefers_moving_away_from_collision(self) -> None:
        def configured_env() -> BarrageVisionEnv:
            env = BarrageVisionEnv(
                bullet_count=1,
                randomize_initial_phase=False,
                dense_reward_scale=0.25,
            )
            env.reset(seed=9)
            env.plane_position[:] = (410.0, 410.0)
            env.bullet_positions[:] = (510.0, 410.0)
            env.bullet_velocities[:] = (-240.0, 0.0)
            return env

        away = configured_env()
        toward = configured_env()
        try:
            _, away_reward, away_done, _, _ = away.step(1)
            _, toward_reward, toward_done, _, _ = toward.step(2)
            self.assertFalse(away_done)
            self.assertFalse(toward_done)
            self.assertGreater(away_reward, toward_reward)
        finally:
            away.close()
            toward.close()

    def test_privileged_planner_avoids_head_on_bullet(self) -> None:
        env = BarrageVisionEnv(bullet_count=1, randomize_initial_phase=False)
        try:
            env.reset(seed=9)
            env.plane_position[:] = (410.0, 410.0)
            env.bullet_positions[:] = (510.0, 410.0)
            env.bullet_velocities[:] = (-240.0, 0.0)
            action = privileged_planner_action(env)
            self.assertIn(action, (1, 5, 7))
        finally:
            env.close()

    def test_privileged_planner_recovers_from_wall_when_safe(self) -> None:
        env = BarrageVisionEnv(bullet_count=1, randomize_initial_phase=False)
        try:
            env.reset(seed=10)
            env.plane_position[:] = (env.plane_size[0] / 2.0, 410.0)
            env.bullet_positions[:] = (700.0, 700.0)
            env.bullet_velocities[:] = (100.0, 100.0)
            self.assertIn(privileged_planner_action(env), (2, 6, 8))
        finally:
            env.close()

    def test_privileged_supervision_covers_all_actions(self) -> None:
        env = BarrageVisionEnv(bullet_count=5)
        try:
            env.reset(seed=14)
            supervision = privileged_planner_supervision(env)
            self.assertEqual(supervision.regrets.shape, (len(env.ACTIONS),))
            self.assertEqual(supervision.collision_mask.shape, (len(env.ACTIONS),))
            self.assertEqual(float(supervision.regrets.min()), 0.0)
        finally:
            env.close()

    def test_targeted_bullet_predicts_moving_plane_interception(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=20,
            randomize_initial_phase=False,
            targeted_bullet_probability=1.0,
            targeted_prediction_scale_min=1.0,
            targeted_prediction_scale_max=1.0,
            targeted_angular_noise=0.0,
        )
        try:
            env.reset(seed=11)
            env.plane_position[:] = (410.0, 410.0)
            env.plane_velocity[:] = (20.0, -10.0)
            env._spawn_bullets(np.arange(env.bullet_count))
            relative_position = env.bullet_positions - env.plane_position
            relative_velocity = env.bullet_velocities - env.plane_velocity
            closest_time = np.maximum(
                -np.sum(relative_position * relative_velocity, axis=1)
                / np.sum(relative_velocity * relative_velocity, axis=1),
                0.0,
            )
            closest_distance = np.linalg.norm(
                relative_position + relative_velocity * closest_time[:, None], axis=1
            )
            self.assertLess(float(closest_distance.max()), 1e-2)
            self.assertFalse(env.wall_collision)
        finally:
            env.close()

    def test_score_is_one_point_per_tenth_second(self) -> None:
        env = BarrageVisionEnv(bullet_count=1)
        try:
            env.reset(seed=99)
            info = {}
            for _ in range(3):
                _, _, terminated, _, info = env.step(0)
                self.assertFalse(terminated)
            self.assertEqual(info["score"], 1)
        finally:
            env.close()

    def test_results_plot_png(self) -> None:
        with TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.csv"
            metrics.write_text(
                "global_step,mean_score_100\n100,10\n200,15\n300,22\n",
                encoding="utf-8",
            )
            output = save_results_plot(metrics, Path(directory) / "results.png")
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)

    def test_results_plot_uses_epoch_axis(self) -> None:
        with TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.csv"
            metrics.write_text(
                "round,epoch,replay_samples,train_loss,validation_loss\n"
                "1,1,100,1.8,1.9\n"
                "1,2,100,1.5,1.7\n"
                "1,3,100,1.2,1.4\n",
                encoding="utf-8",
            )
            output = save_results_plot(metrics, Path(directory) / "results.png")

            import matplotlib.image as mpimg

            image = mpimg.imread(output)
            blue_curve_pixels = (
                (image[..., 2] > image[..., 0] + 0.05)
                & (image[..., 2] > image[..., 1] + 0.02)
            )
            self.assertGreater(int(blue_curve_pixels.sum()), 20)

    def test_evaluation_confidence_intervals_and_results_plot(self) -> None:
        model_times = np.asarray([8.0, 12.0, 18.0, 30.0], dtype=np.float32)
        confidence = _bootstrap_confidence_intervals(
            model_times, seed=123, bootstrap_samples=2_000
        )
        self.assertLess(confidence["model_mean_ci95_low"], float(model_times.mean()))
        self.assertGreater(confidence["model_mean_ci95_high"], float(model_times.mean()))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.csv"
            episodes = root / "evaluation_episodes.csv"
            summary = root / "evaluation_summary.csv"
            metrics.write_text(
                "round,epoch,train_loss,validation_loss\n1,1,1.8,1.9\n1,2,1.5,1.7\n",
                encoding="utf-8",
            )
            episodes.write_text(
                "episode,seed,model_survival_seconds\n"
                "0,10,8.0\n1,11,12.0\n2,12,18.0\n3,13,30.0\n",
                encoding="utf-8",
            )
            summary.write_text(
                "model_mean_ci95_low,model_mean_ci95_high\n"
                f"{confidence['model_mean_ci95_low']},"
                f"{confidence['model_mean_ci95_high']}\n",
                encoding="utf-8",
            )
            output = save_results_plot(
                metrics, root / "results.png", episodes, summary
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)

    def test_dagger_round_plot_contains_rollout_statistics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "round_summaries.csv"
            summary.write_text(
                "round,episodes,model_mean,model_median,model_iqm,model_p10,"
                "model_p5,model_cvar5,success_at_limit,success_at_120_ci95_low,"
                "wall_episode_fraction,median_min_wall_distance\n"
                "1,50,40,35,38,3,2,1.5,0.10,0.04,0.50,30\n"
                "2,50,45,41,43,5,3,2.0,0.16,0.08,0.40,45\n",
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                '{"bullet_count":50,"core_bullet_size":5,'
                '"core_bullet_speed":240,"targeted_bullet_probability":0.35}',
                encoding="utf-8",
            )
            output = save_round_summary_plot(
                summary, root / "results.png", config
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)

    def test_model_output(self) -> None:
        model = ActorCritic(4, 96, 9)
        observation = torch.zeros(3, 4, 96, 96, dtype=torch.uint8)
        action, log_probability, entropy, value = model.get_action_and_value(observation)
        self.assertEqual(tuple(action.shape), (3,))
        self.assertEqual(tuple(log_probability.shape), (3,))
        self.assertEqual(tuple(entropy.shape), (3,))
        self.assertEqual(tuple(value.shape), (3,))

    def test_legacy_model_output(self) -> None:
        model = ActorCritic(4, 96, 9, model_version=1)
        observation = torch.zeros(1, 4, 96, 96, dtype=torch.uint8)
        action, _, _, value = model.get_action_and_value(observation)
        self.assertEqual(tuple(action.shape), (1,))
        self.assertEqual(tuple(value.shape), (1,))

    def test_visual_set_is_extracted_only_from_frames(self) -> None:
        env = BarrageVisionEnv(bullet_count=10)
        try:
            observation, _ = env.reset(seed=321)
            spec = VisualSetSpec(max_objects=16)
            objects, mask, globals_ = SemanticFrameExtractor(spec).extract(observation)
            self.assertEqual(objects.shape, (16, spec.object_features))
            self.assertEqual(mask.shape, (16,))
            self.assertEqual(globals_.shape, (spec.global_features,))
            self.assertGreater(int(mask.sum()), 0)
        finally:
            env.close()

    def test_screen_only_agent_action_shape(self) -> None:
        env = BarrageVisionEnv(bullet_count=5)
        try:
            observation, _ = env.reset(seed=12)
            spec = VisualSetSpec(max_objects=16)
            model = VisualSetRecurrentQNetwork(spec, width=64)
            agent = ScreenOnlyAgent(
                model, SemanticFrameExtractor(spec), torch.device("cpu")
            )
            action = agent.act(observation[None])
            self.assertEqual(action.shape, (1,))
        finally:
            env.close()

    def test_v8_recurrent_attention_outputs(self) -> None:
        env = BarrageVisionEnv(bullet_count=10)
        try:
            observation, _ = env.reset(seed=15)
            spec = VisualSetSpec(max_objects=32, object_features=12)
            objects, mask, globals_ = SemanticFrameExtractor(spec).extract(observation)
            self.assertEqual(objects.shape, (32, 12))
            self.assertTrue(np.all((objects[mask, 8:12] >= 0.0)))
            model = VisualSetRecurrentQNetwork(spec, width=64)
            logits, q_values, collision_logits, hidden = model.forward_sequence(
                torch.as_tensor(objects[None, None]),
                torch.as_tensor(mask[None, None]),
                torch.as_tensor(globals_[None, None]),
            )
            self.assertEqual(tuple(logits.shape), (1, 1, 9))
            self.assertEqual(tuple(q_values.shape), (1, 1, 9))
            self.assertEqual(tuple(collision_logits.shape), (1, 1, 4, 9))
            self.assertEqual(tuple(hidden.shape), (1, 1, 64))
        finally:
            env.close()

    def test_qdagger_stacks_one_aligned_recurrent_state_chain(self) -> None:
        def transition(state: float, next_state: float) -> dict:
            return {
                "objects": np.asarray([[state]], np.float32),
                "masks": np.asarray([True]),
                "globals": np.asarray([state], np.float32),
                "next_objects": np.asarray([[next_state]], np.float32),
                "next_masks": np.asarray([True]),
                "next_globals": np.asarray([next_state], np.float32),
                "actions": np.asarray(0, np.int64),
                "rewards": np.asarray(0.0, np.float32),
                "terminals": np.asarray(0.0, np.float32),
                "truncations": np.asarray(0.0, np.float32),
                "regrets": np.zeros(9, np.float32),
                "collisions": np.zeros((4, 9), np.float32),
            }

        batch = _stack_sequences(
            [([transition(1.0, 2.0), transition(2.0, 3.0)], True)],
            length=4,
            device=torch.device("cpu"),
        )
        (
            objects, _, globals_, _, _, _, _, _, _, teacher_valid, valid,
            starts_at_start
        ) = batch
        np.testing.assert_array_equal(
            objects[0, :, 0, 0].numpy(), np.asarray([1, 2, 3, 0, 0])
        )
        np.testing.assert_array_equal(
            globals_[0, :, 0].numpy(), np.asarray([1, 2, 3, 0, 0])
        )
        np.testing.assert_array_equal(
            valid[0].numpy(), np.asarray([True, True, False, False])
        )
        self.assertTrue(bool(starts_at_start[0]))
        self.assertTrue(bool(teacher_valid[0, :2].all()))

    def test_qdagger_unlabelled_teacher_targets_are_explicitly_masked(self) -> None:
        transition = {
            "objects": np.zeros((1, 1), np.float32),
            "masks": np.asarray([True]),
            "globals": np.zeros(1, np.float32),
            "next_objects": np.zeros((1, 1), np.float32),
            "next_masks": np.asarray([True]),
            "next_globals": np.zeros(1, np.float32),
            "actions": np.asarray(0, np.int64),
            "rewards": np.asarray(0.0, np.float32),
            "terminals": np.asarray(0.0, np.float32),
            "truncations": np.asarray(0.0, np.float32),
            "regrets": np.zeros(9, np.float32),
            "collisions": np.zeros((4, 9), np.float32),
            "teacher_valid": np.asarray(False, np.bool_),
        }
        batch = _stack_sequences(
            [([transition], True)], length=1, device=torch.device("cpu")
        )
        self.assertFalse(bool(batch[9][0, 0]))
        self.assertTrue(bool(batch[10][0, 0]))

    def test_qdagger_three_step_targets_stop_at_terminal(self) -> None:
        online_q = torch.zeros((1, 5, 2), dtype=torch.float32)
        online_q[:, :, 1] = 1.0
        target_q = torch.zeros((1, 5, 2), dtype=torch.float32)
        for state_index in range(5):
            target_q[0, state_index, 1] = 10.0 * state_index + 1.0
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        dones = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        valid = torch.ones((1, 4), dtype=torch.bool)
        truncations = torch.zeros_like(dones)
        targets, target_valid = _n_step_double_q_targets(
            online_q, target_q, rewards, dones, truncations, valid,
            gamma=0.5, n_step=3
        )
        # t=0 bootstraps at s3; all later positions encounter the terminal and
        # must not bootstrap from the reset observation after it.
        expected = torch.tensor([[6.625, 4.5, 5.0, 4.0]])
        torch.testing.assert_close(targets, expected)
        self.assertTrue(bool(target_valid.all()))

        no_terminal = torch.zeros_like(dones)
        _, incomplete_valid = _n_step_double_q_targets(
            online_q, target_q, rewards, no_terminal, truncations, valid,
            gamma=0.5, n_step=3
        )
        np.testing.assert_array_equal(
            incomplete_valid[0].numpy(), np.asarray([True, True, False, False])
        )

    def test_episode_validation_split_has_no_leakage(self) -> None:
        episode_ids = np.repeat(np.arange(10), 6)
        episode_steps = np.tile(np.arange(6), 10)
        training, validation = _episode_split(episode_ids, 0.2, seed=7)
        self.assertFalse(np.intersect1d(training, validation).size)
        train_windows = _sequence_windows(
            episode_ids, episode_steps, training, length=4, stride=2
        )
        validation_windows = _sequence_windows(
            episode_ids, episode_steps, validation, length=4, stride=2
        )
        train_ids = np.unique(episode_ids[train_windows[train_windows >= 0]])
        validation_ids = np.unique(
            episode_ids[validation_windows[validation_windows >= 0]]
        )
        self.assertFalse(np.intersect1d(train_ids, validation_ids).size)

        expanded_ids = np.repeat(np.arange(20), 3)
        expanded_training, expanded_validation = _episode_split(
            expanded_ids, 0.2, seed=7
        )
        for episode_id in np.unique(episode_ids):
            was_validation = episode_id in validation
            is_validation = episode_id in expanded_validation
            self.assertEqual(was_validation, is_validation)
            self.assertEqual(episode_id in training, episode_id in expanded_training)

    def test_recurrent_windows_are_right_padded(self) -> None:
        episode_ids = np.zeros(3, dtype=np.int64)
        episode_steps = np.arange(3, dtype=np.int32)
        windows = _sequence_windows(
            episode_ids, episode_steps, np.asarray([0]), length=4, stride=1
        )
        np.testing.assert_array_equal(windows[0], [0, -1, -1, -1])
        np.testing.assert_array_equal(windows[1], [0, 1, -1, -1])

    def test_stage_one_loss_does_not_train_safety_head(self) -> None:
        model = VisualSetRecurrentQNetwork(
            VisualSetSpec(max_objects=4), width=16, attention_heads=4,
            safety_horizons=(0.1,),
        )
        objects = torch.zeros((2, 3, 4, 12))
        masks = torch.ones((2, 3, 4), dtype=torch.bool)
        globals_ = torch.zeros((2, 3, 8))
        policy, q_values, _, _ = model.forward_sequence(objects, masks, globals_)
        loss, _ = _cost_sensitive_loss(
            policy[:, -1], q_values[:, -1],
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, 9)),
            torch.zeros((2, 1, 9)),
            DAggerConfig(),
        )
        loss.backward()
        self.assertIsNone(model.collision_head.weight.grad)

    def test_scenario_block_is_exactly_stratified(self) -> None:
        block = ScenarioSampler(123).sample_block()
        sources = [item.source for item in block]
        resets = [item.reset_mode for item in block]
        self.assertEqual(sources.count("core"), 60)
        self.assertEqual(sources.count("broad"), 25)
        self.assertEqual(sources.count("stress"), 15)
        self.assertEqual(resets.count("deployment"), 70)
        self.assertEqual(resets.count("mid_episode"), 20)
        self.assertEqual(resets.count("recoverable_hard"), 10)
        broad = {(item.bullet_size, item.bullet_speed) for item in block if item.source == "broad"}
        self.assertEqual(len(broad), 25)

    def test_batched_env_preserves_final_observation_and_cause(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=1, max_episode_seconds=1.0 / 30.0,
            randomize_initial_phase=False,
        )
        batch = BatchedBarrageEnv([env])
        try:
            batch.reset(99)
            reset_frame = env._get_observation().copy()
            observations, _, terminated, truncated, final_observations, _ = (
                batch.step_detailed(np.asarray([0]))
            )
            self.assertFalse(bool(terminated[0]))
            self.assertTrue(bool(truncated[0]))
            self.assertFalse(np.array_equal(observations[0], final_observations[0]))
            self.assertEqual(observations[0].shape, reset_frame.shape)
        finally:
            batch.close()

    def test_nstep_bootstraps_truncation_not_termination(self) -> None:
        online = torch.zeros((1, 3, 2))
        online[:, :, 1] = 1.0
        target = torch.zeros((1, 3, 2))
        target[:, :, 1] = torch.tensor([1.0, 10.0, 20.0])
        rewards = torch.tensor([[2.0, 0.0]])
        valid = torch.tensor([[True, False]])
        terminal = torch.tensor([[1.0, 0.0]])
        truncation = torch.tensor([[0.0, 0.0]])
        terminated_target, _ = _n_step_double_q_targets(
            online, target, rewards, terminal, truncation, valid, 0.5, 1
        )
        truncated_target, _ = _n_step_double_q_targets(
            online, target, rewards, torch.zeros_like(terminal), terminal,
            valid, 0.5, 1
        )
        self.assertEqual(float(terminated_target[0, 0]), 2.0)
        self.assertEqual(float(truncated_target[0, 0]), 7.0)

    def test_safety_filter_rejects_unsafe_high_q(self) -> None:
        spec = VisualSetSpec(max_objects=4)
        model = VisualSetRecurrentQNetwork(
            spec, width=16, attention_heads=4, safety_horizons=(0.3, 0.6)
        )
        agent = ScreenOnlyAgent(
            model, SemanticFrameExtractor(spec), torch.device("cpu"),
            inference_head="q", safety_thresholds=(0.5, 0.5),
        )

        def fake_forward(objects, masks, globals_, hidden):
            batch = len(objects)
            policy = torch.zeros((batch, 9))
            q = torch.arange(9, dtype=torch.float32)[None].repeat(batch, 1)
            safety = torch.full((batch, 2, 9), -10.0)
            safety[:, :, 8] = 10.0
            return policy, q, safety, hidden

        model.forward_step = fake_forward
        action = agent.act_features(
            np.zeros((1, 4, 12), np.float32),
            np.zeros((1, 4), np.bool_),
            np.zeros((1, 8), np.float32),
        )
        self.assertEqual(int(action[0]), 7)

    def test_long_horizon_warning_does_not_hard_block_action(self) -> None:
        spec = VisualSetSpec(max_objects=4)
        model = VisualSetRecurrentQNetwork(
            spec, width=16, attention_heads=4, safety_horizons=(0.1, 1.2)
        )
        agent = ScreenOnlyAgent(
            model, SemanticFrameExtractor(spec), torch.device("cpu"),
            inference_head="q", safety_thresholds=(0.5, 0.5),
        )

        def fake_forward(objects, masks, globals_, hidden):
            batch = len(objects)
            policy = torch.zeros((batch, 9))
            q = torch.arange(9, dtype=torch.float32)[None].repeat(batch, 1)
            safety = torch.full((batch, 2, 9), -10.0)
            safety[:, 1, 8] = 10.0
            return policy, q, safety, hidden

        model.forward_step = fake_forward
        action = agent.act_features(
            np.zeros((1, 4, 12), np.float32),
            np.zeros((1, 4), np.bool_),
            np.zeros((1, 8), np.float32),
        )
        self.assertEqual(int(action[0]), 8)

    def test_safety_calibration_caps_false_negatives(self) -> None:
        probabilities = np.asarray(
            [[[0.10, 0.90]], [[0.20, 0.80]], [[0.30, 0.70]]], np.float32
        )
        targets = np.asarray(
            [[[1, 0]], [[1, 0]], [[1, 0]]], np.bool_
        )
        threshold = calibrate_safety_thresholds(
            probabilities, targets, maximum_false_negative_rate=0.0,
            maximum_all_unsafe_rate=1.0,
        )[0]
        prediction = probabilities[:, 0] >= threshold
        false_negative = targets[:, 0] & ~prediction
        self.assertEqual(int(false_negative.sum()), 0)

    def test_safety_calibration_limits_all_actions_blocked(self) -> None:
        probabilities = np.full((100, 1, 9), 0.05, np.float32)
        probabilities[:10, 0, 0] = 0.001
        targets = np.zeros_like(probabilities, dtype=np.bool_)
        targets[:10, 0, 0] = True
        threshold = calibrate_safety_thresholds(
            probabilities, targets, maximum_false_negative_rate=0.01,
            maximum_all_unsafe_rate=0.02,
        )[0]
        unsafe = probabilities[:, 0] >= threshold
        self.assertLessEqual(float(unsafe.all(axis=1).mean()), 0.02)

    def test_unusable_safety_head_is_not_enabled(self) -> None:
        probabilities = np.full((100, 1, 9), 0.05, np.float32)
        targets = np.zeros_like(probabilities, dtype=np.bool_)
        targets[:20, 0, :] = True
        thresholds = calibrate_safety_thresholds(
            probabilities, targets, 0.01, 0.02
        )
        self.assertFalse(
            safety_filter_is_usable(
                probabilities, targets, thresholds,
                minimum_positive_labels=10,
            )
        )

    def test_teacher_is_deterministic_and_does_not_mutate_rng(self) -> None:
        first = BarrageVisionEnv(
            bullet_count=2, randomize_initial_phase=False,
            targeted_bullet_probability=0.5,
        )
        try:
            first.reset(seed=10)
            first.bullet_positions[0] = (-1.0, 400.0)
            before = first.capture_state()
            first_result = privileged_planner_supervision(
                first, horizon_seconds=0.3, safety_horizons=(0.1, 0.3)
            )
            after = first.capture_state()
            repeated = privileged_planner_supervision(
                first, horizon_seconds=0.3, safety_horizons=(0.1, 0.3)
            )
            self.assertEqual(first_result.action, repeated.action)
            np.testing.assert_allclose(
                first_result.action_costs, repeated.action_costs
            )
            np.testing.assert_array_equal(before.plane_position, after.plane_position)
            np.testing.assert_array_equal(before.bullet_positions, after.bullet_positions)
            self.assertEqual(before.rng_state, after.rng_state)
        finally:
            first.close()

    def test_process_planner_task_matches_serial_planner(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=8,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            randomize_initial_phase=False,
            targeted_bullet_probability=0.35,
        )
        horizons = (0.10, 0.30, 0.60)
        try:
            env.reset(seed=321)
            expected = privileged_planner_supervision(
                env,
                horizon_seconds=0.60,
                reaction_seconds=0.30,
                safety_horizons=horizons,
            )
            _initialize_worker(
                _worker_environment_kwargs(env), 0.60, 0.30, horizons
            )
            actual = _supervise_task(_planner_task(env))
            self.assertEqual(actual.action, expected.action)
            np.testing.assert_array_equal(
                actual.safety_targets, expected.safety_targets
            )
            np.testing.assert_allclose(
                actual.action_costs, expected.action_costs, atol=1e-6
            )
            np.testing.assert_allclose(actual.regrets, expected.regrets, atol=1e-6)
        finally:
            env.close()

    def test_live_semanticizer_ignores_uniform_background_color(self) -> None:
        first = np.full((120, 120, 3), (10, 40, 90), dtype=np.uint8)
        second = np.full((120, 120, 3), (90, 20, 10), dtype=np.uint8)
        for image in (first, second):
            image[56:64, 56:64] = (255, 0, 0)
            image[20:23, 30:33] = (255, 255, 255)
        extractor = DominantBackgroundSemanticizer(output_size=96)
        semantic_first = extractor.convert(first)
        extractor.reset()
        semantic_second = extractor.convert(second)
        np.testing.assert_array_equal(semantic_first, semantic_second)
        self.assertGreater(np.count_nonzero(semantic_first == 255), 0)
        self.assertGreater(
            np.count_nonzero(
                (semantic_first > 160) & (semantic_first < 171)
            ),
            0,
        )
        self.assertGreater(np.count_nonzero(semantic_first == 96), 0)


if __name__ == "__main__":
    unittest.main()
