"""Action-conditioned tracked policy invariants."""

import gc
import unittest
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from barrage_rl.env import BarrageVisionEnv
from barrage_rl.live_screen import LiveVisualController
from barrage_rl.tracked_policy import (
    ActionQueryPolicy,
    TrackedFeatureExtractor,
    TrackedPolicyAgent,
    TrackedPolicySpec,
)


class TrackedPolicyTests(unittest.TestCase):
    def test_optional_action_hysteresis_holds_a_near_tied_safe_action(self) -> None:
        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, objects, masks, globals_):
                batch = len(objects)
                policy = torch.tensor(
                    [1.0, 0.98] if self.calls == 0 else [0.98, 1.0]
                ).repeat(batch, 1)
                self.calls += 1
                risks = torch.full((batch, 4, 2), 0.01)
                return policy, torch.zeros_like(policy), torch.logit(risks)

        model = StubModel()
        agent = TrackedPolicyAgent(
            model,
            torch.device("cpu"),
            use_safety_filter=True,
            action_hysteresis_bonus=0.05,
        )
        features = np.zeros((1, 1, 1), np.float32)
        masks = np.ones((1, 1), np.bool_)
        globals_ = np.zeros((1, 1), np.float32)
        first = agent.act_features(features, masks, globals_, episode_indices=[7])
        second = agent.act_features(features, masks, globals_, episode_indices=[7])
        np.testing.assert_array_equal(first, np.asarray([0]))
        np.testing.assert_array_equal(second, np.asarray([0]))

    def test_default_spec_represents_300_bullets_with_occlusion_headroom(self) -> None:
        spec = TrackedPolicySpec()
        self.assertEqual(spec.expected_bullet_count, 300)
        self.assertEqual(spec.max_objects, 384)
        self.assertEqual(spec.max_objects - spec.expected_bullet_count, 84)
        self.assertGreater(spec.tracker_capacity, spec.expected_bullet_count)
        self.assertGreaterEqual(spec.object_features, 16)
        self.assertGreaterEqual(spec.global_features, 16)

    def test_retired_checkpoint_threshold_cannot_mask_actions(self) -> None:
        class StubModel(torch.nn.Module):
            def forward(self, objects, masks, globals_):
                batch = len(objects)
                policy = torch.zeros(batch, 9)
                policy[:, 0] = 2.0
                risks = torch.full((batch, 4, 9), 0.001)
                risks[:, 0, 0] = 0.012
                collision = torch.logit(risks)
                return policy, torch.zeros_like(policy), collision

        agent = TrackedPolicyAgent(
            StubModel(),
            torch.device("cpu"),
            use_safety_filter=True,
            safety_threshold=0.01,
        )
        actions = agent.act_features(
            np.zeros((2, 1, 1), np.float32),
            np.ones((2, 1), np.bool_),
            np.zeros((2, 1), np.float32),
        )
        np.testing.assert_array_equal(actions, np.asarray([0, 0]))
        diagnostic_actions, diagnostics = agent.act_features_with_diagnostics(
            np.zeros((2, 1, 1), np.float32),
            np.ones((2, 1), np.bool_),
            np.zeros((2, 1), np.float32),
        )
        np.testing.assert_array_equal(diagnostic_actions, actions)
        np.testing.assert_array_equal(
            diagnostics["raw_policy_actions"], np.asarray([0, 0])
        )
        np.testing.assert_array_equal(
            diagnostics["action_was_filtered"], np.asarray([False, False])
        )
        np.testing.assert_allclose(
            diagnostics["raw_immediate_risk"], np.asarray([0.0, 0.0]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            diagnostics["selected_immediate_risk"], np.asarray([0.0, 0.0]),
            rtol=1e-5,
        )
        np.testing.assert_array_equal(
            diagnostics["learned_filter_teacher_cost"], np.zeros(2)
        )
        np.testing.assert_array_equal(
            diagnostics["selected_teacher_cost"], np.zeros(2)
        )

    def test_analytic_shield_replaces_unsafe_raw_action_without_training(self) -> None:
        class StubModel(torch.nn.Module):
            safety_horizons = (0.05, 0.10, 0.20, 0.35)

            def forward(self, objects, masks, globals_):
                batch = len(objects)
                policy = torch.zeros(batch, 9)
                policy[:, 0] = 2.0
                policy[:, 1] = 1.0
                collision = torch.zeros(batch, 4, 9)
                return policy, torch.zeros_like(policy), collision

            def action_geometry(self, objects, masks, globals_):
                geometry = torch.ones(len(objects), 9, 4)
                geometry[:, 0, 1] = -0.05
                geometry[:, 1, 1] = 0.02
                return geometry

        agent = TrackedPolicyAgent(
            StubModel(), torch.device("cpu"), analytic_shield=True
        )
        actions, diagnostics = agent.act_features_with_diagnostics(
            np.zeros((2, 1, 1), np.float32),
            np.ones((2, 1), np.bool_),
            np.zeros((2, 1), np.float32),
        )
        np.testing.assert_array_equal(actions, np.asarray([1, 1]))
        np.testing.assert_allclose(
            diagnostics["raw_analytic_clearance"], np.asarray([-5.0, -5.0])
        )
        np.testing.assert_allclose(
            diagnostics["selected_analytic_clearance"], np.asarray([2.0, 2.0])
        )
        self.assertEqual(agent.safety_filter_mode, "analytic_geometry")
        self.assertEqual(agent.overridden_decision_count, 2)
        self.assertEqual(agent.all_unsafe_count, 0)

    def test_analytic_shield_fallback_maximizes_clearance_when_all_unsafe(self) -> None:
        class StubModel(torch.nn.Module):
            safety_horizons = (0.05, 0.10)

            def forward(self, objects, masks, globals_):
                policy = torch.zeros(len(objects), 9)
                policy[:, 0] = 2.0
                collision = torch.zeros(len(objects), 2, 9)
                return policy, torch.zeros_like(policy), collision

            def action_geometry(self, objects, masks, globals_):
                geometry = torch.full((len(objects), 9, 2), -0.10)
                geometry[:, 4, 1] = -0.01
                return geometry

        agent = TrackedPolicyAgent(
            StubModel(), torch.device("cpu"), analytic_shield=True
        )
        globals_ = np.zeros((1, 16), np.float32)
        globals_[:, 9] = 1.0
        actions = agent.act_features(
            np.zeros((1, 1, 1), np.float32),
            np.ones((1, 1), np.bool_),
            globals_,
        )
        np.testing.assert_array_equal(actions, np.asarray([4]))
        self.assertEqual(agent.all_unsafe_count, 1)







    def test_live_controller_loads_tracked_image_only_checkpoint(self) -> None:
        spec = TrackedPolicySpec(max_objects=8)
        model = ActionQueryPolicy(spec).eval()
        checkpoint = {
            "model": model.state_dict(),
            "model_version": model.model_version,
            "model_hparams": {
                "action_count": model.action_count,
                "width": model.width,
                "attention_layers": model.attention_layers,
                "attention_heads": model.attention_heads,
                "safety_horizons": model.safety_horizons,
            },
            "tracked_policy_spec": asdict(spec),
            "config": {},
            "observation_size": 192,
            "use_safety_filter": True,
            "safety_threshold": 0.5,
        }
        rgb = np.zeros((820, 820, 3), np.uint8)
        rgb[404:416, 401:419] = (255, 0, 0)
        rgb[20:25, 20:25] = 255
        rgb[20, 20] = rgb[20, 24] = rgb[24, 20] = rgb[24, 24] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracked.pt"
            torch.save(checkpoint, path)
            controller = LiveVisualController(str(path), device_name="cpu")
            with patch("gc.collect", wraps=gc.collect) as collect:
                first = controller.prime_rgb(rgb)
                collect.assert_called_once_with(2)
                second = controller.observe_rgb(rgb, physics_steps=4)
                collect.assert_called_once_with(2)
        self.assertIn(first, range(9))
        self.assertIn(second, range(9))
        self.assertEqual(controller.semanticizer.output_size, 192)
        self.assertTrue(controller.tracked_initialized)
        self.assertEqual(controller.action_delay_steps, 0)
        self.assertTrue(controller.tracked_extractor.tracker.refit_known_velocity)
        self.assertFalse(controller.agent.analytic_shield)
        self.assertEqual(
            controller.agent.analytic_shield_gate, "always"
        )

        controller.reset()
        self.assertTrue(controller.tracked_extractor.tracker.refit_known_velocity)

    def test_live_controller_uses_saved_causal_timing(self) -> None:
        spec = TrackedPolicySpec(max_objects=8)
        model = ActionQueryPolicy(spec).eval()
        checkpoint = {
            "model": model.state_dict(),
            "model_version": model.model_version,
            "model_hparams": {
                "action_count": model.action_count,
                "width": model.width,
                "attention_layers": model.attention_layers,
                "attention_heads": model.attention_heads,
                "safety_horizons": model.safety_horizons,
            },
            "tracked_policy_spec": asdict(spec),
            "config": {
                "action_repeat": 4,
                "evaluation_causal_action_delay_steps": 1,
            },
            "observation_size": 192,
            "use_safety_filter": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracked.pt"
            torch.save(checkpoint, path)
            controller = LiveVisualController(
                str(path), device_name="cpu", analytic_shield=False
            )
        self.assertEqual(controller.action_delay_steps, 1)

    def test_live_controller_skips_surface_copy_between_decisions(self) -> None:
        spec = TrackedPolicySpec(max_objects=8)
        model = ActionQueryPolicy(spec).eval()
        checkpoint = {
            "model": model.state_dict(),
            "model_version": model.model_version,
            "model_hparams": {
                "action_count": model.action_count,
                "width": model.width,
                "attention_layers": model.attention_layers,
                "attention_heads": model.attention_heads,
                "safety_horizons": model.safety_horizons,
            },
            "tracked_policy_spec": asdict(spec),
            "config": {"action_repeat": 4, "bullet_size": 5},
            "observation_size": 192,
            "use_safety_filter": False,
        }
        rgb = np.zeros((820, 820, 3), np.uint8)
        rgb[404:416, 401:419] = (255, 0, 0)
        rgb[20:25, 20:25] = 255
        rgb[20, 20] = rgb[20, 24] = rgb[24, 20] = rgb[24, 24] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracked.pt"
            torch.save(checkpoint, path)
            controller = LiveVisualController(
                str(path), device_name="cpu", analytic_shield=False
            )
            controller.prime_rgb(rgb)
            with patch("barrage_rl.live_screen.snapshot_surface_rgb", return_value=rgb) as copy:
                for _ in range(3):
                    controller.observe_surface(object(), physics_steps=1)
                copy.assert_not_called()
                controller.observe_surface(object(), physics_steps=1)
                copy.assert_called_once()

    def test_live_controller_can_split_due_step_from_surface_copy(self) -> None:
        controller = LiveVisualController.__new__(LiveVisualController)
        controller.frame_counter = 0
        controller.decision_interval = 4
        self.assertFalse(controller.decision_due(1))
        self.assertFalse(controller.decision_due(1))
        self.assertFalse(controller.decision_due(1))
        self.assertTrue(controller.decision_due(1))
        self.assertEqual(controller.frame_counter, 0)

    def test_feature_extractor_is_image_only_and_capacity_safe(self) -> None:
        env = BarrageVisionEnv(
            bullet_count=300,
            observation_size=192,
            targeted_bullet_probability=0.10,
            randomize_initial_phase=False,
        )
        try:
            observation, _ = env.reset(seed=1_600_000)
            extractor = TrackedFeatureExtractor()
            objects, mask, globals_ = extractor.reset(observation)
            spec = extractor.spec
            self.assertEqual(
                objects.shape, (spec.max_objects, spec.object_features)
            )
            self.assertEqual(mask.shape, (spec.max_objects,))
            self.assertEqual(globals_.shape, (spec.global_features,))
            self.assertGreaterEqual(int(mask.sum()), 20)
            self.assertTrue(np.isfinite(objects).all())
            self.assertTrue(np.isfinite(globals_).all())
        finally:
            env.close()

    def test_action_query_network_is_permutation_invariant(self) -> None:
        torch.manual_seed(7)
        spec = TrackedPolicySpec()
        model = ActionQueryPolicy(spec).eval()
        objects = torch.randn(2, spec.max_objects, spec.object_features)
        mask = torch.zeros(2, spec.max_objects, dtype=torch.bool)
        mask[:, :150] = True
        globals_ = torch.randn(2, spec.global_features)
        permutation = torch.randperm(spec.max_objects)
        with torch.inference_mode():
            geometry = model.action_geometry(objects, mask, globals_)
            geometry_features = model.action_geometry_features(
                objects, mask, globals_
            )
            original = model(objects, mask, globals_)
            permuted = model(
                objects[:, permutation], mask[:, permutation], globals_
            )
        self.assertEqual(original[0].shape, (2, 9))
        self.assertEqual(geometry.shape, (2, 9, 4))
        self.assertEqual(geometry_features.shape, (2, 9, 12))
        self.assertEqual(original[1].shape, (2, 9))
        self.assertEqual(original[2].shape, (2, 4, 9))
        for left, right in zip(original, permuted):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)

    def test_action_geometry_uses_closest_approach_inside_horizon(self) -> None:
        spec = TrackedPolicySpec(max_objects=1)
        model = ActionQueryPolicy(spec, safety_horizons=(0.30,)).eval()
        objects = torch.zeros(1, 1, spec.object_features)
        mask = torch.ones(1, 1, dtype=torch.bool)
        globals_ = torch.zeros(1, spec.global_features)
        # A bullet starts 24 source pixels to the right and moves left at
        # 240 px/s.  At 0.30 s its endpoint is safely 48 px past the plane,
        # but it crosses the plane at 0.10 s and must remain unsafe.
        objects[0, 0, 0] = 24.0 / spec.source_size
        objects[0, 0, 2] = -1.0
        with torch.inference_mode():
            geometry = model.action_geometry(objects, mask, globals_)
        self.assertLess(float(geometry[0, 0, 0]), 0.0)
        self.assertAlmostEqual(
            float(geometry[0, 0, 0]),
            -spec.collision_radius / 100.0,
            places=5,
        )

    def test_action_geometry_exposes_simultaneous_danger_mass(self) -> None:
        spec = TrackedPolicySpec(max_objects=2)
        model = ActionQueryPolicy(spec, safety_horizons=(0.30,)).eval()
        objects = torch.zeros(2, 2, spec.object_features)
        masks = torch.asarray([[True, False], [True, True]])
        globals_ = torch.zeros(2, spec.global_features)
        objects[:, :, 0] = 24.0 / spec.source_size
        objects[:, :, 2] = -1.0
        with torch.inference_mode():
            features = model.action_geometry_features(objects, masks, globals_)
        # Feature order is minimum, danger_mass, blocked_fraction.  Minimum is
        # unchanged for coincident threats while mass and blockage must grow.
        self.assertAlmostEqual(
            float(features[0, 0, 0]), float(features[1, 0, 0]), places=6
        )
        self.assertGreater(float(features[1, 0, 1]), float(features[0, 0, 1]))
        self.assertGreater(float(features[1, 0, 2]), float(features[0, 0, 2]))


if __name__ == "__main__":
    unittest.main()
