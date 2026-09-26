"""Exact numerical and gradient regressions for policy latency optimizations."""

import copy
import unittest

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from barrage_rl.image_oracle import BulletTrack
from barrage_rl.tracked_policy import (
    ActionQueryPolicy,
    TrackedFeatureExtractor,
    TrackedPolicySpec,
)


def _reference_forward(model, objects, masks, globals_):
    """Original empty-row guard, retaining the same network and operation order."""
    geometry = model.prepare_action_geometry(
        objects, masks.bool(), globals_
    )
    safe_mask = masks.bool().clone()
    empty = ~safe_mask.any(dim=1)
    if torch.any(empty):
        safe_mask[empty, 0] = True
    encoded_objects = model.object_encoder(objects) * masks.unsqueeze(-1)
    action_ids = torch.arange(model.action_count, device=objects.device)
    queries = (
        model.action_embedding(action_ids)[None, :, :]
        + model.action_vector_encoder(model.action_vectors)[None, :, :]
        + model.global_encoder(globals_)[:, None, :]
        + model.geometry_encoder(
            model.action_geometry_features_from_shared(geometry)
        )
    ).expand(len(objects), -1, -1)
    for block in model.cross_attention:
        queries = block(queries, encoded_objects, safe_mask)
    return (
        model.policy_head(queries).squeeze(-1),
        model.teacher_cost_head(queries).squeeze(-1),
        queries.new_full((len(objects), len(model.safety_horizons), model.action_count), -1000.0),
    )


class PolicyLatencyEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def _assert_exact(self, actual, expected):
        torch.testing.assert_close(
            actual, expected, rtol=0.0, atol=0.0, equal_nan=True
        )

    def _inputs(self, device):
        generator = torch.Generator().manual_seed(20260831)
        objects = torch.randn(3, 384, 16, generator=generator) * 0.15
        objects[..., 15] = 1.0
        masks = torch.zeros(3, 384, dtype=torch.bool)
        masks[1, :300] = True
        masks[2, ::2] = True
        globals_ = torch.zeros(3, 16)
        globals_[:, :2] = torch.tensor([[0.5, 0.5], [0.02, 0.98], [0.4, 0.6]])
        globals_[:, 8:10] = 1.0
        return tuple(value.to(device) for value in (objects, masks, globals_))

    def test_feature_projection_matches_frozen_float32_fixture(self):
        # The frozen numerical fixture records legacy 250-bullet normalization.
        extractor = TrackedFeatureExtractor(TrackedPolicySpec(max_objects=6, expected_bullet_count=250))
        tracker = extractor.tracker
        tracker.plane_position[:] = [400.5, 395.25]
        tracker.plane_velocity[:] = [169.70563, -169.70563]
        for index, (position, velocity) in enumerate((
            ([403.5, 399.25], [-240.0, 0.0]),
            ([650.25, 220.5], [0.0, 240.0]),
            ([400.5, 395.25], [169.70563, -169.70563]),
            ([5.0, 818.75], [240.0, 0.0]),
        )):
            tracker.tracks.append(BulletTrack(
                track_id=index,
                position=np.asarray(position, np.float32),
                velocity=np.asarray(velocity, np.float32),
                velocity_known=index % 2 == 0,
                missed=index,
                confidence=0.9 - index * 0.1,
                age=5 + index,
                position_uncertainty=4.0 + index,
                occluded_steps=index % 2,
                association_group_size=index + 1,
            ))
        tracker.last_detection_count = 3
        tracker.last_ambiguous_track_count = 2
        # Original 120 Hz fixture. Preserve all float32 bits, including the
        # age projection and negative zero in the closing-rate feature.
        expected_objects = np.asarray([
            [0.0036585365887731314, 0.004878048785030842, -1.7071068286895752,
             0.7071067690849304, 0.006097560748457909, 0.4585787057876587,
             0.0005596441333182156, -0.06656431406736374, 0.8999999761581421,
             0.0, 0.1666666716337204, 0.8983226418495178, 0.0416666679084301,
             0.0, 0.125, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.11500000208616257,
             0.699999988079071, 0.25, 0.23333333432674408, 0.699999988079071,
             0.0625, 0.0, 0.375, 1.0],
            [0.3045731782913208, -0.21310976147651672, -0.7071067690849304,
             1.7071068286895752, 0.37172651290893555, 1.5580456256866455,
             0.11591658741235733, 1.5236496925354004, 0.800000011920929,
             0.125, 0.20000000298023224, 0.0061527579091489315,
             0.0520833320915699, 0.125, 0.25, 0.0],
            [-0.4823170602321625, 0.5164633989334106, 0.2928932309150696,
             0.7071067690849304, 0.7066570520401001, -0.3168827295303345,
             0.0, 5.6795878410339355, 0.6000000238418579, 0.375,
             0.2666666805744171, 3.337376952572413e-08, 0.0729166641831398,
             0.125, 0.5, 0.0],
            [0.0] * 16,
            [0.0] * 16,
        ], np.float32)
        expected_globals = np.asarray([
            0.4884146451950073, 0.48201218247413635, 0.4774390161037445,
            0.5006097555160522, 0.4710365831851959, 0.5070121884346008,
            0.7071067690849304, -0.7071067690849304, 0.5,
            0.01600000075995922, 0.012000000104308128, 0.9879999756813049,
            0.5, 0.5, 0.5, 0.0572916679084301,
        ], np.float32)
        objects, masks, globals_ = extractor._features()
        np.testing.assert_array_equal(objects.view(np.uint32), expected_objects.view(np.uint32))
        np.testing.assert_array_equal(masks, [True, True, True, True, False, False])
        np.testing.assert_array_equal(globals_, expected_globals)

    def test_empty_mask_guard_preserves_outputs_and_all_gradients(self):
        for device in self._devices():
            # CUDA fused attention backward may use nondeterministic reductions.
            # The math backend permits exact parameter-gradient comparisons.
            with self.subTest(device=device), sdpa_kernel(SDPBackend.MATH):
                torch.manual_seed(31)
                model = ActionQueryPolicy(
                    width=16, attention_layers=1, attention_heads=4
                ).to(device).train()
                reference = copy.deepcopy(model)
                objects, masks, globals_ = self._inputs(device)
                reference_objects = objects.clone().requires_grad_()
                reference_globals = globals_.clone().requires_grad_()
                objects.requires_grad_()
                globals_.requires_grad_()
                original_masks = masks.clone()
                actual = model(objects, masks, globals_)
                expected = _reference_forward(
                    reference, reference_objects, masks, reference_globals
                )
                for left, right in zip(actual, expected):
                    self._assert_exact(left, right)
                self._assert_exact(masks, original_masks)
                sum(value.square().sum() for value in actual).backward()
                sum(value.square().sum() for value in expected).backward()
                self._assert_exact(objects.grad, reference_objects.grad)
                self._assert_exact(globals_.grad, reference_globals.grad)
                for left, right in zip(model.parameters(), reference.parameters()):
                    self._assert_exact(left.grad, right.grad)

    def test_empty_mask_guard_preserves_default_backend_inference(self):
        for device in self._devices():
            with self.subTest(device=device), torch.inference_mode():
                torch.manual_seed(31)
                model = ActionQueryPolicy(
                    width=16, attention_layers=1, attention_heads=4
                ).to(device).eval()
                inputs = self._inputs(device)
                for actual, expected in zip(
                    model(*inputs), _reference_forward(model, *inputs)
                ):
                    self._assert_exact(actual, expected)

if __name__ == "__main__":
    unittest.main()
