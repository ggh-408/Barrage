"""Action and recurrent-state parity for deployment and diagnostic calls."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from barrage_rl.distilled_student import (
    DistilledStudentNetwork,
    DistilledStudentSpec,
    UnifiedDistilledAgent,
)
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec


class DistilledRuntimeEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _agent() -> UnifiedDistilledAgent:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260831)
            backbone = ActionQueryPolicy(
                TrackedPolicySpec(), width=32, attention_layers=1,
                attention_heads=4,
            ).eval()
            student = DistilledStudentNetwork(DistilledStudentSpec(
                model_width=32, hidden_width=16,
            )).eval()
        return UnifiedDistilledAgent(
            backbone, student, torch.device("cpu"), use_safety_filter=True,
        )

    def test_action_only_matches_diagnostics_through_history_and_reset(self) -> None:
        generator = np.random.default_rng(20260831)
        objects = generator.normal(0.0, 0.1, (2, 384, 16)).astype(np.float32)
        objects[:, :, 15] = 1.0
        masks = np.zeros((2, 384), np.bool_)
        masks[1, :300] = True
        globals_ = np.zeros((2, 16), np.float32)
        globals_[:, :2] = [[0.5, 0.5], [0.01, 0.99]]
        globals_[:, 8:10] = 1.0
        actual = self._agent()
        expected = copy.deepcopy(actual)
        episode_ids = np.asarray([10, 20])
        for decision in range(4):
            indices = np.asarray([decision, decision])
            expected_actions, _ = expected.act_features_with_diagnostics(
                objects, masks, globals_, episode_indices=episode_ids,
                decision_indices=indices,
            )
            actual_actions = actual.act_features(
                objects, masks, globals_, episode_indices=episode_ids,
                decision_indices=indices,
            )
            np.testing.assert_array_equal(actual_actions, expected_actions)
            self.assertEqual(set(actual._distilled_hidden), set(expected._distilled_hidden))
            for key in actual._distilled_hidden:
                torch.testing.assert_close(
                    actual._distilled_hidden[key], expected._distilled_hidden[key],
                    rtol=0.0, atol=0.0,
                )
        actual.reset_state(np.asarray([10]))
        expected.reset_state(np.asarray([10]))
        self.assertEqual(set(actual._distilled_hidden), {20})
        actions, diagnostics = actual.act_features_with_diagnostics(
            objects, masks, globals_, episode_indices=episode_ids,
            decision_indices=np.asarray([4, 4]),
        )
        self.assertEqual(actions.shape, (2,))
        self.assertIn("raw_policy_actions", diagnostics)


if __name__ == "__main__":
    unittest.main()
