"""Checks for the unified production action selector."""

import unittest
from types import SimpleNamespace

import torch

from barrage_rl.action_selector import UnifiedActionSelector


def _selector(*, analytic: bool = False) -> UnifiedActionSelector:
    return UnifiedActionSelector(
        use_safety_filter=True,
        safety_threshold=0.5,
        analytic_shield=analytic,
        analytic_horizon_index=0 if analytic else None,
        analytic_clearance_margin=0.0,
        analytic_shield_gate="always",
        analytic_min_tracked_count_ratio=0.90,
        analytic_max_model_risk_increase=0.005,
        analytic_max_selected_violation=0.50,
    )


class UnifiedActionSelectorTests(unittest.TestCase):
    def test_optional_ranking_uses_teacher_cost_only(self) -> None:
        selector = UnifiedActionSelector(
            use_safety_filter=True,
            safety_threshold=0.5,
            analytic_shield=False,
            analytic_horizon_index=None,
            analytic_clearance_margin=0.0,
            analytic_shield_gate="always",
            analytic_min_tracked_count_ratio=0.90,
            analytic_max_model_risk_increase=0.005,
            analytic_max_selected_violation=0.50,
            long_horizon_risk_weight=1.0,
            teacher_cost_ranking_weight=1.0,
        )
        policy = torch.tensor([[1.0, 0.9]])
        risks = torch.tensor([[[0.01, 0.01], [0.9, 0.1], [0.9, 0.1]]])
        selection = selector.select(
            policy,
            torch.tensor([[0.8, 0.1]]),
            torch.logit(risks),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 10),
            SimpleNamespace(normalized_minimum_clearance=torch.ones(1, 2, 1)),
            deterministic=True,
        )
        self.assertEqual(selection.raw_actions.item(), 0)
        self.assertEqual(selection.actions.item(), 1)

    def test_retired_learned_risk_cannot_mask_policy_maximum(self) -> None:
        policy = torch.tensor([[4.0, 3.0, 2.0]])
        collision = torch.full((1, 2, 3), -8.0)
        collision[:, 0, 0] = 8.0
        selection = _selector().select(
            policy,
            torch.zeros_like(policy),
            collision,
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 10),
            SimpleNamespace(normalized_minimum_clearance=torch.ones(1, 3, 1)),
            deterministic=True,
        )
        self.assertEqual(selection.raw_actions.item(), 0)
        self.assertEqual(selection.actions.item(), 0)
        self.assertEqual(selection.counter_values.tolist(), [0, 0, 0, 0, 0])

    def test_analytic_filter_uses_shared_geometry(self) -> None:
        policy = torch.tensor([[4.0, 3.0, 2.0]])
        collision = torch.full((1, 2, 3), -8.0)
        geometry = SimpleNamespace(
            normalized_minimum_clearance=torch.tensor([[[-0.1], [0.2], [0.1]]])
        )
        selection = _selector(analytic=True).select(
            policy,
            torch.zeros_like(policy),
            collision,
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 10),
            geometry,
            deterministic=True,
        )
        self.assertEqual(selection.actions.item(), 1)
        self.assertEqual(selection.counter_values.tolist(), [1, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
