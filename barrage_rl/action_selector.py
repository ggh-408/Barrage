"""Single arbitration boundary for image-policy action constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SharedGeometry(Protocol):
    normalized_minimum_clearance: torch.Tensor


@dataclass(frozen=True)
class WindowActionSelection:
    """Only the actions consumed by the pinned window planner."""
    actions: torch.Tensor
    raw_actions: torch.Tensor


@dataclass(frozen=True)
class ActionSelection:
    """One final action plus the shared data used to arbitrate it."""

    actions: torch.Tensor
    scores: torch.Tensor
    raw_actions: torch.Tensor
    immediate_risk: torch.Tensor
    all_unsafe: torch.Tensor
    analytic_clearance: torch.Tensor | None
    learned_actions: torch.Tensor
    collision_risk: torch.Tensor
    teacher_cost: torch.Tensor
    counter_values: torch.Tensor


class UnifiedActionSelector:
    """Select policy actions with optional image geometry, without learned risk.

    Legacy risk arguments and result slots remain inert adapters for old tools.
    Their values can never change an action.
    """

    def __init__(self, *, analytic_shield=False, analytic_horizon_index=None,
                 analytic_clearance_margin=0.0, teacher_cost_ranking_weight=0.0,
                 analytic_shield_gate="always", **legacy_options):
        retired = {"use_safety_filter", "safety_threshold", "analytic_min_tracked_count_ratio",
                   "analytic_max_model_risk_increase", "analytic_max_selected_violation",
                   "long_horizon_risk_weight"}
        if set(legacy_options) - retired:
            raise TypeError(f"Unknown selector options: {set(legacy_options) - retired}")
        self.analytic_shield = bool(analytic_shield and analytic_shield_gate == "always")
        self.analytic_horizon_index = analytic_horizon_index
        self.analytic_clearance_margin = float(analytic_clearance_margin)
        self.teacher_cost_ranking_weight = float(teacher_cost_ranking_weight)
        if self.teacher_cost_ranking_weight < 0:
            raise ValueError("teacher_cost_ranking_weight must be non-negative")

    def select(self, policy, teacher_cost, collision, masks, globals_, geometry,
               *, deterministic):
        raw_actions = policy.argmax(dim=1)
        scores = policy - self.teacher_cost_ranking_weight * teacher_cost
        unsafe = torch.zeros_like(policy, dtype=torch.bool)
        all_unsafe = torch.zeros(len(policy), dtype=torch.bool, device=policy.device)
        clearance = None
        if self.analytic_shield:
            if self.analytic_horizon_index is None:
                raise RuntimeError("analytic shield horizon was not initialized")
            clearance = geometry.normalized_minimum_clearance[:, :, self.analytic_horizon_index] * 100.0
            unsafe = (clearance <= self.analytic_clearance_margin) & masks.any(dim=1)[:, None]
            all_unsafe = unsafe.all(dim=1)
            masked = scores.masked_fill(unsafe, -torch.inf)
            scores = torch.where(all_unsafe[:, None], clearance + 1e-6 * scores, masked)
        actions = scores.argmax(dim=1) if deterministic else torch.distributions.Categorical(logits=scores).sample()
        zero = torch.zeros((), device=policy.device, dtype=torch.int64)
        counters = torch.stack((unsafe.sum(), all_unsafe.sum(), zero, zero, (actions != raw_actions).sum()))
        return ActionSelection(actions=actions, scores=scores, raw_actions=raw_actions,
            immediate_risk=torch.zeros_like(policy), all_unsafe=all_unsafe,
            analytic_clearance=clearance, learned_actions=raw_actions,
            collision_risk=torch.zeros_like(collision), teacher_cost=teacher_cost,
            counter_values=counters.to(torch.int64))
