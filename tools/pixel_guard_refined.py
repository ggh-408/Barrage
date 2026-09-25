"""Opt-in guard refinement for interval conflicts and young image tracks."""
from dataclasses import replace
import time
import numpy as np
import torch
from barrage_rl.runtime_core import ACTION_VECTORS
from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig


class RefinedPixelGuard(PixelGuard):
    def hazards(self, objects, masks, globals_):
        nominal, possible = super().hazards(objects, masks, globals_)
        for row, o in enumerate(objects):
            young = masks[row] & (o[:, 15] < .5) & (o[:, 9] == 0) & (o[:, 8] >= .15)
            young &= np.linalg.norm(o[:, :2] * 820, axis=1) < 48
            if not young.any():
                continue
            observed = globals_[row, :2] * 820
            plane = observed - self.centroid_bias
            bullets = observed + o[young, :2] * 820 - self.bullet_bias
            for step in range(1, 5):
                positions = np.clip(plane + ACTION_VECTORS * (2 * step), self.half_size, 820-self.half_size)
                # Unknown direction: each component may move by speed * time.
                error = .5 + 2 * step
                lo = self.rounded(bullets-error)[None] - self.rounded(positions+.5-1e-5)[:, None]
                hi = self.rounded(bullets+error-1e-5)[None] - self.rounded(positions-.5)[:, None]
                possible[row] |= (self.rectangle_hits(lo, hi) > 0).any(axis=1)
        return nominal, possible

    def apply(self, selection, objects, masks, globals_):
        chosen = super().apply(selection, objects, masks, globals_)
        started = time.perf_counter()
        actions = chosen.actions.detach().cpu().numpy().copy()
        risk = selection.immediate_risk.detach().cpu().numpy()
        for row, o in enumerate(objects):
            young = masks[row] & (o[:,15] < .5) & (o[:,9] == 0) & (o[:,8] >= .15)
            young &= np.linalg.norm(o[:,:2]*820, axis=1) < 40
            if not young.any():
                continue
            from tools.pixel_recovery_refined import recovery_action
            actions[row] = recovery_action(self, o, masks[row], globals_[row], risk[row], compiled=True)
            self.counters['young_searches'] = self.counters.get('young_searches', 0) + 1
            self.counters['young_overrides'] = self.counters.get('young_overrides', 0) + int(actions[row] != int(chosen.actions[row]))
        counters=chosen.counter_values.clone()
        revised=torch.as_tensor(actions,device=chosen.actions.device)
        counters[4]=(revised != chosen.raw_actions).sum()
        self.elapsed_seconds += time.perf_counter() - started
        return replace(chosen,actions=revised,counter_values=counters)

    def manifest(self):
        return {**super().manifest(), 'variant': 'refined',
                'unobserved_or_unknown_velocity_tracks': 'currently observed young tracks use bounded-motion envelopes and early search; missed tracks retain incumbent behavior',
                'young_trigger_distance_pixels': 40,
                'unknown_velocity_component_bound': 240,
                'young_minimum_confidence': .15}


def install_refined_guard(agent):
    # Cover 0.133 s of mature-track motion before choosing an escape action.
    guard=RefinedPixelGuard(PixelGuardConfig(physics_steps=16,allow_imminent_escape=True,
        resolve_interval_conflicts=True,recovery_search=True,compiled_search=True))
    guard.safety_threshold=agent.safety_threshold
    original=agent._select_actions
    def select(objects,masks,globals_,*,deterministic):
        selection=original(objects,masks,globals_,deterministic=deterministic)
        return guard.apply(selection,objects.detach().cpu().numpy(),masks.detach().cpu().numpy(),globals_.detach().cpu().numpy())
    agent._select_actions=select
    return guard
