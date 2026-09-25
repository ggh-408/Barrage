"""Experimental image-feature guard; no access to a running environment."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
import time

import numpy as np
import pygame
import torch

from barrage_rl.runtime_core import ACTION_VECTORS
from barrage_rl.live_screen import DominantBackgroundSemanticizer

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PixelGuardConfig:
    physics_steps: int = 4
    use_intervals: bool = True
    correct_centroid: bool = True
    # Pixel cells give +/-0.5 px uncertainty for each observed center.
    pixel_half_width: float = 0.5
    velocity_error_pixels: float = 2.0
    allow_imminent_escape: bool = False
    resolve_interval_conflicts: bool = False
    recovery_search: bool = False
    compiled_search: bool = False


class PixelGuard:
    """Check actual sprite footprints over the next committed action interval.

    Templates are static visual calibration assets. Every dynamic input comes
    from the existing image extractor. Uncertain rectangles are queried against
    a summed-area mask table, not replaced by an enlarged collision circle.
    These intervals are a sensitivity model, not a certified state bound.
    """

    def __init__(self, config: PixelGuardConfig):
        if config.physics_steps < 1:
            raise ValueError("physics_steps must be positive")
        self.config = config
        self.plane = pygame.image.load(str(ROOT / "image/plane(0).gif"))
        self.bullet = pygame.image.load(str(ROOT / "image/bullet(5).gif"))
        self.plane_mask = pygame.mask.from_surface(self.plane)
        self.bullet_mask = pygame.mask.from_surface(self.bullet)
        self.half_size = np.asarray(self.plane.get_size(), np.float32) / 2
        frame = pygame.Surface((100, 100))
        frame.fill((0, 0, 0))
        frame.blit(self.plane, self.plane.get_rect(center=(30, 30)))
        frame.blit(self.bullet, self.bullet.get_rect(center=(70, 70)))
        detections = DominantBackgroundSemanticizer().detect(
            pygame.surfarray.array3d(frame).transpose(1, 0, 2), include_semantic=False
        )
        self.centroid_bias = detections.plane_position * 100 - 30
        self.bullet_bias = detections.bullet_positions[0] * 100 - 70
        self.radius = 32
        size = self.radius * 2 + 1
        self.table = np.zeros((size, size), dtype=np.int32)
        plane_rect = self.plane.get_rect(center=(100, 100))
        for y in range(-self.radius, self.radius + 1):
            for x in range(-self.radius, self.radius + 1):
                rect = self.bullet.get_rect(center=(100 + x, 100 + y))
                self.table[y + self.radius, x + self.radius] = self.plane_mask.overlap(
                    self.bullet_mask, (rect.left - plane_rect.left, rect.top - plane_rect.top)
                ) is not None
        self.integral = np.pad(self.table, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.counters = dict(decisions=0, gates=0, overrides=0, no_robust_alternative=0,
                             imminent_escape_overrides=0)
        self.elapsed_seconds = 0.0
        if config.compiled_search:
            from tools.pixel_search_kernel import warm_search
            warm_search(self)

    @staticmethod
    def rounded(values):
        # Pygame Rect center rounds positive world coordinates to nearest integer.
        return np.floor(values + 0.5).astype(np.int32)

    def rectangle_hits(self, lower, upper):
        lo = np.clip(lower + self.radius, 0, self.table.shape[0])
        hi = np.clip(upper + self.radius + 1, 0, self.table.shape[0])
        ix = self.integral
        hits = ix[hi[..., 1], hi[..., 0]] - ix[lo[..., 1], hi[..., 0]]
        hits -= ix[hi[..., 1], lo[..., 0]] - ix[lo[..., 1], lo[..., 0]]
        return hits

    def hazards(self, objects, masks, globals_):
        count = len(objects)
        possible = np.zeros((count, 9), dtype=bool)
        nominal = np.zeros_like(possible)
        for row in range(count):
            o = objects[row]
            # Only established, currently observed image tracks support hard overrides.
            valid = masks[row] & (o[:, 15] > 0.5) & (o[:, 9] == 0) & (o[:, 8] >= 0.5)
            reach = 20 + 2 * 240 * self.config.physics_steps / 120 + 4
            valid &= np.linalg.norm(o[:, :2] * 820, axis=1) <= reach
            if not valid.any():
                continue
            g = globals_[row]
            observed_plane = g[:2] * 820
            bias = self.centroid_bias if self.config.correct_centroid else np.zeros(2)
            plane = observed_plane - bias
            bullets = observed_plane[None] + o[valid, :2] * 820 - self.bullet_bias
            velocity = (o[valid, 2:4] + g[None, 6:8]) * 240
            age = np.maximum(o[valid, 10], 2 / 30)
            velocity_error = self.config.velocity_error_pixels / age
            positions = np.repeat(plane[None], 9, axis=0).astype(np.float32)
            bullet_positions = bullets.astype(np.float32).copy()
            for step in range(1, self.config.physics_steps + 1):
                positions += ACTION_VECTORS * 240 / 120
                positions = np.clip(positions, self.half_size, 820 - self.half_size)
                bullet_positions += velocity / 120
                center = self.rounded(bullet_positions)[None] - self.rounded(positions)[:, None]
                nominal[row] |= (self.rectangle_hits(center, center) > 0).any(axis=1)
                if self.config.use_intervals:
                    # Half-open rounding cells; epsilon excludes an impossible upper tie.
                    pe = self.config.pixel_half_width
                    be = pe + velocity_error[:, None] * (step / 120)
                    lower = self.rounded(bullet_positions - be)[None] - self.rounded(positions + pe - 1e-5)[:, None]
                    upper = self.rounded(bullet_positions + be - 1e-5)[None] - self.rounded(positions - pe)[:, None]
                    possible[row] |= (self.rectangle_hits(lower, upper) > 0).any(axis=1)
                else:
                    possible[row] |= nominal[row]
        return nominal, possible

    def apply(self, selection, objects, masks, globals_):
        started = time.perf_counter()
        nominal, possible = self.hazards(objects, masks, globals_)
        actions = selection.actions.detach().cpu().numpy().copy()
        scores = selection.scores.detach().cpu().numpy()
        raw_risk = selection.immediate_risk.detach().cpu().numpy()
        policy_safe = raw_risk < self.safety_threshold
        for row, action in enumerate(actions):
            self.counters["decisions"] += 1
            if self.config.recovery_search and bool(selection.all_unsafe[row]):
                from tools.pixel_recovery_planner import recovery_action
                actions[row] = recovery_action(self, objects[row], masks[row], globals_[row], raw_risk[row],
                                               compiled=self.config.compiled_search)
                self.counters["gates"] += 1
                self.counters["overrides"] += int(actions[row] != action)
                continue
            if not possible[row, action]:
                continue
            self.counters["gates"] += 1
            eligible = ~possible[row] & policy_safe[row] & np.isfinite(scores[row])
            if eligible.any():
                actions[row] = int(np.argmax(np.where(eligible, scores[row], -np.inf)))
            elif self.config.resolve_interval_conflicts and (~possible[row]).any():
                candidates = np.flatnonzero(~possible[row])
                actions[row] = int(candidates[np.argmin(raw_risk[row, candidates])])
                self.counters["imminent_escape_overrides"] += 1
            elif self.config.resolve_interval_conflicts and nominal[row, action] and (~nominal[row]).any():
                # A possible collision is an uncertainty flag, not a proof that
                # every route is blocked. Retain nominally feasible escape routes.
                candidates = np.flatnonzero(~nominal[row])
                actions[row] = int(candidates[np.argmin(raw_risk[row, candidates])])
                self.counters["imminent_escape_overrides"] += 1
            elif self.config.allow_imminent_escape and nominal[row, action] and (~possible[row]).any():
                # The incumbent predicts a collision in this committed interval.
                # A mask-safe action may be marked unsafe by the longer 0.1 s
                # learned target. Prefer its lowest learned immediate risk.
                candidates = np.flatnonzero(~possible[row])
                actions[row] = int(candidates[np.argmin(raw_risk[row, candidates])])
                self.counters["imminent_escape_overrides"] += 1
            else:
                # Preserve the established fallback when the uncertainty model
                # has eliminated all acceptable routes; do not freeze the plane.
                self.counters["no_robust_alternative"] += 1
            self.counters["overrides"] += int(actions[row] != action)
        revised = torch.as_tensor(actions, device=selection.actions.device)
        counters = selection.counter_values.clone()
        counters[4] = (revised != selection.raw_actions).sum()
        self.elapsed_seconds += time.perf_counter() - started
        return replace(selection, actions=revised, counter_values=counters)

    def manifest(self):
        return {"config": asdict(self.config), "centroid_bias": self.centroid_bias.tolist(),
                "bullet_bias": self.bullet_bias.tolist(), "counters": dict(self.counters),
                "guard_seconds": self.elapsed_seconds,
                "dynamic_input": "current image-derived features only",
                "unobserved_or_unknown_velocity_tracks": "remain governed by incumbent policy"}


def install_guard(agent, config):
    guard = PixelGuard(config)
    guard.safety_threshold = agent.safety_threshold
    original = agent._select_actions

    def select(objects, masks, globals_, *, deterministic):
        selection = original(objects, masks, globals_, deterministic=deterministic)
        return guard.apply(selection, objects.detach().cpu().numpy(),
                           masks.detach().cpu().numpy(), globals_.detach().cpu().numpy())

    agent._select_actions = select
    return guard
