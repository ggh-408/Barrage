"""Static sprite calibration and the exact four-step image commitment check."""
from pathlib import Path
import numpy as np
import pygame
from .runtime_core import ACTION_VECTORS
from .live_screen import DominantBackgroundSemanticizer

ROOT = Path(__file__).resolve().parents[1]


class WindowPixelGeometry:
    def __init__(self):
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

    def hazards(self, objects, masks, globals_, *, actions=None, include_nominal=True):
        count = len(objects)
        width = 9 if actions is None else 1
        possible = np.zeros((count, width), dtype=bool)
        nominal = np.zeros_like(possible) if include_nominal else None
        for row in range(count):
            o = objects[row]
            vectors = ACTION_VECTORS if actions is None else ACTION_VECTORS[[int(actions[row])]]
            # Only established, currently observed image tracks support hard overrides.
            valid = masks[row] & (o[:, 15] > 0.5) & (o[:, 9] == 0) & (o[:, 8] >= 0.5)
            reach = 20 + 2 * 240 * 4 / 120 + 4
            valid &= np.linalg.norm(o[:, :2] * 820, axis=1) <= reach
            if not valid.any():
                continue
            g = globals_[row]
            observed_plane = g[:2] * 820
            plane = observed_plane - self.centroid_bias
            bullets = observed_plane[None] + o[valid, :2] * 820 - self.bullet_bias
            velocity = (o[valid, 2:4] + g[None, 6:8]) * 240
            age = np.maximum(o[valid, 10], 2 / 30)
            velocity_error = 2.0 / age
            positions = np.repeat(plane[None], width, axis=0).astype(np.float32)
            bullet_positions = bullets.astype(np.float32).copy()
            for step in range(1, 5):
                positions += vectors * 240 / 120
                positions = np.clip(positions, self.half_size, 820 - self.half_size)
                bullet_positions += velocity / 120
                if include_nominal:
                    center = self.rounded(bullet_positions)[None] - self.rounded(positions)[:, None]
                    nominal[row] |= (self.rectangle_hits(center, center) > 0).any(axis=1)
                # Half-open rounding cells; epsilon excludes an impossible upper tie.
                pe = 0.5
                be = pe + velocity_error[:, None] * (step / 120)
                lower = self.rounded(bullet_positions - be)[None] - self.rounded(positions + pe - 1e-5)[:, None]
                upper = self.rounded(bullet_positions + be - 1e-5)[None] - self.rounded(positions - pe)[:, None]
                possible[row] |= (self.rectangle_hits(lower, upper) > 0).any(axis=1)
        for row, o in enumerate(objects):
            vectors = ACTION_VECTORS if actions is None else ACTION_VECTORS[[int(actions[row])]]
            young = masks[row] & (o[:, 15] < .5) & (o[:, 9] == 0) & (o[:, 8] >= .15)
            young &= np.linalg.norm(o[:, :2] * 820, axis=1) < 48
            if not young.any():
                continue
            observed = globals_[row, :2] * 820
            plane = observed - self.centroid_bias
            bullets = observed + o[young, :2] * 820 - self.bullet_bias
            for step in range(1, 5):
                positions = np.clip(plane + vectors * (2 * step), self.half_size, 820-self.half_size)
                # Unknown direction: each component may move by speed * time.
                error = .5 + 2 * step
                lo = self.rounded(bullets-error)[None] - self.rounded(positions+.5-1e-5)[:, None]
                hi = self.rounded(bullets+error-1e-5)[None] - self.rounded(positions-.5)[:, None]
                possible[row] |= (self.rectangle_hits(lo, hi) > 0).any(axis=1)
        return nominal, possible

