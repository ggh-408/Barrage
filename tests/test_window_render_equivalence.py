"""Pixel-exact regressions for batching the existing world renderer."""

import numpy as np
import pygame
import pytest

from barrage_rl.runtime_core import render_world_surface


def _reference_render(surface, plane, plane_position, bullet, positions):
    surface.fill((0, 0, 0))
    surface.blit(plane, plane.get_rect(center=tuple(plane_position)))
    for position in np.asarray(positions):
        surface.blit(bullet, bullet.get_rect(center=tuple(position)))


@pytest.mark.parametrize("depth", [16, 24, 32])
@pytest.mark.parametrize("bullet_size", [(5, 5), (6, 4)])
@pytest.mark.parametrize("count", [0, 1, 300])
def test_batch_render_preserves_rounding_clipping_and_alpha(depth, bullet_size, count):
    rng = np.random.default_rng(987)
    plane = pygame.Surface((17, 18), pygame.SRCALPHA, 32)
    plane.fill((13, 71, 201, 0))
    pygame.draw.rect(plane, (237, 59, 13, 155), (1, 1, 14, 15))
    bullet = pygame.Surface(bullet_size, pygame.SRCALPHA, 32)
    bullet.fill((201, 147, 35, 113))
    pygame.draw.line(bullet, (255, 255, 255, 255), (0, 0), (2, 3))
    positions = rng.uniform(-10.0, 75.0, size=(count, 2)).astype(np.float32)
    if count > 1:
        # Exercise fractional ties, screen edges and repeated alpha overlap.
        positions[:8] = [
            [-0.5, -0.5], [0.5, 0.5], [63.5, 48.5], [64.5, 49.5],
            [32.5, 24.5], [32.5, 24.5], [31.5, 23.5], [31.5, 23.5],
        ]
    expected = pygame.Surface((65, 50), depth=depth)
    actual = pygame.Surface((65, 50), depth=depth)
    expected.set_clip((1, 2, 62, 47))
    actual.set_clip((1, 2, 62, 47))
    plane_position = np.asarray([31.5, 23.5], dtype=np.float32)
    _reference_render(expected, plane, plane_position, bullet, positions)
    render_world_surface(actual, plane, plane_position, bullet, positions)
    np.testing.assert_array_equal(
        pygame.surfarray.array3d(actual), pygame.surfarray.array3d(expected)
    )
    assert actual.get_clip() == expected.get_clip()
    assert not actual.get_locked()


def test_batch_render_falls_back_for_single_blit_adapters():
    class SingleBlitSurface:
        def __init__(self):
            self.surface = pygame.Surface((65, 50))

        def fill(self, color):
            return self.surface.fill(color)

        def blit(self, source, destination):
            return self.surface.blit(source, destination)

    plane = pygame.Surface((17, 18), pygame.SRCALPHA, 32)
    plane.fill((255, 17, 31, 137))
    bullet = pygame.Surface((5, 5), pygame.SRCALPHA, 32)
    bullet.fill((255, 255, 255, 177))
    positions = np.asarray([[30.5, 23.5], [30.5, 23.5], [-1.5, 5.5]], np.float32)
    plane_position = np.asarray([31.5, 23.5], np.float32)
    expected = pygame.Surface((65, 50))
    actual = SingleBlitSurface()
    _reference_render(expected, plane, plane_position, bullet, positions)
    render_world_surface(actual, plane, plane_position, bullet, positions)
    np.testing.assert_array_equal(
        pygame.surfarray.array3d(actual.surface), pygame.surfarray.array3d(expected)
    )
