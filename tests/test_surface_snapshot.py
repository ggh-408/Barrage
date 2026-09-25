"""Pixel and ownership regressions for the live RGB capture path."""

import threading

import numpy as np
import pygame
import pytest

from barrage_rl.causal_control import CausalActionPipeline
from barrage_rl.runtime_core import snapshot_surface_rgb


@pytest.mark.parametrize("depth", [8, 16, 24, 32])
@pytest.mark.parametrize("subsurface", [False, True])
def test_snapshot_preserves_pixels_and_does_not_alias_surface(depth, subsurface):
    surface = pygame.Surface((47, 35), depth=depth)
    if depth == 8:
        surface.set_palette([(i, 255 - i, i // 2) for i in range(256)])
    rng = np.random.default_rng(173)
    pixels = rng.integers(0, 256, (47, 35, 3), dtype=np.uint8)
    if depth == 8:
        pygame.surfarray.blit_array(surface, pixels[:, :, 0])
    else:
        pygame.surfarray.blit_array(surface, pixels)
    if subsurface:
        surface = surface.subsurface((3, 7, 29, 19))
    expected = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    snapshot = snapshot_surface_rgb(surface)
    np.testing.assert_array_equal(snapshot, expected)
    assert snapshot.flags.c_contiguous
    assert not snapshot.flags.writeable
    assert not surface.get_locked()
    surface.fill((0, 0, 0))
    np.testing.assert_array_equal(snapshot, expected)


def test_snapshot_drops_alpha_without_premultiplying_rgb():
    surface = pygame.Surface((23, 17), pygame.SRCALPHA, 32)
    surface.fill((17, 93, 201, 0))
    pygame.draw.rect(surface, (250, 111, 7, 80), (2, 3, 5, 6))
    np.testing.assert_array_equal(
        snapshot_surface_rgb(surface),
        pygame.surfarray.array3d(surface).transpose(1, 0, 2),
    )


def test_snapshot_supports_pygame_before_tobytes_alias(monkeypatch):
    surface = pygame.Surface((23, 17))
    surface.fill((17, 93, 201))
    expected = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    monkeypatch.delattr(pygame.image, "tobytes")
    np.testing.assert_array_equal(snapshot_surface_rgb(surface), expected)


@pytest.mark.parametrize("source_kind", ["surface", "array"])
def test_queued_frame_survives_source_mutation(source_kind):
    class Controller:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.frames = []

        def reset(self):
            pass

        def act_rgb_frame(self, frame):
            self.started.set()
            if not self.release.wait(2.0):
                raise TimeoutError("capture ownership test timed out")
            self.frames.append(frame.copy())
            return 4

    controller = Controller()
    pipeline = CausalActionPipeline(controller)
    try:
        if source_kind == "surface":
            source = pygame.Surface((23, 17))
            source.fill((17, 93, 201))
            expected = snapshot_surface_rgb(source).copy()
            pipeline.submit_surface(source, capture_boundary=0, enforce_deadline=False)
        else:
            source = np.full((17, 23, 3), 41, dtype=np.uint8)
            expected = source.copy()
            pipeline.submit_rgb(source, capture_boundary=0, enforce_deadline=False)
        assert controller.started.wait(2.0)
        source.fill(0)
        controller.release.set()
        pipeline.wait_until_ready(1, timeout=2.0)
        assert pipeline.action_for_boundary(1, 0) == 4
        np.testing.assert_array_equal(controller.frames[0], expected)
    finally:
        controller.release.set()
        pipeline.close()
