"""Pygame screen adapter for the trained visual-set policy.

This module does not import or inspect ``Bullet.LIST`` or plane state.  It sees
only RGB pixels from the already-rendered Pygame surface and the actions it
previously emitted.
"""

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch

from .evaluate_visual_set import load_agent


def _foreground_components(mask: np.ndarray) -> List[Tuple[float, float, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: List[Tuple[float, float, int]] = []
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        xs: List[int] = []
        ys: List[int] = []
        while stack:
            cy, cx = stack.pop()
            xs.append(cx)
            ys.append(cy)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append((float(np.mean(xs)), float(np.mean(ys)), len(xs)))
    return components


class DominantBackgroundSemanticizer:
    """Turn a rendered RGB frame into the semantic frames used for training.

    The dominant/median color is treated as background, so uniform background
    color changes do not require retraining.  Sprite color is not hard-coded;
    the player is identified by temporal proximity to its previous center.
    """

    def __init__(self, output_size: int = 96, color_threshold: float = 28.0) -> None:
        self.output_size = int(output_size)
        self.color_threshold = float(color_threshold)
        self.previous_plane: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_plane = None

    def convert(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("rgb frame must have shape [height, width, channels]")
        height, width = image.shape[:2]
        background = np.median(image[:, :, :3].reshape(-1, 3), axis=0)
        difference = image[:, :, :3].astype(np.float32) - background
        foreground = np.linalg.norm(difference, axis=2) >= self.color_threshold
        components = [item for item in _foreground_components(foreground) if item[2] >= 2]
        semantic = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
        if not components:
            return semantic

        normalized = np.asarray(
            [[x / max(width - 1, 1), y / max(height - 1, 1)] for x, y, _ in components],
            dtype=np.float32,
        )
        reference = (
            self.previous_plane
            if self.previous_plane is not None
            else np.asarray([0.5, 0.5], dtype=np.float32)
        )
        areas = np.asarray([area for _, _, area in components], dtype=np.float32)
        # Prefer the component near the previous player position; a mild area
        # prior breaks ties with bullets passing through the same neighborhood.
        distances = np.linalg.norm(normalized - reference[None, :], axis=1)
        area_bonus = 0.02 * np.clip(areas / max(float(np.median(areas)), 1.0), 0.0, 4.0)
        plane_index = int(np.argmin(distances - area_bonus))
        plane = normalized[plane_index]
        self.previous_plane = plane

        def draw_mark(position: np.ndarray, value: int, radius: int) -> Tuple[int, int]:
            x = int(round(position[0] * (self.output_size - 1)))
            y = int(round(position[1] * (self.output_size - 1)))
            semantic[
                max(0, y - radius) : min(self.output_size, y + radius + 1),
                max(0, x - radius) : min(self.output_size, x + radius + 1),
            ] = value
            return x, y

        bullet_centers = []
        for index, position in enumerate(normalized):
            if index != plane_index:
                visible_size = int(np.clip(round(np.sqrt(areas[index])), 1, 10))
                radius = max(1, int(np.ceil(visible_size / 2.0)))
                bullet_centers.append(
                    draw_mark(position, 160 + visible_size, radius)
                )
        for x, y in bullet_centers:
            semantic[y, x] = 255
        draw_mark(plane, 96, 2)
        return semantic


class LiveVisualController:
    """Load ``best.pt`` and choose an action from rendered pixels every 4 frames."""

    def __init__(
        self,
        checkpoint: str,
        device_name: str = "cuda",
        decision_interval: int = 4,
    ) -> None:
        device = torch.device(
            device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.agent, checkpoint_data = load_agent(str(Path(checkpoint)), device)
        config = checkpoint_data["config"]
        self.frame_stack = 4
        self.semanticizer = DominantBackgroundSemanticizer(output_size=96)
        self.frames: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        expected_interval = int(config.get("action_repeat", 4))
        self.decision_interval = max(1, int(decision_interval))
        if self.decision_interval != expected_interval:
            raise ValueError(
                "live decision interval does not match checkpoint action_repeat: "
                f"{self.decision_interval} != {expected_interval}"
            )
        self.frame_counter = 0
        self.action = 0

    def reset(self) -> None:
        self.semanticizer.reset()
        self.agent.reset(1)
        self.frames.clear()
        self.frame_counter = 0
        self.action = 0

    def prime_rgb(self, rgb: np.ndarray) -> int:
        """Initialize the four-frame stack and choose the first action at t=0."""
        frame = self.semanticizer.convert(rgb)
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(self.frame_stack))
        observation = np.stack(tuple(self.frames), axis=0)
        self.action = int(self.agent.act(observation[None], deterministic=True)[0])
        self.frame_counter = 0
        return self.action

    def observe_rgb(self, rgb: np.ndarray, physics_steps: int = 1) -> int:
        self.frame_counter += max(0, int(physics_steps))
        if self.frame_counter < self.decision_interval:
            return self.action
        self.frame_counter %= self.decision_interval
        frame = self.semanticizer.convert(rgb)
        self.frames.append(frame)
        if len(self.frames) < self.frame_stack:
            return self.action
        observation = np.stack(tuple(self.frames), axis=0)
        self.action = int(self.agent.act(observation[None], deterministic=True)[0])
        return self.action

    def observe_surface(self, surface: "object", physics_steps: int = 1) -> int:
        import pygame

        rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        return self.observe_rgb(rgb, physics_steps=physics_steps)

    def prime_surface(self, surface: "object") -> int:
        import pygame

        rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        return self.prime_rgb(rgb)
