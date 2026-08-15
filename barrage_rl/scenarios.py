"""Deterministic scenario curriculum.

Each block contains 60% core deployment cases, 25% broad regularization, and
15% explicitly difficult combinations.  Reset modes are crossed with the same
block so deployment/mid-episode/hard starts remain 70/20/10 rather than being
left to chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    source: str
    bullet_size: int
    bullet_speed: float
    targeted_probability: float
    reset_mode: str


class ScenarioSampler:
    """Yield shuffled, exactly stratified 100-episode curriculum blocks."""

    SOURCE_RESET_COUNTS = {
        "core": (42, 12, 6),
        "broad": (18, 5, 2),
        "stress": (10, 3, 2),
    }
    RESET_MODES = ("deployment", "mid_episode", "recoverable_hard")
    BROAD_SPEEDS = (60.0, 100.0, 140.0, 180.0, 220.0, 260.0, 300.0)
    STRESS = (
        (7, 300.0), (7, 270.0), (6, 300.0), (5, 300.0), (7, 240.0),
    )

    def __init__(
        self,
        seed: int,
        core_size: int = 5,
        core_speed: float = 240.0,
        targeted_probability: float = 0.35,
        stress_targeted_probability: float = 0.50,
    ) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.core_size = int(core_size)
        self.core_speed = float(core_speed)
        self.targeted_probability = float(targeted_probability)
        self.stress_targeted_probability = float(stress_targeted_probability)
        self.block_index = 0
        self.queue: List[ScenarioSpec] = []
        self.broad_index = 0
        self.stress_index = 0

    def _broad_parameters(self) -> Tuple[int, float]:
        # A coprime stride spreads the 7x7 grid before any pair repeats.
        index = self.broad_index % 49
        size = index % 7 + 1
        speed = self.BROAD_SPEEDS[(index * 3 + index // 7) % 7]
        self.broad_index += 1
        return size, speed

    def _parameters(self, source: str) -> Tuple[int, float, float]:
        if source == "core":
            return self.core_size, self.core_speed, self.targeted_probability
        if source == "broad":
            size, speed = self._broad_parameters()
            return size, speed, self.targeted_probability
        size, speed = self.STRESS[self.stress_index % len(self.STRESS)]
        self.stress_index += 1
        return size, speed, self.stress_targeted_probability

    def _refill(self) -> None:
        block: List[ScenarioSpec] = []
        local_index = 0
        for source, counts in self.SOURCE_RESET_COUNTS.items():
            for reset_mode, count in zip(self.RESET_MODES, counts):
                for _ in range(count):
                    size, speed, probability = self._parameters(source)
                    block.append(
                        ScenarioSpec(
                            scenario_id=f"b{self.block_index:05d}-{local_index:03d}",
                            source=source,
                            bullet_size=size,
                            bullet_speed=speed,
                            targeted_probability=probability,
                            reset_mode=reset_mode,
                        )
                    )
                    local_index += 1
        if len(block) != 100:
            raise AssertionError("scenario curriculum block must contain 100 episodes")
        order = self.rng.permutation(len(block))
        self.queue = [block[int(index)] for index in order]
        self.block_index += 1

    def next(self) -> ScenarioSpec:
        if not self.queue:
            self._refill()
        return self.queue.pop()

    def sample_block(self) -> Tuple[ScenarioSpec, ...]:
        """Testing/reporting helper that consumes exactly one block."""
        return tuple(self.next() for _ in range(100))
