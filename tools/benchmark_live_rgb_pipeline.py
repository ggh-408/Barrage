"""Benchmark RGB detection equivalence and live-controller latency.

The equivalence section reads simulator positions only to render a controlled
offline reference image.  The semanticizer and controller under measurement
receive only the resulting RGB pixels, matching the deployment boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from barrage_rl.timing import DECISION_DT, PHYSICS_FPS
from barrage_rl.task_spec import TARGET_TASK


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "p99_ms": float(np.percentile(array, 99.0)),
        "max_ms": float(array.max()),
    }


def benchmark(
    checkpoint: Path,
    seeds: int,
    iterations: int,
    bullet_count: int,
    device: str,
    analytic_shield: bool = False,
    analytic_shield_gate: str = "always",
    torch_threads: int | None = None,
) -> dict[str, object]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    import torch

    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))

    from barrage_rl.env import BarrageVisionEnv
    from barrage_rl.live_screen import (
        DominantBackgroundSemanticizer,
        LiveVisualController,
    )

    pygame.init()
    pygame.display.set_mode((1, 1))
    surface = pygame.Surface((820, 820))
    bullet = pygame.image.load(
        str(PROJECT_ROOT / "image" / "bullet(5).gif")
    ).convert_alpha()
    plane = pygame.image.load(
        str(PROJECT_ROOT / "image" / "plane(0).gif")
    ).convert_alpha()
    semanticizer = DominantBackgroundSemanticizer(output_size=384, bullet_size=5)
    live_counts: list[int] = []
    ideal_counts: list[int] = []
    semantic_times: list[float] = []
    for seed in range(2_200_000, 2_200_000 + seeds):
        env = BarrageVisionEnv(
            bullet_count=bullet_count,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            observation_size=384,
            randomize_initial_phase=False,
        )
        try:
            observation, _ = env.reset(seed=seed)
            surface.fill((0, 0, 0))
            surface.blit(plane, plane.get_rect(center=tuple(env.plane_position)))
            for position in env.bullet_positions:
                surface.blit(bullet, bullet.get_rect(center=tuple(position)))
            rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
            started = time.perf_counter()
            detected = semanticizer.detect(rgb, include_semantic=False)
            semantic_times.append((time.perf_counter() - started) * 1000.0)
            live_counts.append(int(len(detected.bullet_positions)))
            ideal_counts.append(int(len(env.bullet_positions)))
        finally:
            env.close()

    controller = LiveVisualController(
        str(checkpoint),
        device_name=device,
        analytic_shield=analytic_shield,
        analytic_shield_gate=analytic_shield_gate,
    )
    controller.prime_surface(surface)
    warmup = 20
    for _ in range(warmup):
        controller.observe_surface(surface, physics_steps=1)
    controller.begin_measurement()
    elapsed: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        controller.observe_surface(surface, physics_steps=1)
        elapsed.append((time.perf_counter() - started) * 1000.0)
    steady = elapsed
    decision = steady[3::4]
    light = [value for index, value in enumerate(steady) if index % 4 != 3]
    decision_budget_ms = 1000.0 * DECISION_DT
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torch_threads": int(torch.get_num_threads()),
        "equivalence_seeds": seeds,
        "bullets": bullet_count,
        "safety_filter_mode": controller.agent.safety_filter_mode,
        "action_selector_mode": controller.agent.action_selector_mode,
        "ideal_center_mean": float(np.mean(ideal_counts)),
        "rgb_center_mean": float(np.mean(live_counts)),
        "rgb_minus_ideal_center_mean": float(
            np.mean(np.asarray(live_counts) - np.asarray(ideal_counts))
        ),
        "rgb_center_min": int(min(live_counts)),
        "rgb_center_max": int(max(live_counts)),
        "semanticizer": _percentiles(semantic_times[5:]),
        "controller_all_steps": _percentiles(steady),
        "controller_decision_steps": _percentiles(decision),
        "controller_decision_over_budget_fraction": float(
            np.mean(np.asarray(decision) > decision_budget_ms)
        ),
        "controller_nondecision_steps": _percentiles(light),
        "controller_runtime_stages": controller.runtime_stage_report(),
        "physics_budget_ms": 1000.0 / PHYSICS_FPS,
        "decision_budget_ms": decision_budget_ms,
    }
    pygame.quit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--analytic-shield", action="store_true")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.checkpoint,
        args.seeds,
        args.iterations,
        args.bullets,
        args.device,
        analytic_shield=args.analytic_shield,

        torch_threads=args.torch_threads,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
