"""Check exact closed-loop Barrage window behavior on a dummy display.

The real Barrage physics, renderer and current-RGB controller run at fixed
decision boundaries. Damage immunity matches the visible latency diagnostic.
This is a reproducibility/latency diagnostic, with no survival acceptance claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1


def _check_baseline(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    for field in ("format_version", "kind", "config", "checkpoint_sha256", "controller"):
        if result[field] != baseline.get(field):
            raise ValueError(f"Baseline mismatch in {field}")
    actual_records = result["records"]
    expected_records = baseline.get("records", [])
    if len(actual_records) != len(expected_records):
        raise ValueError("Baseline has a different number of closed-loop records")
    for index, (actual, expected) in enumerate(zip(actual_records, expected_records, strict=True)):
        if actual != expected:
            fields = sorted(
                key for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                f"Exact closed-loop parity failed at record {index}, "
                f"physics step {actual.get('physics_step')}: {fields}"
            )
    return {"passed": True, "records_checked": len(actual_records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--physics-steps", type=int, help="Defaults to 120 seconds at the selected source's physics rate")
    parser.add_argument("--seed", type=int, default=2_109_170_720)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    args = parser.parse_args()
    if args.physics_steps is not None and (args.physics_steps < 4 or args.physics_steps % 4):
        parser.error("physics-steps must be a positive multiple of four")
    if args.torch_threads is not None and args.torch_threads < 1:
        parser.error("torch-threads must be positive when supplied")
    args.source_root = args.source_root.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing artifact: {args.output}")
    if not (args.source_root / "Barrage.py").is_file():
        parser.error("source-root must contain Barrage.py")
    if not (args.source_root / "barrage_rl" / "__init__.py").is_file():
        parser.error("source-root must contain the barrage_rl package")

    # Set the display driver before Pygame imports/initialization. The helper
    # never creates or focuses a desktop window, even if SDL was configured.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path[:0] = [str(args.source_root), str(PROJECT_ROOT)]

    import pygame
    import torch
    import Barrage as game_module
    import barrage_rl.runtime_core as runtime_core
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.task_spec import TARGET_TASK, TARGET_TRACKING_CAPACITY
    from tools.benchmark_core_latency import (
        _digest, _file_digest, _record, _resources, _statistics, _write_json,
    )

    # Fail explicitly if import caching or search paths defeat source isolation.
    if Path(game_module.__file__).resolve() != args.source_root / "Barrage.py":
        raise RuntimeError("Barrage.py was imported from the wrong source root")
    if Path(runtime_core.__file__).resolve() != args.source_root / "barrage_rl" / "runtime_core.py":
        raise RuntimeError("runtime_core was imported from the wrong source root")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; no CPU fallback")
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)

    baseline = (
        json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if args.baseline_json is not None else None
    )
    game = game_module.Barrage
    if args.physics_steps is None:
        args.physics_steps = 120 * game.PHYSICS_FPS
    # Frozen sources reuse the unchanged production image assets.
    game_module.PROJECT_ROOT = PROJECT_ROOT
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    game.window = pygame.Surface((820, 820))
    game.SCREEN_WIDTH = game.SCREEN_HEIGHT = 820
    game.QUANTITY = 300
    game.BULLET_SIZE = 5
    game.PLANE_SPEED = game.BULLET_SPEED = 240.0
    game.TARGETED_BULLET_PROBABILITY = 0.10
    game.TARGETED_PREDICTION_SCALE_MIN = 0.65
    game.TARGETED_PREDICTION_SCALE_MAX = 1.0
    game.TARGETED_ANGULAR_NOISE = 0.08
    game.COLLISION = False
    game.INVINCIBLE = False  # The legacy true value enables collision damage.
    game.KEY = True
    game.MUSIC = False
    game.RNG = np.random.default_rng(args.seed)
    game.AI_PIPELINE = None
    game_module.Plane.SKIN = 0
    controller = LiveVisualController(str(args.checkpoint), device_name=args.device)
    game.AI_CONTROLLER = controller
    if game.PHYSICS_FPS <= 0 or controller.decision_interval != 4:
        raise ValueError("This diagnostic requires the fixed physics / four-step schedule")
    if controller.action_delay_steps != 0:
        raise ValueError("This diagnostic requires the checkpoint's synchronous zero-delay policy")
    if (controller.tracked_spec.max_objects, controller.tracked_spec.tracker_capacity) != (
        TARGET_TRACKING_CAPACITY, TARGET_TRACKING_CAPACITY,
    ):
        raise ValueError("The checkpoint must preserve the production 384-slot capacity")

    latest_features: list[Any] = [None]
    latest_rgb: list[Any] = [None]
    original_act = controller.agent.act_features
    original_prime = controller.prime_rgb
    original_observe = controller._observe_due_rgb

    def capture_features(objects: Any, masks: Any, globals_: Any, *positional: Any, **kwargs: Any) -> Any:
        latest_features[0] = (objects, masks, globals_)
        return original_act(objects, masks, globals_, *positional, **kwargs)

    def capture_prime(rgb: np.ndarray) -> int:
        latest_rgb[0] = rgb
        return original_prime(rgb)

    def capture_observe(rgb: np.ndarray, **kwargs: Any) -> int:
        latest_rgb[0] = rgb
        return original_observe(rgb, **kwargs)

    controller.agent.act_features = capture_features
    controller.prime_rgb = capture_prime
    controller._observe_due_rgb = capture_observe

    def record(boundary: int, applied_action: int | None) -> dict[str, Any]:
        # Simulator state is read strictly after inference for diagnostic hashes.
        # Only the captured RGB array enters the policy and its decision logic.
        return _record(
            controller, latest_features[0], latest_rgb[0],
            kind="prime" if boundary == 0 else "decision",
            boundary=boundary, physics_step=boundary * controller.decision_interval,
            applied_action=applied_action, next_action=int(game.AI_ACTION),
            alive=bool(game.KEY), alive_physics_steps=int(game.ALIVE_PHYSICS_STEPS),
            integer_score=int(game.ALIVE_PHYSICS_STEPS * 10 // game.PHYSICS_FPS),
            game_time_seconds=float(game.ALIVE_PHYSICS_STEPS / game.PHYSICS_FPS),
            plane_state_sha256=_digest({
                "position": tuple(game.PLANE.position),
                "velocity": tuple(game.PLANE.velocity),
                "rect": tuple(game.PLANE.rect), "skin": game_module.Plane.SKIN,
            }),
            bullet_state_sha256=_digest(game_module.Bullet.LIST),
            rng_state_sha256=_digest(game.RNG.bit_generator.state),
        )

    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": "barrage_window_closed_loop_exact_parity",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_digest(args.checkpoint),
        "source_root": str(args.source_root),
        "source_sha256": {
            str(path.relative_to(args.source_root)): _file_digest(path)
            for path in [args.source_root / "Barrage.py", *sorted((args.source_root / "barrage_rl").glob("*.py"))]
        },
        "config": {
            "task": TARGET_TASK.manifest(), "seed": args.seed,
            "physics_steps": args.physics_steps, "physics_fps": game.PHYSICS_FPS,
            "decision_interval": controller.decision_interval,
            "bullets": game.QUANTITY, "targeted_bullet_probability": 0.10,
            "damage_immunity": True, "wall_collision": False,
            "screen_size": [820, 820], "bullet_size": game.BULLET_SIZE,
            "plane_speed": game.PLANE_SPEED, "bullet_speed": game.BULLET_SPEED,
            "device": args.device, "torch_threads": torch.get_num_threads(),
            "input_schedule": "Barrage current rendered RGB every four physics steps",
            "action_schedule": "prime action at step zero; each action used until the next boundary",
        },
        "controller": {
            "agent_class": type(controller.agent).__name__,
            "safety_filter_mode": controller.agent.safety_filter_mode,
            "action_selector_mode": controller.agent.action_selector_mode,
            "tracked_spec": asdict(controller.tracked_spec),
            "action_delay_steps": controller.action_delay_steps,
        },
        "render_fps_cap_in_source": int(game.FPS),
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "pygame": pygame.version.ver,
        },
        "measurement_notes": [
            "This is closed-loop exact replay, separate from fixed-200 survival acceptance evaluation.",
            "Barrage.advance_physics, Barrage.render_world and observe_due_surface are used unchanged.",
            "Only current rendered RGB enters the controller; world state is hashed after inference.",
            "Dummy rendering avoids a desktop window. No realtime FPS or display-latency claim is made.",
            "Timing excludes reset/prime, exact-state hashing, resource sampling and JSON output.",
            "Controller timing includes the real surface-to-RGB copy and synchronous action export.",
        ],
    }
    timings: dict[str, list[float]] = {
        name: [] for name in ("physics_four_steps", "render_world", "controller_with_capture", "boundary_total")
    }
    records: list[dict[str, Any]] = []
    before = _resources()
    resource_samples = [before]
    started = time.perf_counter()
    try:
        reset_started = time.perf_counter()
        game.reset_game()
        result["reset_render_prime_ms"] = (time.perf_counter() - reset_started) * 1000.0
        records.append(record(0, None))
        controller.begin_measurement()
        for boundary in range(1, args.physics_steps // controller.decision_interval + 1):
            applied_action = int(game.AI_ACTION)
            boundary_started = time.perf_counter()
            for _ in range(controller.decision_interval):
                game.advance_physics(
                    runtime_core.ACTION_VECTORS[game.AI_ACTION], 1.0 / game.PHYSICS_FPS,
                )
            stepped = time.perf_counter()
            game.render_world()
            rendered = time.perf_counter()
            game.AI_ACTION = controller.observe_due_surface(game.window)
            controlled = time.perf_counter()
            for name, seconds in (
                ("physics_four_steps", stepped - boundary_started),
                ("render_world", rendered - stepped),
                ("controller_with_capture", controlled - rendered),
                ("boundary_total", controlled - boundary_started),
            ):
                timings[name].append(seconds * 1000.0)
            records.append(record(boundary, applied_action))
            if boundary % 300 == 0:
                resource_samples.append(_resources())
                print(json.dumps({
                    "closed_loop_boundaries": boundary,
                    "physics_steps": boundary * controller.decision_interval,
                    "game_seconds": game.ALIVE_PHYSICS_STEPS / game.PHYSICS_FPS,
                }), flush=True)
        after = _resources()
        resource_samples.append(after)
        wall_seconds = time.perf_counter() - started
        result["records"] = records
        result["trajectory_sha256"] = _digest(records)
        result["latency"] = {name: _statistics(values) for name, values in timings.items()}
        result["controller_stages"] = controller.runtime_stage_report()
        result["resources"] = {
            "before": before, "after": after,
            "harness_wall_seconds": wall_seconds,
            "harness_cpu_seconds": float(after["process_cpu_seconds"]) - float(before["process_cpu_seconds"]),
            "sampled_max_process_rss_bytes": max(int(item.get("process_rss_bytes", 0)) for item in resource_samples),
            "sampled_min_system_available_bytes": min(int(item.get("system_available_bytes", 0)) for item in resource_samples),
        }
        if baseline is not None:
            result["baseline_json"] = str(args.baseline_json.resolve())
            try:
                result["baseline_parity"] = _check_baseline(result, baseline)
            except ValueError as error:
                result["baseline_parity"] = {"passed": False, "error": str(error)}
                _write_json(args.output, result)
                raise
        _write_json(args.output, result)
        print(json.dumps({
            "report": str(args.output), "records": len(records),
            "trajectory_sha256": result["trajectory_sha256"],
            "baseline_parity": result.get("baseline_parity"), "latency": result["latency"],
        }), flush=True)
    finally:
        controller.agent.act_features = original_act
        controller.prime_rgb = original_prime
        controller._observe_due_rgb = original_observe
        pygame.quit()


if __name__ == "__main__":
    main()
