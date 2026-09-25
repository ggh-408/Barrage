"""Time the production RGB controller on reproducible, moving image inputs.

This is a latency/parity benchmark, not a survival evaluation. The environment
uses a fixed action script; controller decisions never influence its trajectory.
Only rendered RGB pixels cross the controller input boundary. Hashing and
resource sampling happen outside the timed pipeline and are reported separately.
"""

from __future__ import annotations

import argparse
from collections import deque
import cProfile
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTION_SCRIPT = (0, 1, 5, 4, 8, 2, 6, 3, 7, 0)
SCRIPT_HOLD_DECISIONS = 7
FORMAT_VERSION = 1


def _digest(value: Any) -> str:
    """Hash exact types, shapes and bytes, including complete track histories."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            digest.update(b"array")
            visit(item.dtype.str)
            visit(tuple(item.shape))
            digest.update(np.ascontiguousarray(item).tobytes())
        elif is_dataclass(item) and not isinstance(item, type):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, dict):
            digest.update(b"dict" + str(len(item)).encode() + b":")
            for key in sorted(item, key=lambda entry: (type(entry).__name__, repr(entry))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple, deque)):
            digest.update(b"sequence" + str(len(item)).encode() + b":")
            for child in item:
                visit(child)
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, (bool, np.bool_)):
            digest.update(b"true" if item else b"false")
        elif isinstance(item, (int, np.integer)):
            digest.update(b"int" + str(int(item)).encode() + b";")
        elif isinstance(item, (float, np.floating)):
            digest.update(b"float" + struct.pack("!d", float(item)))
        elif isinstance(item, str):
            data = item.encode("utf-8")
            digest.update(b"str" + str(len(data)).encode() + b":" + data)
        elif hasattr(item, "detach"):
            visit(item.detach().cpu().numpy())
        else:
            raise TypeError(f"Unsupported parity value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Timing samples must be finite and non-negative")
    return {
        "count": int(len(array)),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "p99_ms": float(np.percentile(array, 99.0)),
        "max_ms": float(array.max()),
        "total_ms": float(array.sum()),
    }


def _resources() -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "process_cpu_seconds": time.process_time(),
    }
    try:
        import psutil

        memory = psutil.Process().memory_info()
        system = psutil.virtual_memory()
        result.update(
            source="psutil", process_rss_bytes=int(memory.rss),
            system_total_bytes=int(system.total),
            system_available_bytes=int(system.available),
        )
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            class ProcessMemory(ctypes.Structure):
                _fields_ = [
                    ("size", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                    ("peak_working_set", ctypes.c_size_t),
                    ("working_set", ctypes.c_size_t),
                    ("peak_paged_pool", ctypes.c_size_t),
                    ("paged_pool", ctypes.c_size_t),
                    ("peak_nonpaged_pool", ctypes.c_size_t),
                    ("nonpaged_pool", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t),
                    ("peak_pagefile", ctypes.c_size_t),
                    ("private_usage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD
            ]
            system = MemoryStatus()
            system.length = ctypes.sizeof(system)
            memory = ProcessMemory()
            memory.size = ctypes.sizeof(memory)
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(memory), memory.size
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            result.update(
                source="windows_ctypes", process_rss_bytes=int(memory.working_set),
                process_peak_rss_bytes=int(memory.peak_working_set),
                process_private_bytes=int(memory.private_usage),
                system_total_bytes=int(system.total_physical),
                system_available_bytes=int(system.available_physical),
            )
        else:
            result["source"] = "stdlib_cpu_only"
    return result


def _policy_state(agent: Any) -> dict[str, Any]:
    state = {
        key: value for key, value in vars(agent).items()
        if key.endswith("_count") and isinstance(value, (int, np.integer))
    }
    for key in ("_distilled_hidden", "_distilled_last_decision"):
        if hasattr(agent, key):
            state[key] = getattr(agent, key)
    return state


def _record(controller: Any, features: Any, rgb: np.ndarray, **metadata: Any) -> dict[str, Any]:
    tracker = controller.tracked_extractor.tracker
    tracking = {
        key: getattr(tracker, key) for key in (
            "tracks", "plane_position", "plane_velocity", "_previous_plane",
            "_next_track_id", "_step", "last_detection_count",
            "last_ambiguous_track_count",
        )
    }
    semantic = {
        key: getattr(controller.semanticizer, key) for key in (
            "previous_plane", "_recovery_stripe_index", "last_detection_mode",
            "last_predicted_match_count", "last_recovery_detection_count",
        )
    }
    return {
        **metadata,
        "controller_action": int(controller.action),
        "rgb_sha256": _digest(rgb),
        "track_state_sha256": _digest(tracking),
        "features_sha256": _digest(features),
        "semanticizer_state_sha256": _digest(semantic),
        "policy_state_sha256": _digest(_policy_state(controller.agent)),
        "tracked_count": len(tracker.tracks),
        "detection_mode": controller.semanticizer.last_detection_mode,
    }


def _run_configuration(args: argparse.Namespace, device: str, threads: int) -> dict[str, Any]:
    import pygame
    import torch
    import barrage_rl.env as environment_module
    from barrage_rl.live_screen import LiveVisualController
    import barrage_rl.runtime_core as runtime_core
    from barrage_rl.task_spec import TARGET_TASK, TARGET_TRACKING_CAPACITY

    # Source snapshots intentionally share immutable sprite assets with the project.
    environment_module.PROJECT_ROOT = PROJECT_ROOT
    render_world_surface = runtime_core.render_world_surface
    snapshot_rgb = getattr(
        runtime_core, "snapshot_surface_rgb",
        lambda current_surface: pygame.surfarray.array3d(current_surface).transpose(1, 0, 2),
    )
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; no silent CPU fallback")
    torch.manual_seed(args.seed)
    controller = LiveVisualController(str(args.checkpoint), device_name=device)
    if (
        controller.tracked_spec.max_objects != TARGET_TRACKING_CAPACITY
        or controller.tracked_spec.tracker_capacity != TARGET_TRACKING_CAPACITY
    ):
        raise ValueError("Checkpoint must preserve the current 384-slot model capacity")
    env = environment_module.BarrageVisionEnv(**TARGET_TASK.env_kwargs())
    surface = pygame.Surface((env.screen_width, env.screen_height))
    latest_features: list[Any] = [None]
    original_act = controller.agent.act_features

    def capture_features(objects: Any, masks: Any, globals_: Any, *positional: Any, **kwargs: Any) -> Any:
        # Retain references only; exact-byte hashing stays outside inference timing.
        latest_features[0] = (objects, masks, globals_)
        return original_act(objects, masks, globals_, *positional, **kwargs)

    controller.agent.act_features = capture_features
    timings: dict[str, list[float]] = {
        key: [] for key in (
            "environment_step", "render_world", "rgb_capture", "controller_total",
            "pipeline_total", "rgb_detection", "tracking_and_features", "model_forward",
        )
    }
    cold_timings: list[float] = []
    records_by_repeat: list[dict[str, Any]] = []
    reset_count = 0

    def render_rgb() -> np.ndarray:
        render_world_surface(
            surface, env.plane_surface, env.plane_position,
            env.bullet_surface, env.bullet_positions,
        )
        return snapshot_rgb(surface)

    def reset(seed: int) -> np.ndarray:
        nonlocal reset_count
        env.reset(seed=seed)
        controller.reset()
        rgb = render_rgb()
        controller.prime_rgb(rgb)
        controller.begin_measurement()
        reset_count += 1
        return rgb

    try:
        reset(args.seed + 900_000_000)
        warmup_episode = 0
        for step in range(args.warmup):
            action = ACTION_SCRIPT[(step // SCRIPT_HOLD_DECISIONS) % len(ACTION_SCRIPT)]
            _, _, terminated, truncated, _ = env.step(action)
            controller.observe_rgb(render_rgb(), physics_steps=TARGET_TASK.action_repeat)
            if terminated or truncated:
                warmup_episode += 1
                reset(args.seed + 900_000_000 + warmup_episode)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        reset_count = 0
        before = _resources()
        samples = [before]
        started = time.perf_counter()
        for repeat in range(args.repeats):
            episode = 0
            episode_seed = args.seed + repeat * 1_000_000
            cold_started = time.perf_counter()
            rgb = reset(episode_seed)
            cold_timings.append((time.perf_counter() - cold_started) * 1000.0)
            records = [_record(
                controller, latest_features[0], rgb, kind="prime", step=-1,
                episode_seed=episode_seed, scripted_action=None,
            )]
            for step in range(args.steps):
                scripted_action = ACTION_SCRIPT[(step // SCRIPT_HOLD_DECISIONS) % len(ACTION_SCRIPT)]
                pipeline_started = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(scripted_action)
                stepped = time.perf_counter()
                render_world_surface(
                    surface, env.plane_surface, env.plane_position,
                    env.bullet_surface, env.bullet_positions,
                )
                rendered = time.perf_counter()
                rgb = snapshot_rgb(surface)
                captured = time.perf_counter()
                controller.observe_rgb(rgb, physics_steps=TARGET_TASK.action_repeat)
                controlled = time.perf_counter()
                for key, value in (
                    ("environment_step", stepped - pipeline_started),
                    ("render_world", rendered - stepped),
                    ("rgb_capture", captured - rendered),
                    ("controller_total", controlled - captured),
                    ("pipeline_total", controlled - pipeline_started),
                ):
                    timings[key].append(value * 1000.0)
                for key, attribute in (
                    ("rgb_detection", "_stage_detection_ms"),
                    ("tracking_and_features", "_stage_tracking_ms"),
                    ("model_forward", "_stage_model_ms"),
                ):
                    timings[key].append(float(getattr(controller, attribute)[-1]))
                records.append(_record(
                    controller, latest_features[0], rgb, kind="step", step=step,
                    episode_seed=episode_seed, scripted_action=scripted_action,
                    terminated=bool(terminated), truncated=bool(truncated),
                ))
                if terminated or truncated:
                    episode += 1
                    episode_seed = args.seed + repeat * 1_000_000 + episode
                    cold_started = time.perf_counter()
                    rgb = reset(episode_seed)
                    cold_timings.append((time.perf_counter() - cold_started) * 1000.0)
                    records.append(_record(
                        controller, latest_features[0], rgb, kind="prime", step=step,
                        episode_seed=episode_seed, scripted_action=None,
                    ))
                if step % 30 == 0:
                    samples.append(_resources())
            records_by_repeat.append({"repeat": repeat, "records": records})
        after = _resources()
        samples.append(after)
        wall_seconds = time.perf_counter() - started
        cpu_seconds = float(after["process_cpu_seconds"]) - float(before["process_cpu_seconds"])
        pipeline_seconds = sum(timings["pipeline_total"]) / 1000.0
        resource_result = {
            "before": before, "after": after, "samples": len(samples),
            "harness_wall_seconds": wall_seconds,
            "harness_process_cpu_seconds": cpu_seconds,
            "harness_cpu_percent_of_one_logical_cpu": 100.0 * cpu_seconds / wall_seconds,
            "harness_cpu_percent_of_machine": 100.0 * cpu_seconds / wall_seconds / (os.cpu_count() or 1),
            "sampled_max_process_rss_bytes": max(int(sample.get("process_rss_bytes", 0)) for sample in samples),
            "sampled_min_system_available_bytes": min(int(sample.get("system_available_bytes", 0)) for sample in samples),
        }
        if device == "cuda":
            resource_result.update(
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
        return {
            "device": device, "torch_threads": threads,
            "agent_class": type(controller.agent).__name__,
            "safety_filter_mode": controller.agent.safety_filter_mode,
            "action_selector_mode": controller.agent.action_selector_mode,
            "tracked_spec": asdict(controller.tracked_spec),
            "action_delay_steps": controller.action_delay_steps,
            "measured_decisions": args.steps * args.repeats,
            "measured_pipeline_decisions_per_second": args.steps * args.repeats / pipeline_seconds,
            "controller_decision_over_budget_fraction": float(np.mean(
                np.asarray(timings["controller_total"]) > 1000.0 * controller.decision_interval / env.physics_fps
            )),
            "latency": {key: _statistics(values) for key, values in timings.items()},
            "reset_render_and_prime_latency": _statistics(cold_timings),
            "episode_resets": reset_count,
            "resources": resource_result,
            "repeats": records_by_repeat,
        }
    finally:
        controller.agent.act_features = original_act
        env.close()


def _check_baseline(result: dict[str, Any], baseline: dict[str, Any], reference_first: bool) -> dict[str, Any]:
    for field in ("format_version", "kind", "config", "checkpoint_sha256"):
        if result[field] != baseline.get(field):
            raise ValueError(f"Baseline mismatch in {field}")
    baseline_results = baseline.get("results", [])
    if not baseline_results:
        raise ValueError("Baseline contains no configurations")
    checked = 0
    for current in result["results"]:
        matches = [
            entry for entry in baseline_results
            if (entry["device"], entry["torch_threads"]) == (current["device"], current["torch_threads"])
        ]
        reference = baseline_results[0] if reference_first else (matches[0] if matches else None)
        if reference is None:
            raise ValueError(f"No baseline for {current['device']}, threads={current['torch_threads']}")
        for field in ("agent_class", "safety_filter_mode", "action_selector_mode", "tracked_spec", "action_delay_steps"):
            if current[field] != reference[field]:
                raise ValueError(f"Baseline mismatch in controller field {field}")
        for current_repeat, reference_repeat in zip(current["repeats"], reference["repeats"], strict=True):
            current_records = current_repeat["records"]
            reference_records = reference_repeat["records"]
            if len(current_records) != len(reference_records):
                raise ValueError("Baseline trajectory has a different record count")
            for index, (actual, expected) in enumerate(zip(current_records, reference_records, strict=True)):
                if actual != expected:
                    differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
                    raise ValueError(
                        f"Exact parity failed: device={current['device']} threads={current['torch_threads']} "
                        f"repeat={current_repeat['repeat']} record={index} fields={differing}"
                    )
                checked += 1
    return {"passed": True, "records_checked": checked, "reference_first": reference_first}


def _atomic_publish(temporary: Path, destination: Path) -> None:
    """An exclusive hard link publishes complete bytes without replacing a file."""
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(destination: Path, result: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            json.dump(result, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            path.unlink(missing_ok=True)
            raise
    _atomic_publish(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", "--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--torch-threads", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", "--iterations", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2_200_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--parity-reference-first", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.warmup < 0 or min(args.torch_threads) < 1:
        parser.error("steps, repeats and threads must be positive; warmup must be non-negative")
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.source_root = args.source_root.resolve(strict=True)
    if not (args.source_root / "barrage_rl" / "__init__.py").is_file():
        parser.error("source-root must contain the barrage_rl package")
    for destination in (args.output, args.profile):
        if destination is not None and destination.exists():
            parser.error(f"Refusing to overwrite existing artifact: {destination}")
    if args.profile is not None and args.profile.resolve() == args.output.resolve():
        parser.error("profile and output must have distinct paths")
    sys.path.insert(0, str(args.source_root))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    import torch
    from barrage_rl.task_spec import TARGET_TASK

    baseline = None
    if args.baseline_json is not None:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": "core_latency_and_exact_parity",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_digest(args.checkpoint),
        "source_root": str(args.source_root),
        "config": {
            "task": TARGET_TASK.manifest(), "steps": args.steps,
            "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
            "script": list(ACTION_SCRIPT), "script_hold_decisions": SCRIPT_HOLD_DECISIONS,
            "controller_configuration": "LiveVisualController production defaults",
            "input_schedule": "one current RGB frame per action_repeat physics steps",
        },
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "pygame": pygame.version.ver, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "resources": _resources(),
        },
        "measurement_notes": [
            "Latency excludes warmup, reset/prime, hashing, resource sampling and JSON output.",
            "environment_step includes canonical simulation, rewards and semantic observation creation.",
            "model_forward is the existing controller stage, including feature transfer and action selection.",
            "Agent action export synchronizes CUDA; stage timing includes that synchronization.",
            "Resource utilization and harness wall time include parity hashing and reset/prime work.",
            "Trajectories are scripted timing inputs, with no survival/acceptance evaluation claims.",
            f"The environment uses {TARGET_TASK.bullet_count} bullets; checkpoint feature normalization metadata is preserved unchanged.",
        ],
        "results": [],
    }
    profiler = cProfile.Profile() if args.profile else None
    pygame.init()
    pygame.display.set_mode((1, 1))
    try:
        if profiler:
            profiler.enable()
        for device in dict.fromkeys(args.device):
            for threads in dict.fromkeys(args.torch_threads):
                current = _run_configuration(args, device, threads)
                result["results"].append(current)
                print(json.dumps({
                    "device": device, "torch_threads": threads,
                    "latency": current["latency"],
                    "resources": current["resources"],
                }), flush=True)
    finally:
        if profiler:
            profiler.disable()
        pygame.quit()
    if baseline is not None:
        result["baseline_json"] = str(args.baseline_json.resolve())
        try:
            result["baseline_parity"] = _check_baseline(result, baseline, args.parity_reference_first)
        except ValueError as error:
            # Preserve failed candidates for audit, while keeping a failing exit
            # status so numerical/state differences cannot be selected silently.
            result["baseline_parity"] = {"passed": False, "error": str(error)}
            _write_json(args.output, result)
            raise
    if profiler and args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.profile.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            profiler.dump_stats(str(temporary_path))
            _atomic_publish(temporary_path, args.profile)
        finally:
            temporary_path.unlink(missing_ok=True)
        result["profile"] = str(args.profile.resolve())
    _write_json(args.output, result)
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
