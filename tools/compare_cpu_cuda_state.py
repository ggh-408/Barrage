"""Compare CPU/CUDA recurrent state on one CPU-driven, current-RGB trajectory.

This diagnostic uses Barrage's existing world update, render and live controller.
It never supplies world state to either policy. CUDA is a shadow: its decisions
are recorded but never advance the world. Instrumentation observes the original
forward passes and selector return without invoking a second policy decision.
This is neither a latency benchmark nor a survival evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the project's exact numeric/type hashing rather than hashing tensor
# repr/device strings, which would create a metadata-only first difference.
from benchmark_core_latency import _digest, _file_digest


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "NaN"
        if np.isposinf(number):
            return "+Infinity"
        if np.isneginf(number):
            return "-Infinity"
        return number
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(_json_value(value), output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _array_difference(cpu: np.ndarray, cuda: np.ndarray) -> dict[str, Any]:
    cpu, cuda = np.asarray(cpu), np.asarray(cuda)
    same_shape = cpu.shape == cuda.shape
    result: dict[str, Any] = {
        "cpu_shape": list(cpu.shape), "cuda_shape": list(cuda.shape),
        "cpu_dtype": str(cpu.dtype), "cuda_dtype": str(cuda.dtype),
        "same_shape": same_shape,
        "exact_bytes": bool(
            same_shape and cpu.dtype == cuda.dtype and cpu.tobytes() == cuda.tobytes()
        ),
        "cpu_finite": bool(np.isfinite(cpu).all()),
        "cuda_finite": bool(np.isfinite(cuda).all()),
        "cpu_nonfinite_count": int(np.count_nonzero(~np.isfinite(cpu))),
        "cuda_nonfinite_count": int(np.count_nonzero(~np.isfinite(cuda))),
    }
    if not same_shape:
        result.update(mae=None, maxabs=None, rmse=None, relative_norm=None)
        return result
    left, right = cpu.astype(np.float64), cuda.astype(np.float64)
    common_finite = np.isfinite(left) & np.isfinite(right)
    result["paired_finite_count"] = int(common_finite.sum())
    result["nonfinite_pattern_equal"] = bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    if common_finite.any():
        delta = right[common_finite] - left[common_finite]
        absolute = np.abs(delta)
        reference_norm = float(np.linalg.norm(left[common_finite]))
        result.update(
            mae=float(absolute.mean()), maxabs=float(absolute.max()),
            rmse=float(np.sqrt(np.mean(delta * delta))),
            relative_norm=float(np.linalg.norm(delta) / max(reference_norm, 1e-30)),
            cpu_reference_norm=reference_norm,
        )
    else:
        empty = cpu.size == 0
        result.update(
            mae=0.0 if empty else None, maxabs=0.0 if empty else None,
            rmse=0.0 if empty else None, relative_norm=0.0 if empty else None,
        )
    if not result["exact_bytes"] and same_shape and cpu.dtype == cuda.dtype and cpu.size:
        left_bytes = np.ascontiguousarray(cpu).view(np.uint8).reshape(-1, cpu.itemsize)
        right_bytes = np.ascontiguousarray(cuda).view(np.uint8).reshape(-1, cuda.itemsize)
        differing = np.flatnonzero(np.any(left_bytes != right_bytes, axis=1))
        if len(differing):
            index = int(differing[0])
            result["first_differing_index"] = list(np.unravel_index(index, cpu.shape))
            result["first_cpu_value"] = cpu.reshape(-1)[index].item()
            result["first_cuda_value"] = cuda.reshape(-1)[index].item()
    return result


class _ForwardObserver:
    """Read-only hooks and a pass-through selector observer; no extra forward."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self.values: dict[str, Any] = {}
        self.handles: list[Any] = []
        agent = controller.agent

        def capture_output(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                self.values[name] = output
            return hook

        def capture_input(name: str):
            def hook(_module: Any, inputs: Any) -> None:
                self.values[name] = inputs[0]
            return hook

        for name in ("policy_head", "teacher_cost_head", "collision_head"):
            self.handles.append(getattr(agent.model, name).register_forward_hook(
                capture_output("backbone." + name)
            ))
        for name, module in (
            ("input.objects", agent.model.object_encoder),
            ("input.globals", agent.model.global_encoder),
        ):
            self.handles.append(module.register_forward_pre_hook(capture_input(name)))
        self.handles.append(agent.distilled.register_forward_hook(capture_output("student")))
        self.selector = agent._action_selector
        self.original_select = self.selector.select

        def observed_select(*args: Any, **kwargs: Any) -> Any:
            selection = self.original_select(*args, **kwargs)
            self.values["selection"] = selection
            return selection

        self.selector.select = observed_select

    def clear(self) -> None:
        self.values.clear()

    def snapshot(self) -> dict[str, np.ndarray]:
        import torch

        required = {
            "backbone.policy_head", "backbone.teacher_cost_head", "backbone.collision_head",
            "input.objects", "input.globals", "student", "selection",
        }
        missing = required - self.values.keys()
        if missing:
            raise RuntimeError(f"Forward observation missing fields: {sorted(missing)}")
        result = {
            name: _array(value) for name, value in self.values.items()
            if name not in ("student", "selection")
        }
        prediction, selection = self.values["student"], self.values["selection"]
        for field in fields(prediction):
            result["student." + field.name] = _array(getattr(prediction, field.name))
        for field in fields(selection):
            value = getattr(selection, field.name)
            if value is not None:
                result["selection." + field.name] = _array(value)
        # Post-decision diagnostics use the same backend, dtype and comparison
        # as production. They neither alter state nor replace any decision.
        with torch.inference_mode():
            agent = self.controller.agent
            result["threshold.immediate_risk_margin"] = _array(
                selection.immediate_risk.double() - agent.safety_threshold
            )
            result["threshold.learned_unsafe"] = _array(
                selection.immediate_risk >= agent.safety_threshold
            )
        return result

    def close(self) -> None:
        self.selector.select = self.original_select
        for handle in self.handles:
            handle.remove()


def _state(controller: Any) -> dict[str, Any]:
    agent = controller.agent
    tracker = controller.tracked_extractor.tracker
    # Only the latest eight observations enter the regression. Hash this causal
    # tracker state every step; avoid quadratically hashing unused old history.
    tracks = []
    for track in tracker.tracks:
        item = dict(vars(track))
        item["history"] = track.history[-8:]
        item["full_history_length"] = len(track.history)
        tracks.append(item)
    tracking = {"tracks": tracks}
    for key in (
        "plane_position", "plane_velocity", "_previous_plane", "_next_track_id",
        "_step", "last_detection_count", "last_ambiguous_track_count",
    ):
        tracking[key] = getattr(tracker, key)
    semantic = {
        key: getattr(controller.semanticizer, key) for key in (
            "previous_plane", "_recovery_stripe_index", "last_detection_mode",
            "last_predicted_match_count", "last_recovery_detection_count",
        )
    }
    return {
        "hidden": {str(key): _array(value) for key, value in agent._distilled_hidden.items()},
        "counters": {
            key: int(value) for key, value in vars(agent).items()
            if key.endswith("_count") and isinstance(value, (int, np.integer))
        },
        "last_decision": dict(agent._distilled_last_decision),
        "tracker_causal_sha256": _digest(tracking),
        "semanticizer_sha256": _digest(semantic),
        "features_sha256": _digest(controller.tracked_extractor._features()),
    }


def _state_comparison(cpu: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    cpu_keys, cuda_keys = sorted(cpu["hidden"]), sorted(cuda["hidden"])
    common = sorted(set(cpu_keys) & set(cuda_keys))
    hidden = {
        key: _array_difference(cpu["hidden"][key], cuda["hidden"][key]) for key in common
    }
    combined = _array_difference(
        np.concatenate([cpu["hidden"][key].reshape(-1) for key in common]) if common else np.empty(0),
        np.concatenate([cuda["hidden"][key].reshape(-1) for key in common]) if common else np.empty(0),
    )
    result: dict[str, Any] = {
        "hidden_cpu_keys": cpu_keys, "hidden_cuda_keys": cuda_keys,
        "hidden_keys_equal": cpu_keys == cuda_keys,
        "hidden_by_key": hidden, "hidden_combined": combined,
        "hidden_cpu_sha256": _digest(cpu["hidden"]),
        "hidden_cuda_sha256": _digest(cuda["hidden"]),
        "first_state_difference": None,
    }
    if cpu_keys != cuda_keys:
        result["first_state_difference"] = {"field": "_distilled_hidden.keys", "cpu": cpu_keys, "cuda": cuda_keys}
    else:
        for key, metrics in hidden.items():
            if not metrics["exact_bytes"]:
                result["first_state_difference"] = {
                    "field": f"_distilled_hidden[{key}]", "numeric_difference": metrics,
                }
                break
    for name in ("counters", "last_decision"):
        same = cpu[name] == cuda[name]
        result[name] = {"equal": same, "cpu": cpu[name], "cuda": cuda[name]}
        if not same and result["first_state_difference"] is None:
            for key in sorted(set(cpu[name]) | set(cuda[name]), key=str):
                if cpu[name].get(key) != cuda[name].get(key):
                    result["first_state_difference"] = {
                        "field": f"{name}.{key}", "cpu": cpu[name].get(key), "cuda": cuda[name].get(key),
                    }
                    break
    for name in ("tracker_causal_sha256", "semanticizer_sha256", "features_sha256"):
        result[name] = {"equal": cpu[name] == cuda[name], "cpu": cpu[name], "cuda": cuda[name]}
    return result


def _metric_summary(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    valid = [
        (record["decision"], record["state"]["hidden_combined"][metric])
        for record in records
        if record["state"]["hidden_combined"].get(metric) is not None
    ]
    if not valid:
        return {"count": 0}
    x, y = np.asarray(valid, np.float64).T
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    slope = float(np.dot(centered, y - y.mean()) / denominator) if denominator else 0.0
    return {
        "count": len(valid), "first": float(y[0]), "last": float(y[-1]),
        "mean": float(y.mean()), "median": float(np.median(y)), "max": float(y.max()),
        "slope_per_1000_decisions": slope * 1000.0,
        "linear_fit_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
    }


def _summary(records: list[dict[str, Any]], interval_size: int) -> dict[str, Any]:
    steps = [record for record in records if record["kind"] == "step"]
    first_numeric, first_state, first_action = None, None, None
    for record in records:
        location = {"kind": record["kind"], "decision": record["decision"], "game_seconds": record["game_seconds"]}
        if first_numeric is None and record["first_numeric_difference"] is not None:
            first_numeric = {**location, **record["first_numeric_difference"]}
        if first_state is None and record["state"]["first_state_difference"] is not None:
            first_state = {**location, **record["state"]["first_state_difference"]}
        if first_action is None and not record["actions_equal"]:
            first_action = {**location, "cpu_action": record["cpu_action"], "cuda_action": record["cuda_action"]}
    return {
        "records_including_prime": len(records), "step_decisions": len(steps),
        "action_disagreement_count_including_prime": sum(not record["actions_equal"] for record in records),
        "first_numeric_difference": first_numeric, "first_policy_state_difference": first_state,
        "first_action_disagreement": first_action,
        "all_hidden_values_finite": all(
            record["state"]["hidden_combined"]["cpu_finite"]
            and record["state"]["hidden_combined"]["cuda_finite"] for record in records
        ),
        "all_hidden_keys_equal": all(record["state"]["hidden_keys_equal"] for record in records),
        "all_counters_equal": all(record["state"]["counters"]["equal"] for record in records),
        "all_input_features_equal": all(record["state"]["features_sha256"]["equal"] for record in records),
        "all_tracker_causal_states_equal": all(record["state"]["tracker_causal_sha256"]["equal"] for record in records),
        "all_semanticizer_states_equal": all(record["state"]["semanticizer_sha256"]["equal"] for record in records),
        "hidden_metrics": {metric: _metric_summary(steps, metric) for metric in ("mae", "maxabs", "rmse", "relative_norm")},
        "intervals": [{
            "first_decision": group[0]["decision"], "last_decision": group[-1]["decision"],
            "action_disagreement_count": sum(not record["actions_equal"] for record in group),
            "hidden": {metric: _metric_summary(group, metric) for metric in ("mae", "maxabs", "rmse", "relative_norm")},
        } for start in range(0, len(steps), interval_size) if (group := steps[start:start + interval_size])],
        "trend_interpretation": (
            "Descriptive linear trends of observed hidden-state differences, not a significance test. "
            "No sum of per-step errors is presented as accumulated state error. Shared RGB prevents "
            "closed-loop world divergence; a discrete recovery/fallback divergence can still affect "
            "later action selection and is reported separately from hidden numeric differences."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs/visual_set_v42/best.pt")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--decisions", type=int, help="Defaults to 120 seconds at the current decision rate")
    parser.add_argument("--seed", type=int, default=2109170720)
    parser.add_argument("--progress-every", type=int, choices=(100, 300), default=100)
    parser.add_argument("--interval-size", type=int, default=300)
    args = parser.parse_args()
    if (args.decisions is not None and args.decisions < 1) or args.interval_size < 1:
        parser.error("decisions and interval-size must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    import pygame
    import torch
    from Barrage import Barrage, Bullet, Plane
    from barrage_rl.distilled_student import UnifiedDistilledAgent
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.runtime_core import ACTION_VECTORS, snapshot_surface_rgb
    from barrage_rl.task_spec import TARGET_TASK, TARGET_TRACKING_CAPACITY
    from barrage_rl.timing import PHYSICS_FPS, DECISION_DT

    if args.decisions is None:
        args.decisions = round(120.0 / DECISION_DT)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; a CPU fallback would invalidate this comparison")
    torch.set_num_threads(10)
    torch.manual_seed(args.seed)
    checkpoint_hash = _file_digest(checkpoint)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sources = [PROJECT_ROOT / "Barrage.py", Path(__file__).resolve(), PROJECT_ROOT / "tools/benchmark_core_latency.py"]
    sources.extend(sorted((PROJECT_ROOT / "barrage_rl").glob("*.py")))
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): _file_digest(path) for path in sources}
    controllers = {
        device: LiveVisualController(str(checkpoint), device_name=device) for device in ("cpu", "cuda")
    }
    for device, controller in controllers.items():
        if not isinstance(controller.agent, UnifiedDistilledAgent):
            raise ValueError("This diagnostic requires the v42 unified recurrent distilled agent")
        if (
            controller.tracked_spec.max_objects != 384
            or controller.tracked_spec.tracker_capacity != 384
            or TARGET_TRACKING_CAPACITY != 384
            or controller.decision_interval != 4
            or controller.action_delay_steps != 0
            or not controller.tracked_extractor.tracker.refit_known_velocity
        ):
            raise ValueError("Expected original 384-slot, four-step, zero-delay v42 configuration")
        if controller.agent.device.type != device:
            raise RuntimeError(f"Requested {device}, received {controller.agent.device}")
        controller.agent.model.eval()
        controller.agent.distilled.eval()
        if any(module.training for model in (controller.agent.model, controller.agent.distilled) for module in model.modules()):
            raise RuntimeError("All model modules must be in eval mode")
    parameter_hashes = {
        device: {
            "backbone": _digest(controller.agent.model.state_dict()),
            "student": _digest(controller.agent.distilled.state_dict()),
        } for device, controller in controllers.items()
    }
    if parameter_hashes["cpu"] != parameter_hashes["cuda"]:
        raise RuntimeError("Loaded CPU/CUDA parameter bytes differ before inference")
    if TARGET_TASK.bullet_count != 300 or TARGET_TASK.targeted_bullet_probability != 0.10:
        raise ValueError("Current task must retain 300 bullets and independent 0.10 targeting")

    destination.mkdir(parents=True, exist_ok=False)
    config = {
        "format_version": 1, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "source_sha256": source_hashes, "parameter_sha256": parameter_hashes,
        "checkpoint_config": metadata.get("config", {}),
        "tracked_policy_spec": metadata.get("tracked_policy_spec", {}),
        "loaded_tracked_policy_spec": asdict(controllers["cpu"].tracked_spec),
        "distilled_student_spec": metadata.get("distilled_student_spec", {}),
        "task": TARGET_TASK.manifest(), "seed": args.seed, "step_decisions": args.decisions,
        "prime_decisions": 1, "physics_steps_per_decision": 4, "physics_fps": PHYSICS_FPS,
        "expected_game_seconds": args.decisions * DECISION_DT,
        "window_mode": "hidden SDL display using original Barrage sprite conversion and rendering",
        "damage_enabled": False, "world_driver": "CPU action only; CUDA is shadow",
        "shared_input": "Exactly one owned current RGB snapshot per decision, passed to both controllers",
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "torch": torch.__version__, "numpy": np.__version__, "pygame": pygame.version.ver,
        "python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cuda_runtime": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
        "cuda_properties": str(torch.cuda.get_device_properties(0)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "thresholds": {"safety": controllers["cpu"].agent.safety_threshold},
        "relative_norm_definition": "norm(cuda-cpu)/max(norm(cpu),1e-30), paired finite entries in float64",
        "hash_comparison": "Tensor device/repr metadata excluded; compare numeric dtype, shape and bytes",
        "observation_methods": "Read-only forward hooks and selector return observer; original act_features called once",
        "limitations": [
            "Hidden-window diagnostic, no frame-rate or latency inference.",
            "Immune trajectory; does not measure survival or formal 200-episode acceptance.",
            "One CPU-driven seed, no independent CUDA closed-loop world.",
            "Current backend settings retained; no precision/backend changes for artificial equality.",
            "The live world uses 300 bullets; legacy checkpoint normalization metadata remains unchanged.",
            "Masked infinities may appear in action scores; hidden finiteness is separate.",
            "Recurrent differences combine current backend rounding and carried hidden-state differences.",
            "Per-step tracker hash includes causal last-eight history plus full length; final hash includes all history.",
            "Distilled policy_correction/clearance/viable heads are observed predictions; their presence does not imply use in final action selection.",
        ],
    }
    _write_json_new(destination / "manifest.json", config)
    observers = {device: _ForwardObserver(controller) for device, controller in controllers.items()}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        pygame.display.init()
        Barrage.window = pygame.display.set_mode((820, 820), pygame.HIDDEN)
        Barrage.SCREEN_WIDTH = Barrage.SCREEN_HEIGHT = 820
        Barrage.QUANTITY = 300
        Barrage.BULLET_SIZE = 5
        Barrage.PHYSICS_FPS = PHYSICS_FPS
        Barrage.BULLET_SPEED = Barrage.PLANE_SPEED = 240.0
        Barrage.TARGETED_BULLET_PROBABILITY = 0.10
        Barrage.COLLISION = False
        Barrage.INVINCIBLE = False  # Existing Barrage flag: False disables damage.
        Barrage.MUSIC = False
        Barrage.KEY = True
        Barrage.AI_CONTROLLER = Barrage.AI_PIPELINE = None
        Barrage.RNG = np.random.default_rng(args.seed)
        Plane.SKIN = 0
        Barrage.reset_game()
        for controller in controllers.values():
            controller.reset()

        with (destination / "decisions.jsonl").open("x", encoding="utf-8") as stream:
            for decision in range(args.decisions + 1):
                if decision:
                    direction = ACTION_VECTORS[cpu_action]
                    for _ in range(4):
                        Barrage.advance_physics(direction, 1.0 / PHYSICS_FPS)
                    if Barrage.ALIVE_PHYSICS_STEPS != decision * 4:
                        raise RuntimeError("World failed to advance exactly four fixed physics steps")
                pygame.event.pump()
                Barrage.render_world()
                rgb = snapshot_surface_rgb(Barrage.window)
                rgb_before = _digest(rgb)
                actions: dict[str, int] = {}
                for device, controller in controllers.items():
                    observers[device].clear()
                    actions[device] = int(
                        controller.prime_rgb(rgb) if decision == 0
                        else controller.observe_rgb(rgb, physics_steps=4)
                    )
                    if _digest(rgb) != rgb_before:
                        raise RuntimeError(f"{device} controller mutated the shared RGB snapshot")
                cpu_action = actions["cpu"]
                states = {device: _state(controller) for device, controller in controllers.items()}
                signals = {device: observer.snapshot() for device, observer in observers.items()}
                if signals["cpu"].keys() != signals["cuda"].keys():
                    raise RuntimeError("Observed CPU/CUDA signal schemas differ")
                numeric = {
                    name: _array_difference(signals["cpu"][name], signals["cuda"][name])
                    for name in signals["cpu"]
                }
                first_numeric = next((
                    {"field": name, "numeric_difference": metrics}
                    for name, metrics in numeric.items() if not metrics["exact_bytes"]
                ), None)
                row = {
                    "kind": "prime" if decision == 0 else "step", "decision": decision,
                    "game_seconds": decision * DECISION_DT,
                    "cpu_action": actions["cpu"], "cuda_action": actions["cuda"],
                    "actions_equal": actions["cpu"] == actions["cuda"],
                    "rgb_sha256": rgb_before,
                    "world_sha256": _digest({
                        "plane_position": tuple(Barrage.PLANE.position),
                        "plane_velocity": tuple(Barrage.PLANE.velocity),
                        "bullet_state": Bullet.LIST, "rng": Barrage.RNG.bit_generator.state,
                        "physics_steps": Barrage.ALIVE_PHYSICS_STEPS,
                    }),
                    "state": _state_comparison(states["cpu"], states["cuda"]),
                    "first_numeric_difference": first_numeric, "numeric_signals": numeric,
                    "scores_and_thresholds": {
                        device: {
                            name: value for name, value in signals[device].items()
                            if name.startswith("threshold.") or name in (
                                "backbone.policy_head", "selection.scores", "selection.raw_actions",
                                "selection.learned_actions", "selection.actions",
                            )
                        } for device in ("cpu", "cuda")
                    },
                }
                stream.write(json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False) + "\n")
                # Keep compact state and aggregate metrics in RAM; detailed
                # per-head numeric records remain available in the JSONL file.
                records.append({key: value for key, value in row.items() if key not in ("numeric_signals", "scores_and_thresholds")})
                if decision == 0 or decision % args.progress_every == 0 or decision == args.decisions:
                    stream.flush()
                    hidden = row["state"]["hidden_combined"]
                    print(json.dumps(_json_value({
                        "decision": decision, "target": args.decisions,
                        "elapsed_seconds": time.perf_counter() - started,
                        "hidden_maxabs": hidden["maxabs"], "hidden_rmse": hidden["rmse"],
                        "actions_equal": row["actions_equal"],
                    })), flush=True)
        source_end = {str(path.relative_to(PROJECT_ROOT)): _file_digest(path) for path in sources}
        unchanged = source_end == source_hashes and _file_digest(checkpoint) == checkpoint_hash
        report = {
            "configuration": config, "wall_seconds_with_diagnostics": time.perf_counter() - started,
            "summary": _summary(records, args.interval_size),
            "final_complete_track_state_sha256": {
                device: _digest(controller.tracked_extractor.tracker.tracks)
                for device, controller in controllers.items()
            },
            "source_and_checkpoint_unchanged": unchanged, "source_sha256_end": source_end,
        }
        _write_json_new(destination / "report.json", report)
        if not unchanged:
            raise RuntimeError("Source/checkpoint changed during run; diagnostic has been marked invalid")
        print(f"Report: {destination / 'report.json'}", flush=True)
    except BaseException as exc:
        _write_json_new(destination / "error.json", {
            "type": type(exc).__name__, "message": str(exc),
            "completed_records": len(records), "elapsed_seconds": time.perf_counter() - started,
        })
        raise
    finally:
        for observer in observers.values():
            observer.close()
        pygame.display.quit()


if __name__ == "__main__":
    main()
