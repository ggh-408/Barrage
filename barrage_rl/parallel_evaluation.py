"""Process-parallel environment rollouts with centralized policy inference."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import threading
import traceback
from dataclasses import asdict, dataclass
from queue import Empty
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .env import BarrageVisionEnv
from .visual_set import SemanticFrameExtractor, VisualSetSpec


@dataclass
class ParallelRolloutResult:
    survival_times: np.ndarray
    termination_reasons: List[str]
    bullet_sizes: np.ndarray
    bullet_speeds: np.ndarray
    scenario_sources: List[str]
    reset_modes: List[str]
    minimum_wall_distances: np.ndarray
    wall_steps: int
    model_steps: int
    action_histogram: np.ndarray


def _shared_view(raw: Any, dtype: Any, shape: Sequence[int]) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _rollout_worker(
    episode_indices: Sequence[int],
    episodes: int,
    seed: int,
    env_kwargs: Mapping[str, Any],
    spec_data: Mapping[str, Any],
    wall_threshold: float,
    objects_raw: Any,
    masks_raw: Any,
    globals_raw: Any,
    actions_raw: Any,
    active_raw: Any,
    survival_raw: Any,
    ready_barrier: Any,
    action_barrier: Any,
    stop_event: Any,
    result_queue: Any,
    error_queue: Any,
) -> None:
    envs: List[tuple[int, BarrageVisionEnv]] = []
    try:
        spec = VisualSetSpec(**dict(spec_data))
        extractor = SemanticFrameExtractor(spec)
        objects = _shared_view(
            objects_raw,
            np.float32,
            (episodes, spec.max_objects, spec.object_features),
        )
        masks = _shared_view(masks_raw, np.uint8, (episodes, spec.max_objects))
        globals_ = _shared_view(
            globals_raw, np.float32, (episodes, spec.global_features)
        )
        actions = _shared_view(actions_raw, np.int64, (episodes,))
        active = _shared_view(active_raw, np.uint8, (episodes,))
        survival = _shared_view(survival_raw, np.float64, (episodes,))

        minimum_wall_distances: Dict[int, float] = {}
        records: Dict[int, Dict[str, Any]] = {}
        wall_steps = 0
        model_steps = 0
        histogram = np.zeros(len(BarrageVisionEnv.ACTIONS), dtype=np.int64)

        for episode_index in episode_indices:
            env = BarrageVisionEnv(**dict(env_kwargs))
            observation, _ = env.reset(seed=seed + episode_index)
            episode_objects, episode_mask, episode_globals = extractor.extract(
                observation
            )
            objects[episode_index] = episode_objects
            masks[episode_index] = episode_mask
            globals_[episode_index] = episode_globals
            minimum_wall_distances[episode_index] = float("inf")
            active[episode_index] = 1
            envs.append((episode_index, env))

        ready_barrier.wait()
        while True:
            action_barrier.wait()
            if stop_event.is_set():
                break
            for episode_index, env in envs:
                if not active[episode_index]:
                    continue
                action = int(actions[episode_index])
                half_plane = env.plane_size / 2.0
                wall_distance = min(
                    float(env.plane_position[0] - half_plane[0]),
                    float(
                        env.screen_width
                        - half_plane[0]
                        - env.plane_position[0]
                    ),
                    float(env.plane_position[1] - half_plane[1]),
                    float(
                        env.screen_height
                        - half_plane[1]
                        - env.plane_position[1]
                    ),
                )
                minimum_wall_distances[episode_index] = min(
                    minimum_wall_distances[episode_index], wall_distance
                )
                wall_steps += int(wall_distance < wall_threshold)
                model_steps += 1
                histogram[action] += 1

                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    active[episode_index] = 0
                    survival[episode_index] = float(info["survival_seconds"])
                    records[episode_index] = {
                        "termination_reason": (
                            "collision" if terminated else "time_limit"
                        ),
                        "bullet_size": int(info["bullet_size"]),
                        "bullet_speed": float(info["bullet_speed"]),
                        "scenario_source": str(info["scenario_source"]),
                        "reset_mode": str(info["reset_mode"]),
                    }
                else:
                    episode_objects, episode_mask, episode_globals = extractor.extract(
                        observation
                    )
                    objects[episode_index] = episode_objects
                    masks[episode_index] = episode_mask
                    globals_[episode_index] = episode_globals
            ready_barrier.wait()

        result_queue.put(
            {
                "minimum_wall_distances": minimum_wall_distances,
                "records": records,
                "wall_steps": wall_steps,
                "model_steps": model_steps,
                "action_histogram": histogram,
            }
        )
    except BaseException:
        error_queue.put(traceback.format_exc())
        try:
            ready_barrier.abort()
            action_barrier.abort()
        except BaseException:
            pass
    finally:
        for _, env in envs:
            env.close()


def _queued_error(error_queue: Any) -> Optional[str]:
    try:
        return str(error_queue.get_nowait())
    except Empty:
        return None


def run_parallel_rollout(
    agent: Any,
    spec: VisualSetSpec,
    episodes: int,
    workers: int,
    seed: int,
    env_kwargs: Mapping[str, Any],
    wall_threshold: float,
    progress_interval: int = 0,
) -> ParallelRolloutResult:
    """Run deterministic model evaluation episodes in worker processes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    worker_count = max(1, min(int(workers), episodes))
    context = mp.get_context("spawn")

    objects_raw = context.RawArray(
        ctypes.c_float, episodes * spec.max_objects * spec.object_features
    )
    masks_raw = context.RawArray(ctypes.c_ubyte, episodes * spec.max_objects)
    globals_raw = context.RawArray(ctypes.c_float, episodes * spec.global_features)
    actions_raw = context.RawArray(ctypes.c_int64, episodes)
    active_raw = context.RawArray(ctypes.c_ubyte, episodes)
    survival_raw = context.RawArray(ctypes.c_double, episodes)
    ready_barrier = context.Barrier(worker_count + 1)
    action_barrier = context.Barrier(worker_count + 1)
    stop_event = context.Event()
    result_queue = context.Queue()
    error_queue = context.Queue()

    assignments = [
        list(range(index, episodes, worker_count)) for index in range(worker_count)
    ]
    processes = [
        context.Process(
            target=_rollout_worker,
            args=(
                assignment,
                episodes,
                seed,
                dict(env_kwargs),
                asdict(spec),
                wall_threshold,
                objects_raw,
                masks_raw,
                globals_raw,
                actions_raw,
                active_raw,
                survival_raw,
                ready_barrier,
                action_barrier,
                stop_event,
                result_queue,
                error_queue,
            ),
        )
        for assignment in assignments
    ]

    objects = _shared_view(
        objects_raw,
        np.float32,
        (episodes, spec.max_objects, spec.object_features),
    )
    masks = _shared_view(masks_raw, np.uint8, (episodes, spec.max_objects))
    globals_ = _shared_view(globals_raw, np.float32, (episodes, spec.global_features))
    actions = _shared_view(actions_raw, np.int64, (episodes,))
    active = _shared_view(active_raw, np.uint8, (episodes,))
    survival = _shared_view(survival_raw, np.float64, (episodes,))

    for process in processes:
        process.start()
    try:
        ready_barrier.wait()
        agent.reset(episodes)
        next_progress = progress_interval
        last_progress = 0
        while True:
            active_indices = np.flatnonzero(active)
            if not len(active_indices):
                stop_event.set()
                action_barrier.wait()
                break
            full_hidden = agent.hidden
            if full_hidden is None:
                raise RuntimeError("agent hidden state is not initialized")
            agent.hidden = full_hidden[:, active_indices].contiguous()
            active_actions = agent.act_features(
                objects[active_indices],
                masks[active_indices].astype(np.bool_),
                globals_[active_indices],
                deterministic=True,
            )
            active_hidden = agent.hidden
            agent.hidden = full_hidden
            with torch.inference_mode():
                agent.hidden[:, active_indices] = active_hidden
            actions[active_indices] = active_actions
            action_barrier.wait()
            ready_barrier.wait()

            if progress_interval > 0:
                completed = episodes - int(np.count_nonzero(active))
                while next_progress <= completed and next_progress <= episodes:
                    print(
                        f"evaluation_progress={next_progress}/{episodes}",
                        flush=True,
                    )
                    last_progress = next_progress
                    next_progress += progress_interval
        if (
            progress_interval > 0
            and last_progress < episodes
        ):
            print(f"evaluation_progress={episodes}/{episodes}", flush=True)
    except threading.BrokenBarrierError as error:
        message = _queued_error(error_queue) or str(error)
        raise RuntimeError(f"evaluation worker failed:\n{message}") from error
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join()

    worker_error = _queued_error(error_queue)
    failed_processes = [process.pid for process in processes if process.exitcode]
    if worker_error or failed_processes:
        raise RuntimeError(
            worker_error or f"evaluation workers failed: {failed_processes}"
        )

    termination_reasons = [""] * episodes
    bullet_sizes = np.zeros(episodes, dtype=np.int64)
    bullet_speeds = np.zeros(episodes, dtype=np.float64)
    scenario_sources = [""] * episodes
    reset_modes = [""] * episodes
    minimum_wall_distances = np.full(episodes, np.inf, dtype=np.float64)
    wall_steps = 0
    model_steps = 0
    histogram = np.zeros(len(BarrageVisionEnv.ACTIONS), dtype=np.int64)
    for _ in processes:
        payload = result_queue.get()
        wall_steps += int(payload["wall_steps"])
        model_steps += int(payload["model_steps"])
        histogram += np.asarray(payload["action_histogram"], dtype=np.int64)
        for episode_index, value in payload["minimum_wall_distances"].items():
            minimum_wall_distances[int(episode_index)] = float(value)
        for episode_index, record in payload["records"].items():
            index = int(episode_index)
            termination_reasons[index] = record["termination_reason"]
            bullet_sizes[index] = int(record["bullet_size"])
            bullet_speeds[index] = float(record["bullet_speed"])
            scenario_sources[index] = record["scenario_source"]
            reset_modes[index] = record["reset_mode"]

    return ParallelRolloutResult(
        survival_times=survival.copy(),
        termination_reasons=termination_reasons,
        bullet_sizes=bullet_sizes,
        bullet_speeds=bullet_speeds,
        scenario_sources=scenario_sources,
        reset_modes=reset_modes,
        minimum_wall_distances=minimum_wall_distances,
        wall_steps=wall_steps,
        model_steps=model_steps,
        action_histogram=histogram,
    )
