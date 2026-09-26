"""Process-parallel DAgger collection for persistent tracked policies."""

from __future__ import annotations

from .timing import DECISION_DT

import ctypes
import multiprocessing as mp
import threading
import traceback
from dataclasses import asdict
from queue import Empty
from typing import Any, Mapping, Sequence

import numpy as np

from .baselines import privileged_planner_supervision
from .env import BarrageVisionEnv
from .recovery_planner import vectorized_recovery_supervision
from .runtime_core import render_world_surface, snapshot_surface_rgb, tracker_prediction_hints
from .tracked_policy import TrackedFeatureExtractor, TrackedPolicySpec


class _RenderedRGBObservation:
    """Build the learner observation through the deployed RGB adapter."""

    def __init__(self, env: BarrageVisionEnv, output_size: int) -> None:
        import pygame

        from .live_screen import DominantBackgroundSemanticizer

        self.env = env
        self.pygame = pygame
        self.surface = pygame.Surface((env.screen_width, env.screen_height))
        self.semanticizer = DominantBackgroundSemanticizer(
            output_size=output_size,
            bullet_size=env.bullet_size,
        )

    def _frame(
        self,
        extractor: TrackedFeatureExtractor | None = None,
        decision_steps: int = 1,
    ) -> Any:
        env = self.env
        render_world_surface(
            self.surface,
            env.plane_surface,
            env.plane_position,
            env.bullet_surface,
            env.bullet_positions,
        )
        rgb = snapshot_surface_rgb(self.surface)
        if extractor is None or not extractor.initialized:
            return self.semanticizer.detect(rgb, include_semantic=False)
        predictions, radii = tracker_prediction_hints(
            extractor.tracker, tuple(rgb.shape), decision_steps
        )
        return self.semanticizer.detect(
            rgb,
            include_semantic=False,
            predicted_bullet_positions=predictions,
            prediction_search_radii=radii,
        )

    def detections(
        self,
        reset: bool,
        *,
        extractor: TrackedFeatureExtractor | None = None,
        decision_steps: int = 1,
    ) -> Any:
        if reset:
            self.semanticizer.reset()
            extractor = None
        return self._frame(extractor, decision_steps)


def _delayed_planner_supervision(
    env: BarrageVisionEnv,
    pending_action: int,
    *,
    teacher_kind: str,
    teacher_horizon_seconds: float,
    teacher_reaction_seconds: float,
    safety_horizons: Sequence[float],
) -> Any:
    """Label the command that will take effect after the pending command.

    The learner still receives only the current image-derived features.  This
    privileged training-only branch advances an exact saved simulator state by
    the already queued action, asks the configured teacher what should be sent
    for the following boundary, and then restores the environment bit-for-bit.
    """
    planner = (
        privileged_planner_supervision
        if teacher_kind == "exact"
        else vectorized_recovery_supervision
    )
    base = env.capture_state()
    try:
        env.simulate_action(
            int(pending_action),
            include_scheduled_opening=False,
            include_respawns=False,
        )
        return planner(
            env,
            horizon_seconds=teacher_horizon_seconds,
            reaction_seconds=teacher_reaction_seconds,
            wall_margin=80.0,
            wall_penalty_weight=0.35,
            safety_horizons=safety_horizons,
        )
    finally:
        env.restore_state(base)


def _shared_view(raw: Any, dtype: Any, shape: Sequence[int]) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _worker(
    indices: Sequence[int],
    env_count: int,
    seed: int,
    initial_episode_seeds: Sequence[int],
    repeat_initial_episode_seeds: bool,
    env_kwargs: Mapping[str, Any],
    spec_data: Mapping[str, Any],
    safety_horizons: Sequence[float],
    teacher_kind: str,
    teacher_horizon_seconds: float,
    teacher_reaction_seconds: float,
    deployment_rgb_observation: bool,
    causal_action_delay_steps: int,
    action_repeat_choices: Sequence[int],
    branch_output_dir: str,
    objects_raw: Any,
    masks_raw: Any,
    globals_raw: Any,
    teacher_actions_raw: Any,
    regrets_raw: Any,
    collisions_raw: Any,
    actions_raw: Any,
    done_raw: Any,
    truncated_raw: Any,
    survival_raw: Any,
    episode_serial_raw: Any,
    episode_steps_raw: Any,
    ready_barrier: Any,
    action_barrier: Any,
    stop_event: Any,
    error_queue: Any,
) -> None:
    action_count = len(BarrageVisionEnv.ACTIONS)
    horizon_count = len(safety_horizons)
    entries: list[
        tuple[
            int,
            BarrageVisionEnv,
            TrackedFeatureExtractor,
            _RenderedRGBObservation | None,
            np.random.Generator,
        ]
    ] = []
    try:
        spec = TrackedPolicySpec(**dict(spec_data))
        objects = _shared_view(
            objects_raw, np.float32,
            (env_count, spec.max_objects, spec.object_features),
        )
        masks = _shared_view(masks_raw, np.uint8, (env_count, spec.max_objects))
        globals_ = _shared_view(
            globals_raw, np.float32, (env_count, spec.global_features)
        )
        teacher_actions = _shared_view(teacher_actions_raw, np.int64, (env_count,))
        regrets = _shared_view(regrets_raw, np.float32, (env_count, action_count))
        collisions = _shared_view(
            collisions_raw, np.uint8, (env_count, horizon_count, action_count)
        )
        actions = _shared_view(actions_raw, np.int64, (env_count,))
        done = _shared_view(done_raw, np.uint8, (env_count,))
        truncated = _shared_view(truncated_raw, np.uint8, (env_count,))
        survival = _shared_view(survival_raw, np.float64, (env_count,))
        episode_serial = _shared_view(episode_serial_raw, np.int64, (env_count,))
        episode_steps = _shared_view(episode_steps_raw, np.int32, (env_count,))

        def publish(
            env_index: int,
            env: BarrageVisionEnv,
            extractor: TrackedFeatureExtractor,
            observation: np.ndarray,
            reset: bool,
            rendered: _RenderedRGBObservation | None,
            pending_action: int,
        ) -> None:
            extractor.tracker.decision_dt = env.decision_dt
            if rendered is not None:
                detected = rendered.detections(
                    reset, extractor=extractor, decision_steps=1
                )
                features = (
                    extractor.reset_detections(
                        detected.bullet_positions, detected.plane_position
                    )
                    if reset
                    else extractor.step_detections(
                        detected.bullet_positions, detected.plane_position
                    )
                )
            else:
                learner_observation = observation
                features = (
                    extractor.reset(learner_observation)
                    if reset
                    else extractor.step(learner_observation)
                )
            objects[env_index], masks[env_index], globals_[env_index] = features
            if causal_action_delay_steps:
                supervision = _delayed_planner_supervision(
                    env,
                    pending_action,
                    teacher_kind=teacher_kind,
                    teacher_horizon_seconds=teacher_horizon_seconds,
                    teacher_reaction_seconds=teacher_reaction_seconds,
                    safety_horizons=safety_horizons,
                )
            else:
                planner = (
                    privileged_planner_supervision
                    if teacher_kind == "exact"
                    else vectorized_recovery_supervision
                )
                supervision = planner(
                    env,
                    horizon_seconds=teacher_horizon_seconds,
                    reaction_seconds=teacher_reaction_seconds,
                    wall_margin=80.0,
                    wall_penalty_weight=0.35,
                    safety_horizons=safety_horizons,
                )
            teacher_actions[env_index] = supervision.action
            regrets[env_index] = supervision.regrets
            collisions[env_index] = supervision.safety_targets.astype(np.uint8)
            if env_index in recorders:
                recorders[env_index].observe(env, extractor, rendered, features, int(episode_serial[env_index]), supervision.safety_targets)

        from .branch_records import BranchRecorder
        recorders = {i: BranchRecorder(branch_output_dir, i) for i in indices} if branch_output_dir else {}
        pending_actions: dict[int, int] = {}
        for env_index in indices:
            env = BarrageVisionEnv(**dict(env_kwargs))
            timing_rng = np.random.default_rng(
                int(seed) + 7_919 * (int(env_index) + 1)
            )
            if action_repeat_choices:
                env.action_repeat = int(timing_rng.choice(action_repeat_choices))
            initial_seed = (
                int(initial_episode_seeds[env_index])
                if env_index < len(initial_episode_seeds)
                else seed + env_index
            )
            observation, _ = env.reset(seed=initial_seed)
            extractor = TrackedFeatureExtractor(spec, decision_dt=env.decision_dt)
            rendered = (
                _RenderedRGBObservation(env, int(env.observation_size))
                if deployment_rgb_observation
                else None
            )
            episode_serial[env_index] = 0
            episode_steps[env_index] = 0
            pending_actions[env_index] = 0
            publish(
                env_index,
                env,
                extractor,
                observation,
                True,
                rendered,
                pending_actions[env_index],
            )
            entries.append((env_index, env, extractor, rendered, timing_rng))

        ready_barrier.wait()
        while True:
            action_barrier.wait()
            if stop_event.is_set():
                break
            for env_index, env, extractor, rendered, timing_rng in entries:
                proposed_action = int(actions[env_index])
                applied_action = (
                    pending_actions[env_index]
                    if causal_action_delay_steps
                    else proposed_action
                )
                observation, _, terminated, was_truncated, info = env.step(
                    applied_action
                )
                pending_actions[env_index] = proposed_action
                completed = bool(terminated or was_truncated)
                episode_steps[env_index] += 1
                done[env_index] = completed
                truncated[env_index] = bool(was_truncated)
                if terminated and env_index in recorders:
                    recorders[env_index].failed()
                if completed:
                    survival[env_index] = float(info["survival_seconds"])
                    if (
                        repeat_initial_episode_seeds
                        and env_index < len(initial_episode_seeds)
                    ):
                        observation, _ = env.reset(
                            seed=int(initial_episode_seeds[env_index])
                        )
                    else:
                        observation, _ = env.reset()
                    episode_serial[env_index] += 1
                    episode_steps[env_index] = 0
                    pending_actions[env_index] = 0
                if action_repeat_choices:
                    env.action_repeat = int(timing_rng.choice(action_repeat_choices))
                publish(
                    env_index,
                    env,
                    extractor,
                    observation,
                    completed,
                    rendered,
                    pending_actions[env_index],
                )
            ready_barrier.wait()
    except BaseException:
        error_queue.put(traceback.format_exc())
        try:
            ready_barrier.abort()
            action_barrier.abort()
        except BaseException:
            pass
    finally:
        for recorder in locals().get('recorders', {}).values():
            recorder.flush()
        for _, env, _, _, _ in entries:
            env.close()


def _queued_error(queue: Any) -> str | None:
    try:
        return str(queue.get_nowait())
    except Empty:
        return None


class ParallelTrackedDaggerEnv:
    """Shared-memory feature, teacher-label, and environment pipeline."""

    def __init__(
        self,
        *,
        env_count: int,
        workers: int,
        seed: int,
        initial_episode_seeds: Sequence[int] = (),
        repeat_initial_episode_seeds: bool = False,
        env_kwargs: Mapping[str, Any],
        spec: TrackedPolicySpec,
        safety_horizons: Sequence[float] = (0.10, 0.30, 0.60, 1.20),
        teacher_kind: str = "recovery",
        teacher_horizon_seconds: float = 1.5,
        teacher_reaction_seconds: float = DECISION_DT,
        deployment_rgb_observation: bool = False,
        causal_action_delay_steps: int = 1,
        action_repeat_choices: Sequence[int] = (),
        branch_output_dir: str = "",
    ) -> None:
        if branch_output_dir and (not deployment_rgb_observation or causal_action_delay_steps):
            raise ValueError("branch recording requires current-frame RGB and zero action delay")
        if env_count < 1 or workers < 1:
            raise ValueError("env_count and workers must be positive")
        if teacher_kind not in ("exact", "recovery"):
            raise ValueError("teacher_kind must be 'exact' or 'recovery'")
        if causal_action_delay_steps not in (0, 1):
            raise ValueError("causal_action_delay_steps must be zero or one")
        initial_episode_seeds = tuple(int(value) for value in initial_episode_seeds)
        if len(initial_episode_seeds) > env_count:
            raise ValueError("initial_episode_seeds cannot exceed env_count")
        if len(set(initial_episode_seeds)) != len(initial_episode_seeds):
            raise ValueError("initial_episode_seeds must be unique")
        if any(value < 0 for value in initial_episode_seeds):
            raise ValueError("initial_episode_seeds must be non-negative")
        if repeat_initial_episode_seeds and not initial_episode_seeds:
            raise ValueError(
                "repeat_initial_episode_seeds requires initial_episode_seeds"
            )
        action_repeat_choices = tuple(int(value) for value in action_repeat_choices)
        if any(value < 1 for value in action_repeat_choices):
            raise ValueError("action_repeat_choices must contain positive integers")
        self.env_count = int(env_count)
        self.worker_count = min(int(workers), self.env_count)
        self.spec = spec
        self.safety_horizons = tuple(float(value) for value in safety_horizons)
        self.deployment_rgb_observation = bool(deployment_rgb_observation)
        self.causal_action_delay_steps = int(causal_action_delay_steps)
        self._closed = False
        action_count = len(BarrageVisionEnv.ACTIONS)
        horizon_count = len(self.safety_horizons)
        context = mp.get_context("spawn")
        self._objects_raw = context.RawArray(
            ctypes.c_float,
            self.env_count * spec.max_objects * spec.object_features,
        )
        self._masks_raw = context.RawArray(
            ctypes.c_ubyte, self.env_count * spec.max_objects
        )
        self._globals_raw = context.RawArray(
            ctypes.c_float, self.env_count * spec.global_features
        )
        self._teacher_actions_raw = context.RawArray(ctypes.c_int64, self.env_count)
        self._regrets_raw = context.RawArray(
            ctypes.c_float, self.env_count * action_count
        )
        self._collisions_raw = context.RawArray(
            ctypes.c_ubyte, self.env_count * horizon_count * action_count
        )
        self._actions_raw = context.RawArray(ctypes.c_int64, self.env_count)
        self._done_raw = context.RawArray(ctypes.c_ubyte, self.env_count)
        self._truncated_raw = context.RawArray(ctypes.c_ubyte, self.env_count)
        self._survival_raw = context.RawArray(ctypes.c_double, self.env_count)
        self._episode_serial_raw = context.RawArray(ctypes.c_int64, self.env_count)
        self._episode_steps_raw = context.RawArray(ctypes.c_int32, self.env_count)
        self._ready_barrier = context.Barrier(self.worker_count + 1)
        self._action_barrier = context.Barrier(self.worker_count + 1)
        self._stop_event = context.Event()
        self._error_queue = context.Queue()

        assignments = [
            list(range(index, self.env_count, self.worker_count))
            for index in range(self.worker_count)
        ]
        self._processes = [
            context.Process(
                target=_worker,
                args=(
                    assignment,
                    self.env_count,
                    int(seed),
                    initial_episode_seeds,
                    bool(repeat_initial_episode_seeds),
                    dict(env_kwargs),
                    asdict(spec),
                    self.safety_horizons,
                    str(teacher_kind),
                    float(teacher_horizon_seconds),
                    float(teacher_reaction_seconds),
                    self.deployment_rgb_observation,
                    self.causal_action_delay_steps,
                    action_repeat_choices,
                    str(branch_output_dir),
                    self._objects_raw,
                    self._masks_raw,
                    self._globals_raw,
                    self._teacher_actions_raw,
                    self._regrets_raw,
                    self._collisions_raw,
                    self._actions_raw,
                    self._done_raw,
                    self._truncated_raw,
                    self._survival_raw,
                    self._episode_serial_raw,
                    self._episode_steps_raw,
                    self._ready_barrier,
                    self._action_barrier,
                    self._stop_event,
                    self._error_queue,
                ),
            )
            for assignment in assignments
        ]
        self.objects = _shared_view(
            self._objects_raw, np.float32,
            (self.env_count, spec.max_objects, spec.object_features),
        )
        self.masks = _shared_view(
            self._masks_raw, np.uint8, (self.env_count, spec.max_objects)
        )
        self.globals = _shared_view(
            self._globals_raw, np.float32, (self.env_count, spec.global_features)
        )
        self.teacher_actions = _shared_view(
            self._teacher_actions_raw, np.int64, (self.env_count,)
        )
        self.regrets = _shared_view(
            self._regrets_raw, np.float32, (self.env_count, action_count)
        )
        self.collisions = _shared_view(
            self._collisions_raw, np.uint8,
            (self.env_count, horizon_count, action_count),
        )
        self.episode_serial = _shared_view(
            self._episode_serial_raw, np.int64, (self.env_count,)
        )
        self.episode_steps = _shared_view(
            self._episode_steps_raw, np.int32, (self.env_count,)
        )
        self._actions = _shared_view(
            self._actions_raw, np.int64, (self.env_count,)
        )
        self._done = _shared_view(self._done_raw, np.uint8, (self.env_count,))
        self._truncated = _shared_view(
            self._truncated_raw, np.uint8, (self.env_count,)
        )
        self._survival = _shared_view(
            self._survival_raw, np.float64, (self.env_count,)
        )
        for process in self._processes:
            process.start()
        try:
            self._ready_barrier.wait()
        except threading.BrokenBarrierError as error:
            message = _queued_error(self._error_queue) or str(error)
            self.close()
            raise RuntimeError(f"tracked collection worker failed:\n{message}") from error

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("tracked collection environment is closed")
        self._actions[:] = np.asarray(actions, dtype=np.int64)
        try:
            self._action_barrier.wait()
            self._ready_barrier.wait()
        except threading.BrokenBarrierError as error:
            message = _queued_error(self._error_queue) or str(error)
            raise RuntimeError(f"tracked collection worker failed:\n{message}") from error
        return (
            self._done.astype(np.bool_, copy=True),
            self._truncated.astype(np.bool_, copy=True),
            self._survival.copy(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        try:
            self._action_barrier.wait(timeout=5.0)
        except (threading.BrokenBarrierError, TimeoutError):
            pass
        for process in self._processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join()
        worker_error = _queued_error(self._error_queue)
        failed = [process.pid for process in self._processes if process.exitcode]
        if worker_error or failed:
            raise RuntimeError(worker_error or f"tracked workers failed: {failed}")

    def __enter__(self) -> "ParallelTrackedDaggerEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
