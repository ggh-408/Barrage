"""Causal image-to-action scheduling for real-time Barrage deployment.

The live pipeline owns the visual controller on one inference thread.  The
render/physics thread submits only the newest RGB frame and may apply a result
only at its declared future control boundary.  A late result is never applied
retroactively.
"""

from __future__ import annotations

from .timing import DECISION_DT

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .runtime_core import snapshot_surface_rgb


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "p99_ms": float(np.percentile(array, 99.0)),
        "max_ms": float(array.max()),
    }


@dataclass(frozen=True)
class CausalFrameRequest:
    generation: int
    sequence: int
    capture_boundary: int
    apply_boundary: int
    capture_timestamp: float
    deadline_timestamp: float
    rgb: np.ndarray


@dataclass(frozen=True)
class CausalActionResult:
    generation: int
    sequence: int
    capture_boundary: int
    apply_boundary: int
    capture_timestamp: float
    completed_timestamp: float
    action: int


class FixedActionDelay:
    """Deterministic action queue used by fixed-step evaluation."""

    def __init__(self, delay_steps: int = 1, initial_action: int = 0) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.delay_steps = int(delay_steps)
        self.initial_action = int(initial_action)
        self._queue = [self.initial_action] * self.delay_steps

    def reset(self) -> None:
        self._queue = [self.initial_action] * self.delay_steps

    def push(self, proposed_action: int) -> int:
        proposed = int(proposed_action)
        if not self.delay_steps:
            return proposed
        applied = int(self._queue.pop(0))
        self._queue.append(proposed)
        return applied


class CausalActionPipeline:
    """Latest-frame-only inference with one-boundary causal action delivery."""

    def __init__(
        self,
        controller: Any,
        *,
        decision_period_seconds: float = DECISION_DT,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        if decision_period_seconds <= 0.0:
            raise ValueError("decision_period_seconds must be positive")
        self.controller = controller
        self.decision_period_seconds = float(decision_period_seconds)
        self._time_fn = time_fn
        self._condition = threading.Condition()
        self._generation = 0
        self._worker_generation = -1
        self._sequence = 0
        self._pending: CausalFrameRequest | None = None
        self._completed: dict[int, CausalActionResult] = {}
        self._closed = False
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed_count = 0
        self._applied = 0
        self._deadline_misses = 0
        self._dropped_pending = 0
        self._discarded_results = 0
        self._preloaded_results = 0
        self._preloaded_boundaries: set[int] = set()
        self._skipped_stale_requests = 0
        self._capture_copy_ms: list[float] = []
        self._queue_wait_ms: list[float] = []
        self._inference_ms: list[float] = []
        self._capture_to_completion_ms: list[float] = []
        self._submission_intervals_ms: list[float] = []
        self._capture_to_apply_ms: list[float] = []
        self._last_submission_timestamp: float | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="barrage-causal-inference",
            daemon=True,
        )
        self._thread.start()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("causal inference worker failed") from self._error

    def reset(self) -> None:
        """Start a new episode without racing a still-running old inference."""
        with self._condition:
            self._raise_worker_error()
            self._generation += 1
            self._sequence = 0
            self._pending = None
            self._completed = {}
            self._submitted = 0
            self._completed_count = 0
            self._applied = 0
            self._deadline_misses = 0
            self._dropped_pending = 0
            self._discarded_results = 0
            self._preloaded_results = 0
            self._preloaded_boundaries = set()
            self._skipped_stale_requests = 0
            self._capture_copy_ms = []
            self._queue_wait_ms = []
            self._inference_ms = []
            self._capture_to_completion_ms = []
            self._submission_intervals_ms = []
            self._capture_to_apply_ms = []
            self._last_submission_timestamp = None
            self._condition.notify_all()

    def wait_until_ready(self, apply_boundary: int, timeout: float = 5.0) -> None:
        """Wait only during episode setup for a declared boundary result."""
        deadline = self._time_fn() + max(0.0, float(timeout))
        with self._condition:
            while int(apply_boundary) not in self._completed:
                self._raise_worker_error()
                remaining = deadline - self._time_fn()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "causal inference did not finish episode setup"
                    )
                self._condition.wait(timeout=remaining)

    def begin_measurement(self) -> None:
        """Exclude episode-setup warmup while retaining its queued action."""
        controller_begin = getattr(self.controller, "begin_measurement", None)
        if controller_begin is not None:
            controller_begin()
        with self._condition:
            self._raise_worker_error()
            self._preloaded_results = len(self._completed)
            self._preloaded_boundaries = set(self._completed)
            self._submitted = 0
            self._completed_count = 0
            self._applied = 0
            self._deadline_misses = 0
            self._dropped_pending = 0
            self._discarded_results = 0
            self._skipped_stale_requests = 0
            self._capture_copy_ms = []
            self._queue_wait_ms = []
            self._inference_ms = []
            self._capture_to_completion_ms = []
            self._submission_intervals_ms = []
            self._capture_to_apply_ms = []
            self._last_submission_timestamp = None

    def submit_rgb(
        self,
        rgb: np.ndarray,
        *,
        capture_boundary: int,
        apply_boundary: int | None = None,
        capture_timestamp: float | None = None,
        enforce_deadline: bool = True,
    ) -> int:
        """Submit a frame, replacing any queued frame that has not begun."""
        return self._submit_rgb(
            rgb,
            capture_boundary=capture_boundary,
            apply_boundary=apply_boundary,
            capture_timestamp=capture_timestamp,
            enforce_deadline=enforce_deadline,
            copy_frame=True,
        )

    def _submit_rgb(
        self,
        rgb: np.ndarray,
        *,
        capture_boundary: int,
        apply_boundary: int | None,
        capture_timestamp: float | None,
        enforce_deadline: bool,
        copy_frame: bool,
    ) -> int:
        capture = self._time_fn() if capture_timestamp is None else float(
            capture_timestamp
        )
        capture_boundary = int(capture_boundary)
        apply_boundary = (
            capture_boundary + 1
            if apply_boundary is None
            else int(apply_boundary)
        )
        if apply_boundary <= capture_boundary:
            raise ValueError("apply_boundary must be after capture_boundary")
        frame = np.array(rgb, copy=True, order="C") if copy_frame else rgb
        with self._condition:
            self._raise_worker_error()
            if self._closed:
                raise RuntimeError("causal action pipeline is closed")
            if self._pending is not None:
                self._dropped_pending += 1
            self._sequence += 1
            request = CausalFrameRequest(
                generation=self._generation,
                sequence=self._sequence,
                capture_boundary=capture_boundary,
                apply_boundary=apply_boundary,
                capture_timestamp=capture,
                deadline_timestamp=(
                    capture + self.decision_period_seconds
                    if enforce_deadline
                    else float("inf")
                ),
                rgb=frame,
            )
            self._pending = request
            self._submitted += 1
            if self._last_submission_timestamp is not None:
                self._submission_intervals_ms.append(
                    (capture - self._last_submission_timestamp) * 1000.0
                )
            self._last_submission_timestamp = capture
            self._condition.notify()
            return request.sequence

    def submit_surface(
        self,
        surface: Any,
        *,
        capture_boundary: int,
        apply_boundary: int | None = None,
        enforce_deadline: bool = True,
    ) -> int:
        """Copy the current Pygame surface on the render thread, then submit."""
        started = self._time_fn()
        rgb = snapshot_surface_rgb(surface)
        copied = self._time_fn()
        with self._condition:
            self._capture_copy_ms.append((copied - started) * 1000.0)
        return self._submit_rgb(
            rgb,
            capture_boundary=capture_boundary,
            apply_boundary=apply_boundary,
            capture_timestamp=started,
            enforce_deadline=enforce_deadline,
            copy_frame=False,
        )

    def action_for_boundary(self, boundary: int, fallback_action: int) -> int:
        """Return only an on-time result for this exact control boundary."""
        boundary = int(boundary)
        now = self._time_fn()
        with self._condition:
            self._raise_worker_error()
            stale_boundaries = [
                value for value in self._completed if value < boundary
            ]
            for stale_boundary in stale_boundaries:
                del self._completed[stale_boundary]
                self._discarded_results += 1
            result = self._completed.pop(boundary, None)
            if result is not None:
                self._applied += 1
                if boundary in self._preloaded_boundaries:
                    self._preloaded_boundaries.remove(boundary)
                else:
                    self._capture_to_apply_ms.append(
                        (now - result.capture_timestamp) * 1000.0
                    )
                return int(result.action)
            self._deadline_misses += 1
            return int(fallback_action)

    def report(self) -> dict[str, Any]:
        with self._condition:
            self._raise_worker_error()
            decisions = self._applied + self._deadline_misses
            report = {
                "action_delay_boundaries": 1,
                "submitted_frames": int(self._submitted),
                "completed_inferences": int(self._completed_count),
                "applied_actions": int(self._applied),
                "deadline_miss_count": int(self._deadline_misses),
                "deadline_miss_fraction": float(
                    self._deadline_misses / max(decisions, 1)
                ),
                "dropped_pending_frames": int(self._dropped_pending),
                "discarded_late_or_superseded_results": int(
                    self._discarded_results
                ),
                "preloaded_results": int(self._preloaded_results),
                "skipped_stale_requests": int(
                    self._skipped_stale_requests
                ),
                "capture_copy": _statistics(self._capture_copy_ms),
                "queue_wait": _statistics(self._queue_wait_ms),
                "inference": _statistics(self._inference_ms),
                "capture_to_completion": _statistics(
                    self._capture_to_completion_ms
                ),
                "submission_interval": _statistics(
                    self._submission_intervals_ms
                ),
                "capture_to_apply": _statistics(self._capture_to_apply_ms),
            }
            controller_report = getattr(
                self.controller, "runtime_stage_report", None
            )
            if controller_report is not None:
                report["controller_stages"] = controller_report()
            return report

    def close(self, timeout: float = 5.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(timeout)))
        if self._thread.is_alive():
            raise RuntimeError("causal inference worker did not stop")
        self._raise_worker_error()

    def _worker(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    request = self._pending
                    self._pending = None
                if request is None:
                    continue
                if request.generation != self._worker_generation:
                    self.controller.reset()
                    self._worker_generation = request.generation
                started = self._time_fn()
                if started >= request.deadline_timestamp:
                    with self._condition:
                        if request.generation == self._generation:
                            self._skipped_stale_requests += 1
                            self._discarded_results += 1
                    continue
                boundary_action = getattr(
                    self.controller, "act_rgb_frame_at_boundary", None
                )
                action = int(
                    boundary_action(request.rgb, request.capture_boundary)
                    if boundary_action is not None
                    else self.controller.act_rgb_frame(request.rgb)
                )
                completed = self._time_fn()
                result = CausalActionResult(
                    generation=request.generation,
                    sequence=request.sequence,
                    capture_boundary=request.capture_boundary,
                    apply_boundary=request.apply_boundary,
                    capture_timestamp=request.capture_timestamp,
                    completed_timestamp=completed,
                    action=action,
                )
                with self._condition:
                    if request.generation != self._generation:
                        self._discarded_results += 1
                        continue
                    self._queue_wait_ms.append(
                        (started - request.capture_timestamp) * 1000.0
                    )
                    self._inference_ms.append((completed - started) * 1000.0)
                    self._capture_to_completion_ms.append(
                        (completed - request.capture_timestamp) * 1000.0
                    )
                    self._completed_count += 1
                    if completed > request.deadline_timestamp:
                        self._discarded_results += 1
                        continue
                    if request.apply_boundary in self._completed:
                        self._discarded_results += 1
                    self._completed[request.apply_boundary] = result
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
