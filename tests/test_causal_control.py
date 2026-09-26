"""Causal scheduling regression tests."""

import threading
import time

import numpy as np

from barrage_rl.causal_control import CausalActionPipeline, FixedActionDelay


class _ImmediateController:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def act_rgb_frame(self, rgb: np.ndarray) -> int:
        return int(rgb[0, 0, 0])


class _BlockingController(_ImmediateController):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def act_rgb_frame(self, rgb: np.ndarray) -> int:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            if not self.release.wait(timeout=2.0):
                raise TimeoutError("test inference was not released")
        return super().act_rgb_frame(rgb)


class _BoundaryController(_ImmediateController):
    def __init__(self) -> None:
        super().__init__()
        self.boundaries: list[int] = []

    def act_rgb_frame_at_boundary(
        self, rgb: np.ndarray, capture_boundary: int
    ) -> int:
        self.boundaries.append(int(capture_boundary))
        return super().act_rgb_frame(rgb)


def _frame(action: int) -> np.ndarray:
    return np.full((2, 2, 3), int(action), dtype=np.uint8)


def _wait_for_completed(
    pipeline: CausalActionPipeline, count: int, timeout: float = 2.0
) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if int(pipeline.report()["completed_inferences"]) >= count:
            return
        time.sleep(0.002)
    raise AssertionError(f"pipeline did not complete {count} inference requests")


def test_fixed_action_delay_applies_previous_decision() -> None:
    delay = FixedActionDelay(delay_steps=1, initial_action=0)
    assert delay.push(4) == 0
    assert delay.push(7) == 4
    assert delay.push(2) == 7


def test_pipeline_applies_result_only_at_declared_boundary() -> None:
    controller = _ImmediateController()
    pipeline = CausalActionPipeline(controller)
    try:
        pipeline.reset()
        pipeline.submit_rgb(
            _frame(7), capture_boundary=0, apply_boundary=1
        )
        _wait_for_completed(pipeline, 1)
        assert pipeline.action_for_boundary(1, fallback_action=3) == 7
        assert pipeline.action_for_boundary(2, fallback_action=7) == 7
        report = pipeline.report()
        assert report["applied_actions"] == 1
        assert report["deadline_miss_count"] == 1
        assert controller.reset_count == 1
    finally:
        pipeline.close()


def test_pipeline_discards_late_result_instead_of_applying_retroactively() -> None:
    controller = _BlockingController()
    pipeline = CausalActionPipeline(controller)
    try:
        pipeline.reset()
        pipeline.submit_rgb(
            _frame(8), capture_boundary=0, apply_boundary=1
        )
        assert controller.started.wait(timeout=1.0)
        assert pipeline.action_for_boundary(1, fallback_action=5) == 5
        controller.release.set()
        _wait_for_completed(pipeline, 1)
        assert pipeline.action_for_boundary(2, fallback_action=5) == 5
        report = pipeline.report()
        assert report["applied_actions"] == 0
        assert report["deadline_miss_count"] == 2
        assert report["discarded_late_or_superseded_results"] >= 1
    finally:
        controller.release.set()
        pipeline.close()


def test_pipeline_queue_keeps_only_latest_waiting_frame() -> None:
    controller = _BlockingController()
    pipeline = CausalActionPipeline(controller)
    try:
        pipeline.reset()
        pipeline.submit_rgb(
            _frame(1), capture_boundary=0, apply_boundary=1
        )
        assert controller.started.wait(timeout=1.0)
        pipeline.submit_rgb(
            _frame(2), capture_boundary=1, apply_boundary=2
        )
        pipeline.submit_rgb(
            _frame(3), capture_boundary=2, apply_boundary=3
        )
        controller.release.set()
        _wait_for_completed(pipeline, 2)
        assert controller.calls == 2
        assert pipeline.action_for_boundary(3, fallback_action=0) == 3
        report = pipeline.report()
        assert report["dropped_pending_frames"] == 1
        assert report["discarded_late_or_superseded_results"] >= 1
    finally:
        controller.release.set()
        pipeline.close()


def test_pipeline_passes_capture_boundary_to_aware_controller() -> None:
    controller = _BoundaryController()
    pipeline = CausalActionPipeline(controller)
    try:
        pipeline.reset()
        pipeline.submit_rgb(_frame(4), capture_boundary=7, apply_boundary=8)
        _wait_for_completed(pipeline, 1)
        assert controller.boundaries == [7]
    finally:
        pipeline.close()
