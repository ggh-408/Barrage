"""Behavioral regressions for the pinned window inference fast paths."""
import copy
import numpy as np
import torch

from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec, TrackedFeatureExtractor
from barrage_rl.runtime_core import tracker_prediction_hints
from tools.benchmark_tracker_exact import exact, state


def test_prediction_reuse_handles_missing_detections_skipped_steps_and_reset():
    rng = np.random.default_rng(124)
    before = TrackedFeatureExtractor()
    after = copy.deepcopy(before)
    positions = rng.uniform(.1, .9, (300, 2)).astype(np.float32)
    for frame in range(100):
        steps = 1 + frame % 3
        if frame == 60:
            before = TrackedFeatureExtractor()
            after = copy.deepcopy(before)
        plane = None if frame % 7 == 0 else np.array([.5, .5], np.float32)
        bullets = positions if frame % 11 else positions[:0]
        exact(tracker_prediction_hints(before.tracker, (820, 820, 3), steps),
              tracker_prediction_hints(after.tracker, (820, 820, 3), steps, reuse_for_update=True))
        exact(before.step_detections(bullets, plane, decision_steps=steps),
              after.step_detections(bullets, plane, decision_steps=steps))
        exact(state(before), state(after))


def test_teacher_cost_skip_is_inference_only_and_preserves_policy():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = ActionQueryPolicy(TrackedPolicySpec(max_objects=8), width=16,
            attention_layers=1, attention_heads=4).eval()
        inputs = (torch.randn(2, 8, 16) * .1, torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 16))
        calls = []
        hook = model.teacher_cost_head.register_forward_hook(lambda *args: calls.append(1))
        with torch.inference_mode():
            expected = model.forward_with_geometry(*inputs)
            actual = model.forward_with_geometry(*inputs, compute_teacher_cost=False)
        assert len(calls) == 1
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        assert torch.count_nonzero(actual[1]) == 0
        model.train()
        teacher = model.forward_with_geometry(*inputs, compute_teacher_cost=False)[1]
        teacher.sum().backward()
        assert len(calls) == 2
        assert model.teacher_cost_head.weight.grad is not None
        hook.remove()
    finally:
        torch.set_num_threads(previous)
