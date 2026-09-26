"""Check exact window-query caching and invalidation against uncached models."""

import copy
import unittest

import torch

from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from barrage_rl.window_inference import enable_window_inference


class WindowInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _fixture(self, device="cpu"):
        torch.manual_seed(20260901)
        model = ActionQueryPolicy(
            TrackedPolicySpec(max_objects=8),
            width=16,
            attention_layers=1,
            attention_heads=4,
        ).to(device).eval()
        objects = torch.randn(2, 8, 16, device=device) * 0.1
        masks = torch.zeros(2, 8, dtype=torch.bool, device=device)
        masks[1, :5] = True
        globals_ = torch.zeros(2, 16, device=device)
        globals_[:, :2] = 0.5
        return model, (objects, masks, globals_)

    def _assert_outputs(self, model, reference, inputs):
        actual = model(*inputs)
        expected = reference(*inputs)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(
                left, right, rtol=0.0, atol=0.0, equal_nan=True
            )
        return actual

    def test_repeated_inference_is_exact_and_keeps_checkpoint_schema(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                model, inputs = self._fixture(device)
                reference = copy.deepcopy(model)
                enable_window_inference(model)
                cache = model._window_action_query_cache
                with torch.inference_mode():
                    self._assert_outputs(model, reference, inputs)
                    saved = cache._value
                    self._assert_outputs(model, reference, inputs)
                    self.assertIs(cache._value, saved)
                self.assertEqual(list(model.state_dict()), list(reference.state_dict()))

    def test_weight_edits_load_and_parameter_replacement_invalidate(self):
        model, inputs = self._fixture()
        reference = copy.deepcopy(model)
        enable_window_inference(model)
        cache = model._window_action_query_cache
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        previous = cache._value
        with torch.no_grad():
            model.action_embedding.weight.add_(0.125)
        reference.load_state_dict(model.state_dict())
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        self.assertIsNot(cache._value, previous)

        previous = cache._value
        replacement = copy.deepcopy(reference.state_dict())
        replacement["action_vector_encoder.0.weight"].mul_(0.75)
        model.load_state_dict(replacement)
        reference.load_state_dict(replacement)
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        self.assertIsNot(cache._value, previous)

        previous = cache._value
        model.action_embedding.weight = torch.nn.Parameter(
            model.action_embedding.weight.detach().clone() + 0.25
        )
        reference.load_state_dict(model.state_dict())
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        self.assertIsNot(cache._value, previous)

    def test_training_and_gradient_paths_bypass_cache(self):
        model, inputs = self._fixture()
        reference = copy.deepcopy(model)
        enable_window_inference(model)
        cache = model._window_action_query_cache
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        self.assertIsNotNone(cache._value)

        for training in (True, False):
            model.train(training)
            reference.train(training)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            actual = model(*inputs)
            expected = reference(*inputs)
            sum(value.square().sum() for value in actual).backward()
            sum(value.square().sum() for value in expected).backward()
            self.assertIsNone(cache._value)
            for left, right in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=0.0, atol=0.0)
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        self.assertIsNotNone(cache._value)

    def test_dtype_changes_and_autocast_keep_exact_outputs(self):
        model, inputs = self._fixture()
        reference = copy.deepcopy(model)
        enable_window_inference(model)
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        model.double()
        reference.double()
        inputs = tuple(value if value.dtype == torch.bool else value.double()
                       for value in inputs)
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
        self.assertEqual(model._window_action_query_cache._value.dtype, torch.float64)

        model.float()
        reference.float()
        inputs = tuple(value if value.dtype == torch.bool else value.float()
                       for value in inputs)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            self._assert_outputs(model, reference, inputs)
        self.assertIsNone(model._window_action_query_cache._value)

    def test_thread_count_change_refreshes_the_prefix(self):
        model, inputs = self._fixture()
        reference = copy.deepcopy(model)
        enable_window_inference(model)
        with torch.inference_mode():
            self._assert_outputs(model, reference, inputs)
            previous = model._window_action_query_cache._value
            try:
                torch.set_num_threads(2)
                self._assert_outputs(model, reference, inputs)
                self.assertIsNot(model._window_action_query_cache._value, previous)
            finally:
                torch.set_num_threads(1)


if __name__ == "__main__":
    unittest.main()
