"""Correctness checks for latency artifacts and strict before/after parity."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.benchmark_core_latency import _check_baseline, _digest, _write_json


class CoreLatencyHarnessTests(unittest.TestCase):
    def test_digest_ignores_array_strides_but_preserves_exact_values(self) -> None:
        source = np.arange(60, dtype=np.uint8).reshape(4, 5, 3).transpose(1, 0, 2)
        self.assertFalse(source.flags.c_contiguous)
        self.assertEqual(_digest(source), _digest(np.ascontiguousarray(source)))
        changed = source.copy()
        changed[1, 1, 1] += 1
        self.assertNotEqual(_digest(source), _digest(changed))
        self.assertNotEqual(_digest(source), _digest(source.astype(np.int32)))
        self.assertNotEqual(_digest(source), _digest(source.reshape(-1)))

    def test_history_timestamp_and_old_position_affect_digest(self) -> None:
        history = [(0, np.array([1.0, 2.0], np.float32)), (1, np.array([3.0, 4.0], np.float32))]
        changed = deepcopy(history)
        changed[0][1][0] = np.nextafter(np.float32(1.0), np.float32(2.0))
        self.assertNotEqual(_digest(history), _digest(changed))
        self.assertNotEqual(_digest(history), _digest([(9, history[0][1]), history[1]]))

    def test_json_publication_refuses_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            _write_json(destination, {"original": 1})
            original = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                _write_json(destination, {"replacement": 2})
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    @staticmethod
    def _report() -> dict:
        return {
            "format_version": 1, "kind": "core_latency_and_exact_parity",
            "config": {"steps": 1}, "checkpoint_sha256": "checkpoint",
            "source_root": "original",
            "results": [{
                "device": "cpu", "torch_threads": 10,
                "agent_class": "Agent", "safety_filter_mode": "production",
                "action_selector_mode": "production", "tracked_spec": {},
                "action_delay_steps": 0,
                "latency": {"mean_ms": 20.0},
                "repeats": [{"repeat": 0, "records": [{
                    "controller_action": 1, "track_state_sha256": "tracks",
                }]}],
            }],
        }

    def test_parity_accepts_source_and_latency_changes(self) -> None:
        baseline = self._report()
        current = deepcopy(baseline)
        current["source_root"] = "optimized"
        current["results"][0]["latency"]["mean_ms"] = 5.0
        checked = _check_baseline(current, baseline, False)
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["records_checked"], 1)

    def test_parity_rejects_action_or_track_changes(self) -> None:
        baseline = self._report()
        for field, replacement in (("controller_action", 2), ("track_state_sha256", "changed")):
            current = deepcopy(baseline)
            current["results"][0]["repeats"][0]["records"][0][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                _check_baseline(current, baseline, False)

    def test_thread_sweep_requires_explicit_reference_override(self) -> None:
        baseline = self._report()
        current = deepcopy(baseline)
        current["results"][0]["torch_threads"] = 1
        with self.assertRaisesRegex(ValueError, "No baseline"):
            _check_baseline(current, baseline, False)
        self.assertTrue(_check_baseline(current, baseline, True)["passed"])


if __name__ == "__main__":
    unittest.main()
