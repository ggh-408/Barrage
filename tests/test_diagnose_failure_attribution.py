"""Failure-attribution artifact tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.diagnose_failure_attribution import (
    ATTRIBUTION_CSV_FIELDNAMES,
    _write_attribution_csv,
    diagnose,
)


class DiagnoseFailureAttributionTests(unittest.TestCase):
    def test_reproduction_preserves_analytic_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            (root / "evaluation_config.json").write_text(json.dumps({
                "checkpoint": str(checkpoint),
                "episode_limit_seconds": 120.0,
                "bullet_count": 300,
                "targeted_bullet_probability": 0.10,
                "rendered_rgb": True,
                "causal_action_delay_steps": 0,
                "analytic_shield": True,
                "analytic_clearance_margin": 2.0,
            }), encoding="utf-8")
            (root / "evaluation_episodes.csv").write_text(
                "episode,seed,model_survival_seconds\n0,960000018,1.525\n",
                encoding="utf-8",
            )
            rollout = SimpleNamespace(
                survival_times=[1.525], failure_diagnostics=[], policy_events=[]
            )
            with patch("tools.diagnose_failure_attribution.load_tracked_agent",
                       return_value=(object(), object(), {})) as load, patch(
                "tools.diagnose_failure_attribution.run_parallel_rollout",
                return_value=rollout,
            ):
                report = diagnose(root, root / "report.json", workers=1, device_name="cpu")
        self.assertTrue(load.call_args.kwargs["analytic_shield"])
        self.assertEqual(load.call_args.kwargs["analytic_clearance_margin"], 2.0)
        self.assertEqual(report["reproduction_max_absolute_error_seconds"], 0.0)

    def test_empty_attribution_still_writes_a_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attribution.csv"
            _write_attribution_csv(output, [])
            with output.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames or ()), ATTRIBUTION_CSV_FIELDNAMES)
                self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
