"""CLI invariants for tracked-policy production evaluation."""

import contextlib
import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from barrage_rl.evaluate_tracked_policy import (
    _RolloutAccumulator,
    _build_cli_parser,
    _write_evaluation_progress,
    checkpoint_action_delay_steps,
    evaluate_tracked_checkpoint,
)
from barrage_rl.parallel_evaluation import ParallelRolloutProgress, ParallelRolloutResult


class EvaluateTrackedPolicyCliTests(unittest.TestCase):
    def test_supplemental_count_requires_explicit_full_length_seed_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 200"):
            evaluate_tracked_checkpoint("unused.pt", episodes=300)
        with self.assertRaisesRegex(ValueError, "explicit seeds"):
            evaluate_tracked_checkpoint("unused.pt", episodes=300, supplemental_test=True)
        with self.assertRaisesRegex(ValueError, "one seed per episode"):
            evaluate_tracked_checkpoint("unused.pt", episodes=300, supplemental_test=True,
                                        episode_seeds=[1])
        with patch("barrage_rl.evaluate_tracked_policy.load_tracked_agent",
                   side_effect=RuntimeError("validation passed")):
            with self.assertRaisesRegex(RuntimeError, "validation passed"):
                evaluate_tracked_checkpoint("unused.pt", episodes=300,
                                            supplemental_test=True,
                                            episode_seeds=list(range(300)))

    def test_production_episode_count_is_locked_to_200(self) -> None:
        parser = _build_cli_parser()
        episodes = next(
            action for action in parser._actions if action.dest == "episodes"
        )
        self.assertEqual(tuple(episodes.choices), (200,))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["model.pt", "--episodes", "50"])

    def test_bullet_count_can_be_swept_without_changing_episode_count(self) -> None:
        args = _build_cli_parser().parse_args(["model.pt", "--bullets", "350"])
        self.assertEqual(args.bullets, 350)
        self.assertEqual(args.episodes, 200)

    def test_evaluation_commits_progress_in_ten_episode_batches(self) -> None:
        args = _build_cli_parser().parse_args(["model.pt"])
        self.assertEqual(args.bullets, 300)
        self.assertEqual(args.evaluation_batch_size, 10)
        self.assertIsNone(args.causal_action_delay_steps)
        self.assertFalse(args.analytic_shield)

    def test_production_requires_current_300_bullet_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 300 bullets"):
            evaluate_tracked_checkpoint("unused.pt", bullet_count=250)
        with patch("barrage_rl.evaluate_tracked_policy.load_tracked_agent",
                   side_effect=RuntimeError("validation passed")):
            with self.assertRaisesRegex(RuntimeError, "validation passed"):
                evaluate_tracked_checkpoint("unused.pt", bullet_count=300)

    def test_geometry_constraint_defaults_off(self) -> None:
        args = _build_cli_parser().parse_args([
            "model.pt", "--no-analytic-shield"
        ])
        self.assertFalse(args.analytic_shield)

    def test_smoke_cli_can_override_action_delay_for_a_paired_diagnostic(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args([
            "model.pt", "--smoke-test", "--causal-action-delay-steps", "0"
        ])
        self.assertEqual(args.causal_action_delay_steps, 0)

    def test_checkpoint_timing_defaults_legacy_to_zero_and_new_to_saved_value(self) -> None:
        self.assertEqual(checkpoint_action_delay_steps({"config": {}}), 0)
        self.assertEqual(
            checkpoint_action_delay_steps({
                "config": {"evaluation_causal_action_delay_steps": 1}
            }),
            1,
        )

    def test_progress_batch_is_atomically_serialized(self) -> None:
        progress = ParallelRolloutProgress(
            completed_indices=np.asarray([1, 0], dtype=np.int64),
            survival_times=np.asarray([80.0, 120.0], dtype=np.float64),
            termination_reasons=["collision", "time_limit"],
            minimum_wall_distances=np.asarray([41.0, 62.0], dtype=np.float64),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            _write_evaluation_progress(
                output,
                progress,
                all_episode_seeds=(700, 701, 702),
                episodes=3,
                episode_limit_seconds=120.0,
            )
            with (output / "evaluation_episodes.partial.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            summary = json.loads(
                (output / "evaluation_progress.json").read_text(encoding="utf-8")
            )
        self.assertEqual([row["episode"] for row in rows], ["0", "1"])
        self.assertEqual([row["seed"] for row in rows], ["700", "701"])
        self.assertEqual(summary["completed_episodes"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertFalse(summary["complete"])

    def test_rollout_batches_accumulate_without_changing_episode_order(self) -> None:
        def part(offset: int) -> ParallelRolloutResult:
            return ParallelRolloutResult(
                survival_times=np.asarray([offset + 0.5, offset + 1.5]),
                termination_reasons=["collision", "time_limit"],
                bullet_sizes=np.asarray([5, 5]),
                bullet_speeds=np.asarray([240.0, 240.0]),
                reset_modes=[f"reset-{offset}", f"reset-{offset + 1}"],
                minimum_wall_distances=np.asarray([30.0 + offset, 40.0 + offset]),
                wall_steps=offset + 1,
                model_steps=offset + 10,
                action_histogram=np.asarray([offset, offset + 1], dtype=np.int64),
            )

        accumulator = _RolloutAccumulator(4)
        first_progress = accumulator.append(part(0))
        final_progress = accumulator.append(part(2))
        result = accumulator.result()
        np.testing.assert_array_equal(first_progress.completed_indices, [0, 1])
        np.testing.assert_array_equal(final_progress.completed_indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.survival_times, [0.5, 1.5, 2.5, 3.5])
        self.assertEqual(
            result.termination_reasons,
            ["collision", "time_limit", "collision", "time_limit"],
        )
        self.assertEqual(result.reset_modes, ["reset-0", "reset-1", "reset-2", "reset-3"])
        self.assertEqual(result.wall_steps, 4)
        self.assertEqual(result.model_steps, 22)
        np.testing.assert_array_equal(result.action_histogram, [2, 4])


if __name__ == "__main__":
    unittest.main()
