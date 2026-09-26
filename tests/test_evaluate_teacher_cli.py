"""Regression tests for privileged-teacher evaluation entry points."""

import contextlib
import io
import unittest

from barrage_rl.evaluate_teacher import (
    _build_cli_parser,
    _cli_episode_count,
    _cli_episode_limit_seconds,
)


class EvaluateTeacherCliTests(unittest.TestCase):
    def test_production_episode_count_is_locked_to_200(self) -> None:
        parser = _build_cli_parser()
        episodes = next(
            action for action in parser._actions if action.dest == "episodes"
        )
        self.assertEqual(episodes.default, 200)
        self.assertEqual(tuple(episodes.choices), (100, 200))
        self.assertEqual(_cli_episode_count(parser.parse_args([])), 200)
        self.assertEqual(parser.parse_args([]).bullets, 300)

        with self.assertRaisesRegex(ValueError, "production teacher"):
            _cli_episode_count(parser.parse_args(["--episodes", "100"]))

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--episodes", "50"])

    def test_capacity_mode_is_locked_to_100_full_episodes(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--capacity-test", "--episodes", "100"])
        self.assertEqual(_cli_episode_count(args), 100)
        self.assertEqual(_cli_episode_limit_seconds(args), 120.0)

        with self.assertRaisesRegex(ValueError, "capacity-test"):
            _cli_episode_count(parser.parse_args(["--capacity-test"]))

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--capacity-test", "--smoke-test"])

    def test_exact_teacher_and_smoke_test_are_explicit(self) -> None:
        args = _build_cli_parser().parse_args([
            "--teacher-kind", "exact",
            "--teacher-horizon-seconds", "1.8",
            "--smoke-test",
        ])

        self.assertEqual(args.teacher_kind, "exact")
        self.assertEqual(args.teacher_horizon_seconds, 1.8)
        self.assertEqual(_cli_episode_count(args), 2)
        self.assertEqual(_cli_episode_limit_seconds(args), 5.0)

    def test_resume_is_explicit(self) -> None:
        args = _build_cli_parser().parse_args(["--resume"])
        self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()
