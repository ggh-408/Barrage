"""CLI invariants for the image-only ceiling evaluation."""

import contextlib
import io
import unittest

from barrage_rl.evaluate_image_oracle import _build_cli_parser


class EvaluateImageOracleCliTests(unittest.TestCase):
    def test_production_episode_count_is_locked_to_200(self) -> None:
        parser = _build_cli_parser()
        self.assertEqual(parser.parse_args([]).bullets, 300)
        episodes = next(
            action for action in parser._actions if action.dest == "episodes"
        )
        self.assertEqual(episodes.default, 200)
        self.assertEqual(tuple(episodes.choices), (200,))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--episodes", "50"])

    def test_smoke_test_is_explicit(self) -> None:
        args = _build_cli_parser().parse_args(["--smoke-test", "--resume"])
        self.assertTrue(args.smoke_test)
        self.assertTrue(args.resume)
        self.assertEqual(args.planner_kind, "recovery")


if __name__ == "__main__":
    unittest.main()
