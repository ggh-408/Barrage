"""Tests for the 300-bullet exact-teacher batch utility."""

import csv
import io
import unittest

from tools.batch_test_exact_teacher_300 import (
    BULLET_COUNTS,
    CSV_FIELDS,
    _build_parser,
    episode_csv,
    generate_unique_seeds,
)


class ExactTeacher300BatchTests(unittest.TestCase):
    def test_defaults_use_ten_workers(self) -> None:
        args = _build_parser().parse_args([])
        self.assertEqual(args.episodes_per_stage, 300)
        self.assertEqual(args.workers, 10)
        self.assertEqual(args.progress_every, 10)
        self.assertIsNone(args.output_dir)
        self.assertEqual(BULLET_COUNTS, (300, 350))

    def test_random_seeds_are_unique_and_reproducible(self) -> None:
        first = generate_unique_seeds(1000, 12345)
        second = generate_unique_seeds(1000, 12345)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(0 <= seed < 2**31 - 1 for seed in first))

    def test_stages_share_one_seed_set(self) -> None:
        seeds = generate_unique_seeds(300, 77)
        stages = [list(seeds) for _ in BULLET_COUNTS]
        self.assertEqual(stages[0], stages[1])
        self.assertEqual(len(stages), len(BULLET_COUNTS))
        self.assertEqual(len(set(seeds)), 300)

    def test_csv_contains_death_diagnostics(self) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update({
            "stage": 1,
            "bullet_count": 300,
            "global_episode": 0,
            "episode": 0,
            "seed": 42,
            "passed": 0,
            "termination_reason": "collision",
            "death_plane_center_x": 123.5,
            "death_plane_center_y": 456.5,
            "collision_bullet_index": 7,
        })
        parsed = list(csv.DictReader(io.StringIO(episode_csv([row]))))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(tuple(parsed[0]), CSV_FIELDS)
        self.assertEqual(parsed[0]["death_plane_center_x"], "123.5")
        self.assertEqual(parsed[0]["collision_bullet_index"], "7")


if __name__ == "__main__":
    unittest.main()
