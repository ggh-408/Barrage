"""Check route-row reuse without relying on incidental numerical kernel results."""
import copy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'diagnostics/planner_readiness_20260921'))
from tools.reuse_retained_assessment import install
from tools.benchmark_tracker_exact import exact


@dataclass
class Selection:
    actions: object
    immediate_risk: object
    all_unsafe: object
    counter_values: object
    raw_actions: object


class RetainedReuseTests(unittest.TestCase):
    def test_same_route_ranking_memory_and_counters_with_fewer_assessed_rows(self):
        spec = importlib.util.spec_from_file_location('reviewed_planner',
            ROOT / 'diagnostics/planner_readiness_20260921/candidate_parallel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Guard(module.ContinuationMixin):
            def __init__(self, prior, certified):
                self.recovery_config = SimpleNamespace(preserve_plans=True,
                    interval_consensus=True, search_depth=15, trigger_safe_actions=2,
                    wall_reserve_pixels=48., compiled_beam=True, beam_width=8, wall_cost_weight=1.,
                    velocity_error_floor=2.)
                self.half_size = np.array([9., 9.], np.float32)
                self.centroid_bias = self.bullet_bias = np.zeros(2, np.float32)
                self.table = self.integral = np.zeros((1, 1))
                self._plans = ({0: dict(path=np.full(14, 2, np.int64),
                    expected_plane=np.array([410., 410.], np.float32), remaining=14)} if prior else {})
                self._context_indices = None
                self._context_steps = 1
                self.commit_guard = self
                self.counters = dict.fromkeys(('plan_invalidations', 'decisions', 'model_releases',
                    'horizon_searches', 'plan_reuses', 'overrides', 'interval_conflicts',
                    'searches', 'geometric_searches', 'wall_searches'), 0)
                self.elapsed_seconds = 0.
                self.rows = 0
                self.certified = certified
                self.traces = []
                self._decision_observer = lambda key, row: self.traces.append((key, row))
            def hazards(self, objects, masks, globals_):
                return np.zeros((1, 9), bool), np.zeros((1, 9), bool)
            def _assess(self, plane, half, bullets, velocity, error, paths, lengths):
                self.rows += len(paths)
                value = np.zeros((len(paths), 7), np.float64)
                value[:, :3] = [4., .5, 1.]
                value[:, 3] = 1.
                value[:, 4] = 100.
                value[:, 5] = float(self.certified)
                value[:, 6] = paths.sum(axis=1) / 100.
                return value
        def beam(*args):
            return np.zeros(9), np.repeat(np.arange(9, dtype=np.int64)[:, None], 15, axis=1)
        for prior in (False, True):
            for certified in (False, True):
                old, new = Guard(prior, certified), Guard(prior, certified)
                install(new)
                selection = Selection(torch.tensor([0]), torch.zeros((1, 9)),
                    torch.tensor([False]), torch.zeros(5), torch.tensor([0]))
                objects = np.zeros((1, 384, 16), np.float32)
                masks = np.zeros((1, 384), bool)
                globals_ = np.zeros((1, 16), np.float32)
                globals_[0, :2] = .5
                with patch('readiness_kernel_parallel.beam_paths', beam):
                    a = old.apply(copy.deepcopy(selection), objects, masks, globals_)
                    b = new.apply(copy.deepcopy(selection), objects, masks, globals_)
                self.assertTrue(torch.equal(a.actions, b.actions))
                self.assertTrue(torch.equal(a.counter_values, b.counter_values))
                exact(old._plans, new._plans)
                exact(old.traces, new.traces)
                exact(old.counters, new.counters)
                self.assertEqual(old.rows - new.rows, 9 if prior and not certified else 0)


if __name__ == '__main__':
    unittest.main()
