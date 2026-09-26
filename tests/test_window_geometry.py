"""Exact fused geometry, edge cases, and conservative inference eligibility."""
import unittest

import torch

from barrage_rl.action_geometry import ImageGeometryBelief,constant_action_clearance_by_object
from barrage_rl.window_inference import window_clearance
from barrage_rl.window_geometry_kernel import warmup


class WindowGeometryTests(unittest.TestCase):
    def test_exact_random_and_stationary_geometry(self):
        warmup()
        generator=torch.Generator().manual_seed(925123)
        actions=torch.randn(9,2,generator=generator)
        horizons=torch.tensor([0.,.1,.3,.6,1.2])
        for b,n in ((1,384),(3,57),(1,1),(2,0)):
            for trial in range(12):
                positions=torch.randn(b,n,2,generator=generator)*820
                velocities=torch.randn(b,n,2,generator=generator)*240
                masks=torch.rand(b,n,generator=generator)>.15
                if n:
                    velocities[:,0]=actions[0]*240
                    if trial%3==0:positions[:,0]=0
                    if trial%4==0:masks.zero_()
                belief=ImageGeometryBelief(positions,velocities,masks)
                with torch.inference_mode():
                    actual=window_clearance(belief,actions,horizons,bullet_speed=240.,collision_radius=11.5)
                    if actual is None:self.skipTest('Optional compiler unavailable')
                    expected=constant_action_clearance_by_object(belief,actions,horizons,bullet_speed=240.,collision_radius=11.5)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)

    def test_gradients_and_other_dtypes_defer(self):
        for dtype in (torch.float32,torch.float64):
            belief=ImageGeometryBelief(torch.ones(1,3,2,dtype=dtype),torch.ones(1,3,2,dtype=dtype),torch.ones(1,3,dtype=torch.bool))
            actions=torch.ones(9,2,dtype=dtype);horizons=torch.ones(4,dtype=dtype)
            self.assertIsNone(window_clearance(belief,actions,horizons,bullet_speed=240.,collision_radius=11.5))
            if dtype==torch.float64:
                with torch.inference_mode():
                    self.assertIsNone(window_clearance(belief,actions,horizons,bullet_speed=240.,collision_radius=11.5))


if __name__=='__main__':unittest.main()
