import unittest
import importlib.util
import numpy as np
import pygame
import torch
from barrage_rl.action_selector import ActionSelection

from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig
from tools.pixel_recovery_planner import recovery_action
from barrage_rl.runtime_core import colliding_bullet_indices, advance_plane, ACTION_VECTORS


class PixelGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = PixelGuard(PixelGuardConfig(use_intervals=False))

    def test_template_lookup_matches_real_pixel_kernel_for_subpixel_positions(self):
        rng = np.random.default_rng(904)
        guard = self.guard
        for _ in range(6):
            plane = rng.uniform(100, 700, 2).astype(np.float32)
            bullets = plane + rng.uniform(-20, 20, (1000, 2)).astype(np.float32)
            expected = np.zeros(len(bullets), bool)
            expected[colliding_bullet_indices(plane, bullets, guard.plane, guard.bullet,
                                              guard.plane_mask, guard.bullet_mask)] = True
            center = guard.rounded(bullets) - guard.rounded(plane)
            np.testing.assert_array_equal(guard.rectangle_hits(center, center)>0, expected)

    def test_centroid_calibration_recovers_sprite_center(self):
        np.testing.assert_allclose(self.guard.centroid_bias, [-0.5, 1.4468085], atol=1e-5)

    def test_interval_query_matches_exhaustive_integer_offsets(self):
        guard = self.guard
        rng = np.random.default_rng(172)
        for _ in range(100):
            lower = rng.integers(-40, 35, 2)
            upper = lower + rng.integers(0, 8, 2)
            total = 0
            for y in range(lower[1], upper[1]+1):
                for x in range(lower[0], upper[0]+1):
                    if -32 <= x <= 32 and -32 <= y <= 32:
                        total += guard.table[y+32, x+32]
            self.assertEqual(guard.rectangle_hits(lower, upper), total)

    def test_horizon_is_one_control_period_and_no_tracks_is_safe(self):
        self.assertEqual(PixelGuardConfig().physics_steps, 4)
        objects=np.zeros((2,384,16),np.float32)
        nominal,possible=self.guard.hazards(objects,np.zeros((2,384),bool),np.zeros((2,16),np.float32))
        self.assertFalse(nominal.any())
        self.assertFalse(possible.any())

    def test_nominal_prediction_matches_kernel_after_centroid_correction(self):
        guard = self.guard
        plane = np.asarray([410,410],np.float32)
        bullets = np.asarray([[430,420],[400,388],[390,417]],np.float32)
        velocities=np.asarray([[-240,0],[0,240],[240,0]],np.float32)
        observed=plane+guard.centroid_bias
        objects=np.zeros((1,3,16),np.float32)
        objects[0,:,:2]=(bullets-observed)/820
        objects[0,:,2:4]=velocities/240
        objects[0,:,8]=objects[0,:,10]=objects[0,:,15]=1
        globals_=np.zeros((1,16),np.float32);globals_[0,:2]=observed/820
        nominal,_=guard.hazards(objects,np.ones((1,3),bool),globals_)
        truth=[]
        for action in ACTION_VECTORS:
            p=plane.copy();b=bullets.copy();hit=False
            for _ in range(4):
                p,_=advance_plane(p,action,240,1/120,guard.half_size,820,820)
                b+=velocities/120
                hit |= bool(len(colliding_bullet_indices(p,b,guard.plane,guard.bullet,guard.plane_mask,guard.bullet_mask)))
            truth.append(hit)
        np.testing.assert_array_equal(nominal[0],truth)

    def test_interval_envelope_contains_nominal_collisions(self):
        guard=PixelGuard(PixelGuardConfig())
        rng=np.random.default_rng(119)
        o=np.zeros((8,30,16),np.float32)
        o[:,:,:2]=rng.uniform(-30,30,(8,30,2))/820
        o[:,:,2:4]=rng.uniform(-1,1,(8,30,2))
        o[:,:,8]=o[:,:,10]=o[:,:,15]=1
        g=np.zeros((8,16),np.float32);g[:,:2]=.5
        nominal,possible=guard.hazards(o,np.ones((8,30),bool),g)
        self.assertTrue(np.all(~nominal | possible))

    def test_unknown_tracks_do_not_create_hard_overrides(self):
        o=np.zeros((1,1,16),np.float32);o[:,:,8]=1
        g=np.zeros((1,16),np.float32);g[:,:2]=.5
        n,p=self.guard.hazards(o,np.ones((1,1),bool),g)
        self.assertFalse(n.any());self.assertFalse(p.any())

    def test_imminent_pixel_collision_can_escape_a_longer_learned_veto(self):
        guard=PixelGuard(PixelGuardConfig(allow_imminent_escape=True))
        guard.safety_threshold=.18
        nominal=np.array([[True,False]])
        guard.hazards=lambda *args:(nominal,nominal)
        selection=ActionSelection(
            actions=torch.tensor([0]),scores=torch.tensor([[1.,-torch.inf]]),
            raw_actions=torch.tensor([0]),immediate_risk=torch.tensor([[.07,.197]]),
            all_unsafe=torch.tensor([False]),analytic_clearance=None,
            learned_actions=torch.tensor([0]),collision_risk=torch.zeros(1,4,2),
            teacher_cost=torch.zeros(1,2),counter_values=torch.zeros(5,dtype=torch.int64))
        chosen=guard.apply(selection,None,None,None)
        self.assertEqual(chosen.actions.item(),1)
        self.assertEqual(guard.counters['imminent_escape_overrides'],1)
        guard.hazards=lambda *args:(np.zeros_like(nominal),nominal)
        self.assertEqual(guard.apply(selection,None,None,None).actions.item(),0)

    def test_uncertainty_conflict_keeps_available_routes(self):
        guard=PixelGuard(PixelGuardConfig(resolve_interval_conflicts=True))
        guard.safety_threshold=.18
        selection=ActionSelection(
            actions=torch.tensor([0]),scores=torch.full((1,3),-torch.inf),
            raw_actions=torch.tensor([0]),immediate_risk=torch.tensor([[.6,.8,.7]]),
            all_unsafe=torch.tensor([True]),analytic_clearance=None,
            learned_actions=torch.tensor([0]),collision_risk=torch.zeros(1,4,3),
            teacher_cost=torch.zeros(1,3),counter_values=torch.zeros(5,dtype=torch.int64))
        guard.hazards=lambda *args:(np.array([[False,False,False]]),np.array([[True,False,False]]))
        self.assertEqual(guard.apply(selection,None,None,None).actions.item(),2)
        guard.hazards=lambda *args:(np.array([[True,False,True]]),np.ones((1,3),bool))
        self.assertEqual(guard.apply(selection,None,None,None).actions.item(),1)
        guard.hazards=lambda *args:(np.ones((1,3),bool),np.ones((1,3),bool))
        self.assertEqual(guard.apply(selection,None,None,None).actions.item(),0)

    def test_recovery_search_escapes_a_pixel_collision(self):
        g=np.zeros(16,np.float32)
        g[:2]=(np.array([410,410])+self.guard.centroid_bias)/820
        o=np.zeros((1,16),np.float32)
        o[0,:2]=(np.array([430,410])-g[:2]*820)/820
        o[0,2]=-1
        o[0,8]=o[0,10]=o[0,15]=1
        mask=np.ones(1,bool)
        chosen=recovery_action(self.guard,o,mask,g,np.zeros(9),depth=6,beam_width=4)
        nominal,_=self.guard.hazards(o[None],mask[None],g[None])
        self.assertFalse(nominal[0,chosen])
        self.assertEqual(chosen,recovery_action(self.guard,o,mask,g,np.zeros(9),depth=6,beam_width=4))

    @unittest.skipUnless(importlib.util.find_spec('numba') is not None,
        'optional requirements.txt dependency')
    def test_compiled_search_step_matches_numpy_collision_and_interval_mass(self):
        from tools.pixel_search_kernel import search_step
        rng=np.random.default_rng(913)
        guard=self.guard
        for _ in range(10):
            positions=rng.uniform(390,430,(80,2)).astype(np.float32)
            move=ACTION_VECTORS[rng.integers(0,9,80)]*2
            bullets=rng.uniform(380,440,(20,2)).astype(np.float32)
            error=rng.uniform(.5,3,20).astype(np.float32)
            p,hit,mass=search_step(positions,move,guard.half_size,bullets,error,guard.table,guard.integral)
            expected=np.clip(positions+move,guard.half_size,820-guard.half_size)
            center=guard.rounded(bullets)[None]-guard.rounded(expected)[:,None]
            lo=guard.rounded(bullets-error[:,None])[None]-guard.rounded(expected+.5-1e-5)[:,None]
            hi=guard.rounded(bullets+error[:,None]-1e-5)[None]-guard.rounded(expected-.5)[:,None]
            np.testing.assert_array_equal(p,expected)
            np.testing.assert_array_equal(hit,(guard.rectangle_hits(center,center)>0).any(axis=1))
            np.testing.assert_array_equal(mass,guard.rectangle_hits(lo,hi)/np.prod(hi-lo+1,axis=-1))


if __name__ == '__main__':
    unittest.main()
