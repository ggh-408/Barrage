import unittest
import numpy as np
from tools.pixel_guard_candidate import PixelGuardConfig
from tools.pixel_guard_refined import RefinedPixelGuard
from barrage_rl.runtime_core import ACTION_VECTORS


class RefinedGuardTests(unittest.TestCase):
    def test_unknown_velocity_envelope_covers_sampled_directions(self):
        guard=RefinedPixelGuard(PixelGuardConfig())
        plane=np.array([410.,410.],np.float32)
        g=np.zeros((1,16),np.float32);g[0,:2]=(plane+guard.centroid_bias)/820
        o=np.zeros((1,1,16),np.float32)
        bullet=np.array([426.,412.],np.float32)
        o[0,0,:2]=(bullet+guard.bullet_bias-g[0,:2]*820)/820
        o[0,0,8]=.15
        _,possible=guard.hazards(o,np.ones((1,1),bool),g)
        self.assertTrue(possible.any())
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            for step in range(1,5):
                positions=plane+ACTION_VECTORS*2*step
                b=bullet+np.array([np.cos(angle),np.sin(angle)])*2*step
                centers=guard.rounded(b)-guard.rounded(positions)
                hits=guard.rectangle_hits(centers,centers)>0
                self.assertTrue(np.all(~hits | possible[0]))

    def test_missing_tracks_do_not_create_new_unknown_envelopes(self):
        guard=RefinedPixelGuard(PixelGuardConfig())
        o=np.zeros((1,1,16),np.float32);o[0,0,8]=.15;o[0,0,9]=1
        g=np.zeros((1,16),np.float32);g[0,:2]=.5
        _,possible=guard.hazards(o,np.ones((1,1),bool),g)
        self.assertFalse(possible.any())


if __name__=='__main__':unittest.main()
