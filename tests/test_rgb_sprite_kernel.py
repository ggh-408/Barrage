"""Compiled sprite operations preserve clipped pixels, ordering and fallback."""
import unittest
from unittest.mock import patch

import numpy as np

from barrage_rl.live_screen import DominantBackgroundSemanticizer
from barrage_rl import rgb_sprite_kernel as kernels


class SpriteKernelTests(unittest.TestCase):
    def test_compiled_matches_numpy_on_layouts_edges_and_duplicates(self):
        if kernels.validate_candidates is None:
            self.skipTest('Optional compiled kernels unavailable')
        detector=DominantBackgroundSemanticizer()
        rng=np.random.default_rng(62583)
        for shape in ((0,0),(1,3),(5,5),(37,53)):
            for density in (0.,.1,.7,1.):
                mask=rng.random(shape)<density
                for view in (mask,mask.T,mask[::-1,::2]):
                    height,width=view.shape
                    centers=rng.integers([-3,-3],[height+4,width+4],size=(80,2))
                    centers[1]=centers[0]
                    for ordered in (centers,centers[::-1],centers[:0]):
                        np.testing.assert_array_equal(
                            detector._validated_candidates(view,ordered),
                            detector._validated_candidates_numpy(view,ordered))
                        a,b=view.copy(),view.copy()
                        detector._erase_sprite_coverage(a,ordered)
                        detector._erase_sprite_coverage_numpy(b,ordered)
                        np.testing.assert_array_equal(a,b)
                        residual=view & (rng.random(view.shape)>.4)
                        unchanged=residual.copy()
                        np.testing.assert_array_equal(
                            detector._select_cover(view,ordered,uncovered_mask=residual),
                            detector._select_cover_numpy(view,ordered,uncovered_mask=residual))
                        np.testing.assert_array_equal(residual,unchanged)

    def test_optional_compiler_fallback(self):
        detector=DominantBackgroundSemanticizer()
        mask=np.ones((8,8),np.bool_)
        centers=np.array([[3,3],[3,4]],np.int64)
        with patch.multiple(kernels,validate_candidates=None,select_cover=None,erase_coverage=None):
            np.testing.assert_array_equal(detector._validated_candidates(mask,centers),
                                          detector._validated_candidates_numpy(mask,centers))
            np.testing.assert_array_equal(detector._select_cover(mask,centers),
                                          detector._select_cover_numpy(mask,centers))
            expected=mask.copy()
            detector._erase_sprite_coverage(mask,centers)
            detector._erase_sprite_coverage_numpy(expected,centers)
            np.testing.assert_array_equal(mask,expected)


if __name__=='__main__':unittest.main()
