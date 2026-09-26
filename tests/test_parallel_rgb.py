import numpy as np
import pytest
from barrage_rl.rgb_capture_kernel import copy_rgb_serial,copy_rgb_parallel,copy_rgb
from barrage_rl.foreground_kernel import fused_foreground_serial,fused_foreground_parallel
from barrage_rl.rgb_parallel import rgb_workers,dispatch
from numba import get_num_threads,set_num_threads

@pytest.mark.parametrize('workers',[2,4,9])
def test_pixels_strides_thresholds_and_thread_mask(workers):
    rng=np.random.default_rng(72)
    base=rng.integers(0,256,(37,49,4),dtype=np.uint8)
    previous=get_num_threads()
    for image in (base[:,:,:3],base[::-1,::2,2::-1],base[:0,:,:3]):
        with rgb_workers(workers):
            result=copy_rgb(image)
            assert result.tobytes()==copy_rgb_serial(image).tobytes()
            for lower,upper in (([-1]*3,[256]*3),([0,120,254],[1,121,255])):
                lo,hi=np.array(lower),np.array(upper)
                expected=fused_foreground_serial(image,lo,hi)
                actual=dispatch(fused_foreground_serial,fused_foreground_parallel,image,lo,hi)
                assert expected.tobytes()==actual.tobytes()
        assert get_num_threads()==previous

def test_thread_mask_restored_on_failure():
    previous=get_num_threads()
    def fail(*args):raise ValueError('probe')
    with rgb_workers(2),pytest.raises(ValueError):dispatch(fail,fail)
    assert get_num_threads()==previous

@pytest.mark.parametrize('depth',[16,24,32])
def test_parallel_snapshot_is_owned_readonly_and_unlocked(depth):
    import pygame
    from barrage_rl.runtime_core import snapshot_surface_rgb
    surface=pygame.Surface((57,43),depth=depth).subsurface((3,2,41,37))
    surface.fill((31,97,203))
    expected=pygame.surfarray.array3d(surface).transpose(1,0,2).copy()
    with rgb_workers(4):image=snapshot_surface_rgb(surface)
    assert image.tobytes()==expected.tobytes()
    assert image.flags.c_contiguous and not image.flags.writeable
    assert not surface.get_locked()
    surface.fill((0,0,0))
    assert image.tobytes()==expected.tobytes()
