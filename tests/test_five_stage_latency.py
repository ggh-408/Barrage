import copy
import numpy as np
import pygame
import pytest
from barrage_rl.runtime_core import snapshot_surface_rgb, tracker_prediction_hints
from barrage_rl.image_oracle import PersistentImageTracker, BulletTrack
from barrage_rl.live_screen import DominantBackgroundSemanticizer as Detector


@pytest.mark.parametrize('size',[(0,0),(0,5),(5,0)])
def test_capture_empty_surface(size):
    image=snapshot_surface_rgb(pygame.Surface(size,depth=32))
    assert image.shape==(size[1],size[0],3)
    assert not image.flags.writeable


@pytest.mark.parametrize('depth',[16,24,32])
@pytest.mark.parametrize('subsurface',[False,True])
def test_capture_values_ownership_and_unlock(depth,subsurface):
    rng=np.random.default_rng(88)
    surface=pygame.Surface((41,29),depth=depth)
    pygame.surfarray.blit_array(surface,rng.integers(0,256,(41,29,3),dtype=np.uint8))
    if subsurface:surface=surface.subsurface((3,2,31,23))
    expected=np.ascontiguousarray(pygame.surfarray.array3d(surface).transpose(1,0,2))
    actual=snapshot_surface_rgb(surface)
    assert np.array_equal(actual,expected)
    assert actual.flags.c_contiguous and not actual.flags.writeable
    assert not surface.get_locked()
    surface.fill((0,0,0))
    assert np.array_equal(actual,expected)


@pytest.mark.parametrize('strided',[False,True])
def test_sparse_classifier_exact(strided):
    rng=np.random.default_rng(45)
    image=rng.integers(0,256,(37,41,4),dtype=np.uint8)
    if strided:image=image[::-1,::2]
    for foreground in (np.zeros(image.shape[:2],bool),np.ones(image.shape[:2],bool),rng.random(image.shape[:2])<.05):
        expected=Detector._classify_foreground_numpy(image,foreground)
        actual=Detector._classify_foreground(image,foreground)
        for a,b in zip(actual,expected):
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_prediction_hints_exact_and_no_state_mutation(dtype):
    rng=np.random.default_rng(27)
    t=PersistentImageTracker()
    t.tracks=[BulletTrack(i,rng.normal(size=2).astype(dtype),rng.normal(size=2).astype(dtype),
                         velocity_known=bool(i%3),position_uncertainty=float(i%96)) for i in range(384)]
    t.tracks[0].position[:]=[-0.,0.]
    before=copy.deepcopy(t.tracks)
    for steps in (1,2,5):
        positions=np.stack([a.position+(a.velocity*(t.decision_dt*steps) if a.velocity_known else 0.)
                            for a in t.tracks]).astype(np.float32)
        u=np.asarray([a.position_uncertainty for a in t.tracks],np.float32)
        known=np.asarray([a.velocity_known for a in t.tracks],bool)
        radius=np.where(known,np.clip(u+4.+2.*(steps-1),6.,24.),np.clip(u+8.*steps,24.,64.))
        expected=(positions/t.source_size,radius*(830/max(t.source_size,1.)))
        actual=tracker_prediction_hints(t,(810,830,3),steps)
        for a,b in zip(actual,expected):assert a.tobytes()==b.tobytes()
    for a,b in zip(t.tracks,before):
        assert a.position.tobytes()==b.position.tobytes()
        assert a.velocity.tobytes()==b.velocity.tobytes()
