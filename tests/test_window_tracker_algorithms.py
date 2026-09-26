import numpy as np
import pytest
from barrage_rl.image_oracle import PersistentImageTracker,BulletTrack
from barrage_rl.window_tracker import WindowImageTracker
from barrage_rl.window_tracker_kernel import stack_vectors
from tools.benchmark_tracker_exact import exact,state
from barrage_rl.tracked_policy import TrackedFeatureExtractor

@pytest.mark.parametrize('count',[1,2,17,384])
def test_association_exact_distances_ties_gates_and_layouts(count):
    rng=np.random.default_rng(97)
    a,b=PersistentImageTracker(),WindowImageTracker()
    for t in (a,b):t.tracks=[BulletTrack(i,np.zeros(2,np.float32),np.zeros(2,np.float32),velocity_known=i%2==0) for i in range(count)]
    predictions=rng.uniform(-200,1000,(count,2)).astype(np.float32)
    detections=rng.uniform(-200,1000,(53,2)).astype(np.float32)
    detections[:3]=predictions[0];detections[3]=predictions[0]+[15,0]
    for p,d in ((predictions,detections),(predictions[:,::-1],detections[:,::-1]),
                (predictions.astype(np.float64),detections.astype(np.float64))):
        exact(a._association_arrays(p,d),b._association_arrays(p,d))
    predictions[0]=np.nan
    exact(a._association_arrays(predictions,detections),b._association_arrays(predictions,detections))

def test_custom_occlusion_threshold_keeps_numpy_scalar_promotion():
    a,b=PersistentImageTracker(occlusion_gate=7.123456),WindowImageTracker(occlusion_gate=7.123456)
    for t in (a,b):t.tracks=[BulletTrack(0,np.zeros(2,np.float32),np.zeros(2,np.float32))]
    p=np.zeros((1,2),np.float32)
    d=np.array([[7.123456,0.],[7.123457,0.]],np.float32)
    exact(a._association_arrays(p,d),b._association_arrays(p,d))

@pytest.mark.parametrize('dtype',[None,np.float32,np.float64])
def test_vector_packing_exact_fallback_and_ownership(dtype):
    for values in ([np.array([0.,-0.],np.float32),np.array([np.inf,np.nan],np.float32)],
                   [np.array([1.,2.],np.float64)],[],[[1.,2.],[3.,4.]],
                   [np.arange(4,dtype=np.float32)[::2]]):
        actual=stack_vectors(values,dtype)
        exact(np.asarray(values,dtype=dtype),actual)
        if values and isinstance(values[0],np.ndarray):
            before=actual.copy();values[0][0]=51
            exact(before,actual)

def test_dynamic_tracks_occlusions_empty_frames_and_multisteps():
    rng=np.random.default_rng(13)
    extractors=[TrackedFeatureExtractor(tracker_class=t) for t in (PersistentImageTracker,WindowImageTracker)]
    for e in extractors:e.tracker.history_limit=8
    points=rng.uniform(.1,.9,(300,2)).astype(np.float32)
    velocity=rng.uniform(-.004,.004,(300,2)).astype(np.float32)
    for i in range(70):
        p=points+i*velocity
        if i%9==0:p=p[:200]
        if i%17==0:p=p[:0]
        outputs=[e.step_detections(p,np.array([.5,.5],np.float32),decision_steps=2 if i%13==0 else 1) for e in extractors]
        exact(*outputs);exact(state(extractors[0]),state(extractors[1]))
