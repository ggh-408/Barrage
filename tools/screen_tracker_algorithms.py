"""Pair exact tracker kernels against current NumPy on recorded image states."""
import json,pickle,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from barrage_rl.window_tracker_kernel import association_scan,stack_vectors
from barrage_rl.tracked_policy import TrackedFeatureExtractor
from tools.benchmark_tracker_exact import exact

def reference(p,d,g,o):
    dx=p[:,None,0]-d[None,:,0];dy=p[:,None,1]-d[None,:,1]
    distance=dx*dx+dy*dy
    return distance,distance<=g[:,None],np.argmin(distance,axis=1),np.argmin(distance,axis=0),np.count_nonzero(distance<=o,axis=0)

def benchmark(a,b,inputs):
    times=[[],[]]
    for x in inputs:exact(a(*x),b(*x))
    for repeat in range(12):
        for x in inputs:
            for j in ((0,1) if repeat%2 else (1,0)):
                start=time.perf_counter();(a,b)[j](*x);times[j].append((time.perf_counter()-start)*1000)
    return dict(before_ms=float(np.median(times[0])),after_ms=float(np.median(times[1])),cases=len(inputs),exact=True)

if __name__=='__main__':
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:records=pickle.load(f)
    e=TrackedFeatureExtractor();e.tracker.history_limit=8;cases=[];vectors=[]
    for i,(d,kw,_) in enumerate(records):
        t=e.tracker
        if i%60==1 and len(t.tracks) and len(d.bullet_positions):
            cases.append((t.predict_positions(kw.get('decision_steps',1)),d.bullet_positions*t.source_size,
                np.square(np.array([15. if x.velocity_known else 24. for x in t.tracks],np.float32)),t.occlusion_gate*t.occlusion_gate))
            vectors.append(([x.position for x in t.tracks],))
        e.step_detections(d.bullet_positions,d.plane_position,**kw)
    r=dict(association=benchmark(reference,association_scan,cases),
        vector_packing=benchmark(lambda x:np.asarray(x,dtype=np.float32),lambda x:stack_vectors(x,np.float32),vectors))
    (ROOT/'diagnostics/tracker_algorithm_20260925/screening.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r,indent=2),flush=True)
