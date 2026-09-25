"""Exact candidate screening on image-derived inputs; no deployment mutation."""
import copy
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'diagnostics/five_stage_20260925'
from barrage_rl.foreground_kernel import njit


@njit(cache=False,fastmath=False)
def copy_rgb(image):
    result=np.empty((image.shape[0],image.shape[1],3),np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(3):result[y,x,c]=image[y,x,c]
    return result


@njit(cache=False,fastmath=False)
def classify_rgb(image,lower,upper):
    bullets=np.empty(image.shape[:2],np.bool_)
    planes=np.empty(image.shape[:2],np.bool_)
    count=0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r,g,b=np.int64(image[y,x,0]),np.int64(image[y,x,1]),np.int64(image[y,x,2])
            visible=(r<=lower[0] or r>=upper[0] or g<=lower[1] or g>=upper[1] or b<=lower[2] or b>=upper[2])
            bullet=visible and abs(r-g)<=16 and abs(r-b)<=16
            plane=visible and not bullet
            bullets[y,x]=bullet;planes[y,x]=plane
            if plane:count+=1
    coords=np.empty((count,2),np.int64)
    i=0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if planes[y,x]:coords[i,0]=y;coords[i,1]=x;i+=1
    return bullets,coords


@njit(cache=False,fastmath=False)
def classify_sparse(image,indices):
    bullets=np.zeros(image.shape[:2],np.bool_)
    planes=np.empty((len(indices),2),np.int64)
    count=0
    for i in indices:
        y,x=i//image.shape[1],i%image.shape[1]
        r,g,b=np.int64(image[y,x,0]),np.int64(image[y,x,1]),np.int64(image[y,x,2])
        if abs(r-g)<=16 and abs(r-b)<=16:bullets[y,x]=True
        else:planes[count,0]=y;planes[count,1]=x;count+=1
    return bullets,planes[:count]


def hint_candidate(tracker,shape,decision_steps=1):
    from barrage_rl.runtime_core import tracker_prediction_hints
    tracks=tracker.tracks
    if not tracks or not all(t.position.dtype==np.float32 and t.velocity.dtype==np.float32 for t in tracks):
        return tracker_prediction_hints(tracker,shape,decision_steps)
    elapsed=tracker.decision_dt*max(1,int(decision_steps))
    known=np.asarray([t.velocity_known for t in tracks],np.bool_)
    positions=np.asarray([t.position for t in tracks],np.float32)
    velocities=np.asarray([t.velocity for t in tracks],np.float32)
    displacement=np.zeros_like(positions)
    displacement[known]=velocities[known]*elapsed
    positions=positions+displacement
    uncertainty=np.asarray([t.position_uncertainty for t in tracks],np.float32)
    steps=max(1,int(decision_steps))
    radius=np.where(known,np.clip(uncertainty+4.+2.*(steps-1),6.,24.),np.clip(uncertainty+8.*steps,24.,64.))
    return positions/tracker.source_size,radius*(max(shape[:2])/max(tracker.source_size,1.))


def timing(fn,inputs,repeats=8):
    values=[]
    for _ in range(repeats):
        for x in inputs:
            start=time.perf_counter();fn(x);values.append((time.perf_counter()-start)*1000)
    return dict(mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)))


def main():
    import pygame
    from barrage_rl.live_screen import DominantBackgroundSemanticizer as D
    from barrage_rl.runtime_core import snapshot_surface_rgb,tracker_prediction_hints
    from tools.validate_five_stage_changes import before_function
    snapshot_surface_rgb=before_function('snapshot_surface_rgb')
    tracker_prediction_hints=before_function('tracker_prediction_hints')
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    from tools.benchmark_tracker_exact import exact
    with (ROOT/'diagnostics/rgb_blocks_20260925_112807_783291/rgb_samples.pkl').open('rb') as f:samples=pickle.load(f)
    images=[s['image'] for s in samples]
    triples=[]
    for image in images:
        h,w=image.shape[:2];background=np.median(image[::max(1,h//32),::max(1,w//32),:3].reshape(-1,3),axis=0).astype(np.int16)
        lower=np.floor(background-24).astype(np.int64);upper=np.ceil(background+24).astype(np.int64)
        triples.append((image,lower,upper))
    from barrage_rl.foreground_kernel import fused_foreground
    old=lambda x:D._classify_foreground_numpy(x[0],fused_foreground(*x))
    new=lambda x:classify_rgb(*x)
    for x in triples:exact(old(x),new(x))
    results={'rgb_classification':dict(exact=True,before=timing(old,triples),after=timing(new,triples))}
    sparse=lambda x:classify_sparse(x[0],np.flatnonzero(fused_foreground(*x)))
    for x in triples:exact(old(x),sparse(x))
    results['rgb_sparse_classification']=dict(exact=True,before=timing(old,triples),after=timing(sparse,triples))
    surface=pygame.Surface((820,820),depth=32)
    def capture(surface):
        image=np.asarray(surface.get_view('3')).transpose(1,0,2)
        result=copy_rgb(image);result.setflags(write=False)
        return result
    for image in images:
        pygame.surfarray.blit_array(surface,image.transpose(1,0,2))
        exact(snapshot_surface_rgb(surface),capture(surface));assert not surface.get_locked()
    results['rgb_capture']=dict(exact=True,before=timing(snapshot_surface_rgb,[surface],400),after=timing(capture,[surface],400))
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:records=pickle.load(f)
    extractor=TrackedFeatureExtractor();states=[]
    for i,(d,kw,_) in enumerate(records):
        extractor.step_detections(d.bullet_positions,d.plane_position,**kw)
        if i%60==0:states.append(copy.deepcopy(extractor.tracker))
    for t in states:
        for steps in (1,2,4):exact(tracker_prediction_hints(t,(820,820,3),steps),hint_candidate(t,(820,820,3),steps))
    results['prediction_hints']=dict(exact=True,before=timing(lambda t:tracker_prediction_hints(t,(820,820,3)),states),after=timing(lambda t:hint_candidate(t,(820,820,3)),states))
    (OUT/'screening.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
