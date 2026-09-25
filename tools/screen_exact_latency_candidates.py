"""Screen bitwise-equivalent candidates without changing deployment code."""
import copy
import inspect
import json
import pickle
import sys
import textwrap
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from barrage_rl.rgb_capture_kernel import copy_rgb
from barrage_rl.foreground_kernel import fused_foreground, njit
from tools.benchmark_tracker_exact import exact, state

@njit(cache=False,fastmath=False)
def copy_unrolled(image):
    result=np.empty((image.shape[0],image.shape[1],3),np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y,x,0]=image[y,x,0]
            result[y,x,1]=image[y,x,1]
            result[y,x,2]=image[y,x,2]
    return result

@njit(cache=False,fastmath=False)
def foreground_unrolled(image,lower,upper):
    result=np.empty(image.shape[:2],np.bool_)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r,g,b=image[y,x,0],image[y,x,1],image[y,x,2]
            result[y,x]=(r<=lower[0] or r>=upper[0] or g<=lower[1] or g>=upper[1] or b<=lower[2] or b>=upper[2])
    return result

def paired(functions,inputs,repeats=12):
    timings=[[],[]]
    for args in inputs:exact(functions[0](*args),functions[1](*args))
    for i in range(repeats):
        for args in inputs:
            for j in ((0,1) if i%2 else (1,0)):
                t=time.perf_counter();functions[j](*args);timings[j].append((time.perf_counter()-t)*1000)
    return dict(before_ms=float(np.median(timings[0])),candidate_ms=float(np.median(timings[1])),
                samples_per_variant=len(timings[0]),bitwise_equal=True)

def main():
    import pygame
    from barrage_rl.image_oracle import PersistentImageTracker as T
    from barrage_rl.tracked_policy import TrackedFeatureExtractor as E
    from barrage_rl.runtime_core import tracker_prediction_hints
    out=ROOT/'diagnostics/exact_latency_screen_20260925'
    out.mkdir(exist_ok=False)
    with (ROOT/'diagnostics/rgb_blocks_20260925_112807_783291/rgb_samples.pkl').open('rb') as f:samples=pickle.load(f)
    capture=[];surfaces=[];foreground=[]
    for sample in samples:
        image=sample['image'];h,w=image.shape[:2]
        surface=pygame.Surface((w,h),depth=32)
        pygame.surfarray.blit_array(surface,image.transpose(1,0,2));surfaces.append(surface)
        capture.append((np.asarray(surface.get_view('3')).transpose(1,0,2),))
        bg=np.median(image[::max(1,h//32),::max(1,w//32),:3].reshape(-1,3),axis=0).astype(np.int16)
        foreground.append((image,np.floor(bg-28).astype(np.int64),np.ceil(bg+28).astype(np.int64)))
    result=dict(scope='Paired recorded image inputs; candidates are process-local and not deployed.',
        capture_unrolled=paired((copy_rgb,copy_unrolled),capture),
        foreground_unrolled=paired((fused_foreground,foreground_unrolled),foreground))
    # Keep matching order and all floating-point arithmetic unchanged.
    old=T._update_measurements
    source=textwrap.dedent(inspect.getsource(old))
    target='''        if not valid[track_index, detection_index]:
            break
'''
    assert target in source
    source=source.replace(target,'')
    scope=dict(old.__globals__);exec(compile(source,'<candidate_exact_assignment>','exec'),scope)
    new=scope['_update_measurements']
    extractors=[E(),E()];timings=[[],[]]
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:records=pickle.load(f)
    try:
        for i,(d,kw,_) in enumerate(records):
            features={}
            for j in ((0,1) if i%2 else (1,0)):
                T._update_measurements=(old,new)[j]
                t=time.perf_counter();features[j]=extractors[j].step_detections(d.bullet_positions,d.plane_position,**kw)
                timings[j].append((time.perf_counter()-t)*1000)
            exact(features[0],features[1]);exact(state(extractors[0]),state(extractors[1]))
            exact(tracker_prediction_hints(extractors[0].tracker,(820,820,3)),tracker_prediction_hints(extractors[1].tracker,(820,820,3)))
            if (i+1)%600==0:print(f'{i+1} exact tracker updates',flush=True)
    finally:T._update_measurements=old
    result['redundant_valid_check']=dict(updates=len(records),features_full_state_hints_bitwise_equal=True,
        before_ms=float(np.mean(timings[0])),candidate_ms=float(np.mean(timings[1])))
    (out/'screening.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
