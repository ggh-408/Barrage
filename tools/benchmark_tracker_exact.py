"""Paired exact tracker validation and foreground timing against saved source."""
import argparse
import ast
import copy
import json
from pathlib import Path
import pickle
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'diagnostics/tracker_exact_20260925'


def previous_method(cls, name, path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    namespace = dict(getattr(cls, name).__globals__)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


def variants():
    from barrage_rl.image_oracle import PersistentImageTracker as Tracker
    from barrage_rl.tracked_policy import TrackedFeatureExtractor as Extractor
    return [(Tracker, '_update_measurements', previous_method(Tracker, '_update_measurements', BASE/'image_oracle_before.py'), Tracker._update_measurements),
            (Extractor, '_features', previous_method(Extractor, '_features', BASE/'tracked_policy_before.py'), Extractor._features)]


def switch(items, before):
    for cls, name, old, new in items:
        setattr(cls, name, old if before else new)


def exact(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), 'Bitwise mismatch'
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x, y in zip(a, b): exact(x, y)
    elif isinstance(a, dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k], b[k])
    else:
        assert a==b, 'State mismatch'


def state(extractor):
    result = dict(vars(extractor.tracker))
    result['tracks'] = [vars(t) for t in result['tracks']]
    return result


def stats(values):
    return dict(count=len(values), mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values,95)))


def synthetic(items):
    from barrage_rl.tracked_policy import TrackedFeatureExtractor, TrackedPolicySpec
    rng = np.random.default_rng(2519)
    old = TrackedFeatureExtractor(TrackedPolicySpec())
    new = copy.deepcopy(old)
    initial = rng.uniform(.02,.98,(300,2)).astype(np.float32)
    velocity = rng.normal(0,.002,(300,2)).astype(np.float32)
    for i in range(240):
        bullets = np.clip(initial + velocity*i,0,1)
        if i % 23 < 4: bullets = bullets[35:]
        if i % 79 == 78: bullets = np.empty((0,2),np.float32)
        plane = None if i % 17 == 16 else np.array([.5+.05*np.sin(i),.5],np.float32)
        results=[]
        for before, extractor in [(True,old),(False,new)]:
            switch(items,before)
            results.append(extractor.step_detections(bullets,plane,decision_steps=1+i%3))
        exact(*results);exact(state(old),state(new))
    for dtype in (np.float16,np.float32,np.float64):
        for index,track in enumerate(old.tracker.tracks):
            track.position=track.position.astype(dtype if index%2 else np.float32)
            track.velocity=track.velocity.astype(dtype)
            if index%3==0:
                track.velocity[:]=[-0.0,0.0]
                track.velocity_known=False
        new=copy.deepcopy(old)
        results=[]
        for before,extractor in [(True,old),(False,new)]:
            switch(items,before)
            results.append(extractor.step_detections(bullets,plane,decision_steps=2))
        exact(*results);exact(state(old),state(new))
    result=dict(continuous_updates=240,mixed_dtype_and_signed_zero_cases=3,
                features_and_full_tracker_state_bitwise_equal=True)
    (BASE/'synthetic_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


def live(items, variant):
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    output=BASE/('before_measured' if variant=='before' else variant)
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    switch(items,variant=='before')
    records=[];times=[]
    step=TrackedFeatureExtractor.step_detections
    def timed(extractor,*a,**kw):
        start=time.perf_counter()
        try:return step(extractor,*a,**kw)
        finally:times.append((time.perf_counter()-start)*1000)
    TrackedFeatureExtractor.step_detections=timed
    act=LiveVisualController._act_detections
    def capture(controller,detections,**kw):
        record=(copy.deepcopy(detections),kw.copy())
        action=act(controller,detections,**kw)
        records.append((*record,action))
        return action
    LiveVisualController._act_detections=capture
    sys.argv=[str(ROOT/'tools/diagnose_window_pacing.py'),'--seconds','120','--record-window-state','--output-dir',str(output)]
    try:runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        TrackedFeatureExtractor.step_detections=step
        LiveVisualController._act_detections=act
        switch(items,False)
    (output/'tracker_timing.json').write_text(json.dumps(stats(times),indent=2),encoding='utf-8')
    with (output/'image_detections.pkl').open('wb') as stream:pickle.dump(records,stream,protocol=5)
    print(json.dumps(stats(times)),flush=True)


def replay(items, recording_path=None, configure_controllers=None, verify_planner_state=False):
    import torch
    from tools.train_targeted_dagger import install_runtime, CONTROLLER
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    from barrage_rl.runtime_core import tracker_prediction_hints
    install_runtime();torch.set_num_threads(10)
    from barrage_rl.deployment import configure_image_controller
    controllers=[]
    for _ in range(2):
        c=LiveVisualController(str(ROOT/'best.pt'),experimental_controller=CONTROLLER)
        configure_image_controller(c.agent,'receding',search_workers=9)
        controllers.append(c)
    if configure_controllers is not None:
        configure_controllers(controllers)
    recording_path = BASE/'after/image_detections.pkl' if recording_path is None else Path(recording_path)
    with recording_path.open('rb') as stream:records=pickle.load(stream)
    times=[[],[]];decision_times=[[],[]];features={};model_outputs={}
    for j,c in enumerate(controllers):
        forward=c.agent.model.forward_with_geometry
        def capture_model(*args,_forward=forward,_j=j,**kw):
            result=_forward(*args,**kw)
            tensors=(*result[:3],result[3].clearance_by_object,result[3].normalized_minimum_clearance)
            model_outputs[_j]=tuple(t.detach().cpu().numpy().copy() for t in tensors)
            return result
        c.agent.model.forward_with_geometry=capture_model
    for i,(detections,kwargs,expected) in enumerate(records):
        outputs={}
        for j in (range(2) if i%2==0 else (1,0)):
            switch(items,j==0)
            feature_method=TrackedFeatureExtractor._features
            def capture(extractor):
                result=feature_method(extractor);features[j]=result
                return result
            TrackedFeatureExtractor._features=capture
            c=controllers[j]
            decision_started=time.perf_counter()
            outputs[j]=c._act_detections(detections,**kwargs)
            decision_times[j].append((time.perf_counter()-decision_started)*1000)
            times[j].append(c._stage_tracking_ms[-1])
            TrackedFeatureExtractor._features=feature_method
        exact(features[0],features[1])
        exact(model_outputs[0],model_outputs[1])
        exact(state(controllers[0].tracked_extractor),state(controllers[1].tracked_extractor))
        if verify_planner_state:
            first, second = [c.agent._receding_pixel_guard for c in controllers]
            exact(first._plans, second._plans)
            exact(first.counters, second.counters)
        exact(tracker_prediction_hints(controllers[0].tracked_extractor.tracker,(820,820,3)),
              tracker_prediction_hints(controllers[1].tracked_extractor.tracker,(820,820,3)))
        assert outputs[0]==outputs[1]==expected, (i,outputs,expected)
        if i%600==599:print(f'Paired: {i+1}/{len(records)} exact features, state, hints, final action',flush=True)
    switch(items,False)
    result=dict(updates=len(records),features_state_hints_bitwise_equal=True,model_outputs_bitwise_equal=True,all_final_actions_equal=True,
                before=stats(times[0][1:]),after=stats(times[1][1:]))
    if verify_planner_state:
        result['planner_memory_and_counters_bitwise_equal'] = True
    result['decision_before']=stats(decision_times[0][1:])
    result['decision_after']=stats(decision_times[1][1:])
    result['decision_timing_scope']='Recorded detection through final action, with identical feature/model-output capture hooks; excludes capture and RGB detection.'
    (BASE/'paired_replay.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['synthetic','before','after','replay'])
    args=parser.parse_args();items=variants()
    if args.mode=='synthetic':synthetic(items)
    elif args.mode=='replay':replay(items)
    else:live(items,args.mode)
