"""Replay locally captured RGB inputs against the saved detector source."""
import argparse
import ast
import copy
import gc
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load_reference(path):
    import barrage_rl.live_screen as live
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DominantBackgroundSemanticizer')
    namespace=dict(vars(live))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['DominantBackgroundSemanticizer']


def main():
    from barrage_rl.live_screen import DominantBackgroundSemanticizer as Optimized
    from barrage_rl.artifacts import contents_equal
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args()
    # Only read samples created by this local diagnostic, never external pickles.
    with (args.baseline/'rgb_samples.pkl').open('rb') as stream:samples=pickle.load(stream)
    Reference=load_reference(args.baseline/'baseline_live_screen.py')
    before,after=Reference(),Optimized()
    exact=0
    for sample in samples:
        for variant in ('recorded','full','semantic'):
            kwargs=copy.deepcopy(sample['kwargs'])
            if variant=='full':kwargs=dict(include_semantic=False)
            if variant=='semantic':kwargs['include_semantic']=True
            outputs=[];states=[]
            for detector in (before,after):
                detector.__dict__=copy.deepcopy(sample['state'])
                result=detector.detect(sample['image'],**kwargs)
                outputs.append(vars(result));states.append(copy.deepcopy(vars(detector)))
            assert contents_equal(outputs[0],outputs[1]),f'Detection mismatch: {exact}, {variant}'
            assert contents_equal(states[0],states[1]),f'Detector state mismatch: {exact}, {variant}'
            if variant=='recorded':
                assert contents_equal(outputs[0],vars(sample['result'])),f'Instrumentation mismatch: {exact}'
                assert contents_equal(states[0],sample['after']),f'Instrumentation state mismatch: {exact}'
            exact+=1
    gc.collect(2)
    timings={'before':[],'after':[]}
    for repeat in range(args.repeats):
        for index,sample in enumerate(samples):
            order=(('before',before),('after',after))
            if (repeat+index)%2:order=order[::-1]
            for label,detector in order:
                detector.__dict__=copy.deepcopy(sample['state'])
                started=time.perf_counter()
                detector.detect(sample['image'],**sample['kwargs'])
                timings[label].append((time.perf_counter()-started)*1000)
    stats=lambda values:dict(count=len(values),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)))
    report=dict(samples=len(samples),exact_cases=exact,outputs_and_state_exact=True,
                instrumentation_exact=True,repeats=args.repeats,before=stats(timings['before']),after=stats(timings['after']))
    (args.baseline/'rgb_replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
