"""Paired full-model replay for the window-only fused geometry path."""
import argparse
import copy
import gc
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.benchmark_model_norm import model_for


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    args=parser.parse_args()
    window=json.loads((args.baseline/'window.json').read_text())
    torch.set_num_threads(window['torch_threads'])
    model=model_for(window['checkpoint'])
    reference=copy.deepcopy(model);reference._window_clearance=None
    with (args.baseline/'model_samples.pkl').open('rb') as stream:samples=pickle.load(stream)
    inputs=[tuple(torch.from_numpy(x) for x in sample['inputs']) for sample in samples]
    def arrays(result):return [x.numpy() for x in (*result[:3],result[3].clearance_by_object,result[3].normalized_minimum_clearance)]
    with torch.inference_mode():
        for index,(x,sample) in enumerate(zip(inputs,samples)):
            old,new=arrays(reference.forward_with_geometry(*x)),arrays(model.forward_with_geometry(*x))
            assert all(np.array_equal(a,b) for a,b in zip(old,sample['expected'])),f'Baseline mismatch {index}'
            assert all(np.array_equal(a,b) for a,b in zip(old,new)),f'Optimized mismatch {index}'
        gc.collect(2)
        timings={'before':[],'after':[]}
        for repeat in range(20):
            for index,x in enumerate(inputs):
                order=(('before',reference),('after',model))
                if (index+repeat)%2:order=order[::-1]
                for label,net in order:
                    start=time.perf_counter();net.forward_with_geometry(*x)
                    timings[label].append((time.perf_counter()-start)*1000)
    stats=lambda v:dict(count=len(v),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),p99_ms=float(np.percentile(v,99)))
    report=dict(samples=len(samples),all_outputs_and_geometry_exact=True,threads=torch.get_num_threads(),
                before=stats(timings['before']),after=stats(timings['after']))
    (args.baseline/'geometry_replay.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
