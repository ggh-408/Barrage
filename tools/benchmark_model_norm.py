"""Compare exact norm layouts on captured model inputs, without deployment edits."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def model_for(checkpoint):
    from barrage_rl.tracked_policy import ActionQueryPolicy,TrackedPolicySpec
    from barrage_rl.window_inference import enable_window_inference
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ActionQueryPolicy(TrackedPolicySpec(**saved['tracked_policy_spec']),**saved['model_hparams']).eval()
    model.load_state_dict(saved['model'])
    enable_window_inference(model)
    return model


def norm_variant(callback):
    import barrage_rl.action_geometry as geometry
    tree=ast.parse(inspect.getsource(geometry.constant_action_clearance_by_object))
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='vector_norm':node.func=ast.Name(id='_norm_callback',ctx=ast.Load())
            return node
    tree=ast.fix_missing_locations(Replace().visit(tree))
    namespace=dict(vars(geometry),_norm_callback=callback)
    exec(compile(tree,'<norm comparison>','exec'),namespace)
    return namespace['constant_action_clearance_by_object']


def main():
    import barrage_rl.tracked_policy as policy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    args=parser.parse_args()
    report=json.loads((args.baseline/'window.json').read_text())
    torch.set_num_threads(report['torch_threads'])
    model=model_for(report['checkpoint'])
    # This diagnostic isolates Torch norm alternatives even after deployment
    # enables the fused geometry path.
    model._window_clearance=None
    with (args.baseline/'model_samples.pkl').open('rb') as stream:samples=pickle.load(stream)
    inputs=[tuple(torch.from_numpy(x) for x in sample['inputs']) for sample in samples]
    original=policy.constant_action_clearance_by_object
    original_norm=torch.linalg.vector_norm
    functions={
        'baseline':original,
        'split_xy':norm_variant(lambda x,dim:torch.sqrt(x[...,0].square()+x[...,1].square())),
        'transpose':norm_variant(lambda x,dim:original_norm(x.movedim(-1,0).contiguous(),dim=0)),
    }
    def flatten(result):return (*result[:3],result[3].clearance_by_object,result[3].normalized_minimum_clearance)
    results={}
    try:
        with torch.inference_mode():
            expected=[flatten(model.forward_with_geometry(*x)) for x in inputs]
            for sample,result in zip(samples,expected):
                assert all(np.array_equal(t.numpy(),a) for t,a in zip(result,sample['expected'])),'Baseline capture mismatch'
            for label,fn in functions.items():
                policy.constant_action_clearance_by_object=fn
                exact=True;maximum=0.
                for x,ref in zip(inputs,expected):
                    actual=flatten(model.forward_with_geometry(*x))
                    exact &= all(torch.equal(a,b) for a,b in zip(actual,ref))
                    for a,b in zip(actual,ref):
                        finite=torch.isfinite(a)&torch.isfinite(b)
                        if finite.any():maximum=max(maximum,float((a[finite]-b[finite]).abs().max()))
                times=[]
                for _ in range(12):
                    for x in inputs:
                        start=time.perf_counter();model.forward_with_geometry(*x)
                        times.append((time.perf_counter()-start)*1000)
                results[label]=dict(exact=exact,max_absolute_difference=maximum,mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)))
    finally:policy.constant_action_clearance_by_object=original
    (args.baseline/'norm_variants.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
