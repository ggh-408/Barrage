"""Profile only remaining model hotspots; capture image-derived model inputs."""
import argparse
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import pickle
import runpy
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.profile_tracker_blocks import BlockProfiler


class ModelProfiler(BlockProfiler):
    specs={
        'constant_action_clearance_by_object':[(56,60,'relative_velocity'),(61,65,'closest_time'),
            (66,68,'horizon_clipping'),(69,72,'closest_position'),(73,75,'norm_and_radius'),(76,76,'mask')],
        'forward':[(249,251,'attention'),(252,252,'residual_norm1'),(253,253,'feed_forward_and_norm2')],
    }
    def __init__(self):
        super().__init__(10)
        self.in_decision=False;self.samples=[];self.projection_depth=0

    def timed(self,obj,name,label,condition=None):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            if not self.active or (condition is not None and not condition()):return original(*args,**kwargs)
            with self.section(label):return original(*args,**kwargs)
        setattr(obj,name,measured);self.restores.append((obj,name,original))
        path=inspect.getsourcefile(original) or '<torch native>'
        try:start=inspect.getsourcelines(original)[1]
        except (OSError,TypeError):start=0
        self.locations[label]=dict(file=path,start=start,end=start)

    def install(self):
        import torch.nn.functional as functional
        import barrage_rl.tracked_policy as policy
        import barrage_rl.action_geometry as geometry
        import barrage_rl.window_inference as window_inference
        from barrage_rl.live_screen import LiveVisualController
        self.compile_method(geometry,'constant_action_clearance_by_object')
        original=policy.constant_action_clearance_by_object
        policy.constant_action_clearance_by_object=geometry.constant_action_clearance_by_object
        self.restores.append((policy,'constant_action_clearance_by_object',original))
        self.compile_method(policy._ActionCrossAttention,'forward')
        for name in ('prepare_action_geometry',):self.timed(policy.ActionQueryPolicy,name,name)
        if hasattr(window_inference,'window_clearance'):
            self.timed(window_inference,'window_clearance','fused_window_clearance')
        projection=functional._in_projection_packed
        def project(*args,**kwargs):
            self.projection_depth+=1
            try:return projection(*args,**kwargs)
            finally:self.projection_depth-=1
        functional._in_projection_packed=project
        self.restores.append((functional,'_in_projection_packed',projection))
        linear=functional.linear
        def project_linear(*args,**kwargs):
            if not self.active or not self.projection_depth:return linear(*args,**kwargs)
            label='attention.Q_projection' if args[1].shape[0]==args[1].shape[1] else 'attention.KV_projection'
            with self.section(label):return linear(*args,**kwargs)
        functional.linear=project_linear;self.restores.append((functional,'linear',linear))
        for label in ('attention.Q_projection','attention.KV_projection'):
            self.locations[label]=dict(file=inspect.getsourcefile(projection),start=inspect.getsourcelines(projection)[1],end=inspect.getsourcelines(projection)[1])
        # Native SDPA has no Python source body; report its measured boundary.
        sdpa=functional.scaled_dot_product_attention
        def attention(*args,**kwargs):
            if not self.active:return sdpa(*args,**kwargs)
            with self.section('attention.SDPA'):return sdpa(*args,**kwargs)
        functional.scaled_dot_product_attention=attention
        self.restores.append((functional,'scaled_dot_product_attention',sdpa))
        self.locations['attention.SDPA']=dict(file='<torch native scaled_dot_product_attention>',start=0,end=0)
        original_forward=policy.ActionQueryPolicy.forward_with_geometry
        def forward(model,objects,mask,globals_):
            if not self.in_decision:return original_forward(model,objects,mask,globals_)
            self.decisions+=1;sampled=(self.decisions-1)%10==0
            capture=(self.decisions-1)%60==0
            inputs=[x.detach().cpu().numpy().copy() for x in (objects,mask,globals_)] if capture else None
            self.active=sampled;self.current={};start=time.perf_counter()
            try:result=original_forward(model,objects,mask,globals_)
            finally:
                elapsed=(time.perf_counter()-start)*1000;self.active=False
                self.rows.append(dict(decision=self.decisions,sampled=sampled,total_ms=elapsed,blocks=self.current if sampled else {}))
            if capture:
                expected=[x.detach().cpu().numpy().copy() for x in result[:3]]
                expected.extend(x.detach().cpu().numpy().copy() for x in (result[3].clearance_by_object,result[3].normalized_minimum_clearance))
                self.samples.append(dict(inputs=inputs,expected=expected))
            return result
        policy.ActionQueryPolicy.forward_with_geometry=forward
        self.restores.append((policy.ActionQueryPolicy,'forward_with_geometry',original_forward))
        observe=LiveVisualController.observe_due_surface
        def observed(controller,surface):
            self.in_decision=True
            try:return observe(controller,surface)
            finally:self.in_decision=False
        LiveVisualController.observe_due_surface=observed
        self.restores.append((LiveVisualController,'observe_due_surface',observe))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=120)
    args=parser.parse_args()
    output=ROOT/'diagnostics'/('model_blocks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    profiler=ModelProfiler();profiler.install();previous=sys.argv
    try:
        sys.argv=[str(ROOT/'tools/diagnose_window_pacing.py'),'--seconds',str(args.seconds),'--record-window-state','--output-dir',str(output)]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        profiler.close();sys.argv=previous;output.mkdir(parents=True,exist_ok=True)
        if profiler.rows:
            (output/'model_blocks.json').write_text(json.dumps(profiler.report(),indent=2),encoding='utf-8')
        (output/'model_updates.json').write_text(json.dumps(profiler.rows),encoding='utf-8')
        with (output/'model_samples.pkl').open('wb') as stream:pickle.dump(profiler.samples,stream,protocol=5)
    print('Results: '+str(output),flush=True)


if __name__=='__main__':main()
