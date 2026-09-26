"""Sample block-level tracker timings from live images, without source edits."""
import argparse
import ast
import copy
from datetime import datetime
import functools
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

SPECS = {'_update_detections': [(321, 329, 'normalize_detections')],
 '_update_measurements': [(437, 458, 'plane_motion'),
                          (460, 466, 'initial_tracks'),
                          (468, 473, 'reuse_predictions'),
                          (505, 525, 'matched_track_updates'),
                          (529, 564, 'occlusion_and_missed_tracks'),
                          (565, 568, 'new_tracks')],
 '_fit_velocities': [(267, 271, 'regression_basis'),
                     (272, 284, 'history_to_array'),
                     (285, 287, 'regression_and_norm'),
                     (290, 307, 'normalize_and_store_velocities')],
 '_fit_velocity': [],
 '_trim_tracks': [(361, 366, 'stable_retention_sort')],
 '_retention_threats': [(372, 374, 'retention_arrays'),
                        (375, 382, 'retention_validation'),
                        (383, 396, 'retention_geometry_and_scores')],
 '_features': [(111, 118, 'allocate_feature_buffers'),
               (120, 121, 'stack_track_arrays'),
               (122, 136, 'relative_motion_ttc_clearance'),
               (137, 151, 'track_metadata_arrays'),
               (152, 158, 'threat_sort'),
               (159, 185, 'pack_object_features'),
               (187, 225, 'global_features')],
 '_history_groups': [],
 '_association_arrays': [],
 '_assignment_pairs': []}


class Section:
    def __init__(self, owner, label):self.created=time.perf_counter();self.owner=owner;self.label=label
    def __enter__(self):
        self.child=0.;self.start=time.perf_counter();self.owner.stack.append(self)
    def __exit__(self,*exc):
        elapsed=time.perf_counter()-self.start
        self.owner.stack.pop()
        value=self.owner.current.setdefault(self.label,[0.,0.,0])
        value[0]+=elapsed*1000;value[1]+=(elapsed-self.child)*1000;value[2]+=1
        # Do not charge a nested timer's bookkeeping to its parent code block.
        if self.owner.stack:self.owner.stack[-1].child+=time.perf_counter()-self.created


class BlockProfiler:
    def __init__(self,sample_every=10):
        self.sample_every=sample_every;self.active=False;self.stack=[];self.current={}
        self.rows=[];self.restores=[];self.locations={};self.decisions=0
    def section(self,label):return Section(self,label)
    def compile_method(self,cls,name):
        original=getattr(cls,name);path=Path(inspect.getsourcefile(original))
        descriptor=inspect.getattr_static(cls,name)
        tree=ast.parse(path.read_text(encoding='utf-8'))
        owner=tree if inspect.ismodule(cls) else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls.__name__)
        function=copy.deepcopy(next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name))
        pristine=ast.dump(function,include_attributes=False)
        specs=getattr(self,'specs',SPECS)[name]
        if not specs:
            label='method:'+name
            self.locations[label]=dict(file=str(path.resolve()),start=function.lineno,end=function.end_lineno)
            @functools.wraps(original)
            def method_only(*a,**kw):
                if not self.active:return original(*a,**kw)
                with self.section(label):return original(*a,**kw)
            setattr(cls,name,staticmethod(method_only) if isinstance(descriptor,staticmethod) else method_only)
            self.restores.append((cls,name,descriptor))
            return
        def wrap(body,label):
            self.locations[label]=dict(file=str(path.resolve()),start=body[0].lineno,
                end=max(n.end_lineno for statement in body for n in ast.walk(statement)
                        if getattr(n,'end_lineno',None) is not None))
            node=ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Attribute(value=ast.Name(id='_tracker_block_timer',ctx=ast.Load()),attr='section',ctx=ast.Load()),
                args=[ast.Constant(label)],keywords=[]))],body=body)
            return ast.copy_location(node,body[0])
        def rewrite(body):
            out=[];i=0
            while i<len(body):
                node=body[i]
                match=next((s for s in specs if s[0]<=node.lineno and node.end_lineno<=s[1]),None)
                if match:
                    group=[node];i+=1
                    while i<len(body) and match[0]<=body[i].lineno and body[i].end_lineno<=match[1]:
                        group.append(body[i]);i+=1
                    out.append(wrap(group,name+'.'+match[2]))
                else:
                    for field,value in ast.iter_fields(node):
                        if isinstance(value,list) and value and all(isinstance(x,ast.stmt) for x in value):
                            setattr(node,field,rewrite(value))
                    out.append(node);i+=1
            return out
        function.body=rewrite(function.body)
        function.body=[wrap(function.body,'method:'+name)]
        class Strip(ast.NodeTransformer):
            def visit_With(self,node):
                if isinstance(node.items[0].context_expr,ast.Call) and isinstance(node.items[0].context_expr.func,ast.Attribute) and isinstance(node.items[0].context_expr.func.value,ast.Name) and node.items[0].context_expr.func.value.id=='_tracker_block_timer':
                    output=[]
                    for item in node.body:
                        result=self.visit(item);output.extend(result if isinstance(result,list) else [result])
                    return output
                return self.generic_visit(node)
        assert ast.dump(Strip().visit(copy.deepcopy(function)),include_attributes=False)==pristine
        if isinstance(descriptor,staticmethod):function.decorator_list=[]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
        namespace=dict(original.__globals__,_tracker_block_timer=self)
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
        instrumented=namespace[name]
        @functools.wraps(original)
        def measured(*a,**kw):return instrumented(*a,**kw) if self.active else original(*a,**kw)
        setattr(cls,name,staticmethod(measured) if isinstance(descriptor,staticmethod) else measured)
        self.restores.append((cls,name,descriptor))
    def install(self):
        from barrage_rl.image_oracle import PersistentImageTracker
        from barrage_rl.tracked_policy import TrackedFeatureExtractor
        from barrage_rl.window_tracker import WindowImageTracker
        for name in SPECS:
            owner=TrackedFeatureExtractor if name=='_features' else PersistentImageTracker
            self.compile_method(owner,name)
            if name in ('_history_groups','_association_arrays','_assignment_pairs'):
                self.compile_method(WindowImageTracker,name)
        original=TrackedFeatureExtractor.step_detections
        def step(extractor,*a,**kw):
            self.decisions+=1
            sampled=(self.decisions-1)%self.sample_every==0
            self.active=sampled;self.current={};start=time.perf_counter()
            try:return original(extractor,*a,**kw)
            finally:
                elapsed=(time.perf_counter()-start)*1000
                self.active=False
                self.rows.append(dict(decision=self.decisions,sampled=sampled,total_ms=elapsed,
                                      track_count=len(extractor.tracker.tracks),blocks=self.current if sampled else {}))
        TrackedFeatureExtractor.step_detections=step
        self.restores.append((TrackedFeatureExtractor,'step_detections',original))
    def close(self):
        for cls,name,original in reversed(self.restores):setattr(cls,name,original)
    def report(self):
        sampled=[r for r in self.rows if r['sampled']];n=len(sampled)
        ranking=[]
        for label,location in self.locations.items():
            values=np.array([r['blocks'].get(label,[0.,0.,0]) for r in sampled])
            if not n or not values[:,2].sum():continue
            ranking.append(dict(block=label,**location,mean_exclusive_ms=float(values[:,1].mean()),
                                mean_inclusive_ms=float(values[:,0].mean()),calls_per_sample=float(values[:,2].mean()),
                                exclusive_us_per_call=float(values[:,1].sum()/values[:,2].sum()*1000)))
        ranking.sort(key=lambda r:r['mean_exclusive_ms'],reverse=True)
        return dict(sample_every=self.sample_every,updates=len(self.rows),sampled_updates=n,ranking=ranking,
                    sampled_mean_ms=float(np.mean([r['total_ms'] for r in sampled])),
                    unsampled_mean_ms=float(np.mean([r['total_ms'] for r in self.rows if not r['sampled']])),
                    note='Block timers add overhead, especially in per-track loops. Exclusive times exclude children and their measured timer bookkeeping; inner timer overhead remains. Unsampled updates execute original methods. Line ranges refer to original source.')


def validate():
    from barrage_rl.tracked_policy import TrackedFeatureExtractor,TrackedPolicySpec
    from barrage_rl.artifacts import contents_equal
    rng=np.random.default_rng(41)
    positions=rng.uniform(.1,.9,(300,2)).astype(np.float32)
    velocities=rng.normal(0,.003,(300,2)).astype(np.float32)
    pool=[]
    for i in range(24):
        detection=np.clip(positions+velocities*i,0,1)
        if 8<=i<13:detection=detection[7:]
        if i==18:detection=np.empty((0,2),np.float32)
        pool.append((detection,np.array([.5+i*.0001,.5],np.float32)))
    def collect():
        extractor=TrackedFeatureExtractor(TrackedPolicySpec(expected_bullet_count=300));result=[]
        for bullets,plane in pool:
            features=extractor.step_detections(bullets,plane)
            result.append((copy.deepcopy(features),copy.deepcopy(extractor.tracker.__dict__)))
        return result
    baseline=collect();timer=BlockProfiler(1);timer.install()
    try:actual=collect()
    finally:timer.close()
    # Dataclass equality contains arrays; compare their dictionaries explicitly.
    for old,new in zip(baseline,actual):
        assert contents_equal(old[0],new[0])
        for state in (old[1],new[1]):state['tracks']=[vars(track) for track in state['tracks']]
        assert contents_equal(old[1],new[1])
    return dict(updates=24,all_features_and_tracker_state_equal=True,ast_operations_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--sample-every',type=int,default=10)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.sample_every<2 and not args.validate_only:parser.error('Use sample-every >= 2 to retain untimed comparison updates')
    validation=validate();print(json.dumps(validation),flush=True)
    if args.validate_only:return
    output=args.output_dir or ROOT/'diagnostics'/('tracker_blocks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    timer=BlockProfiler(args.sample_every);timer.install();previous=sys.argv
    try:
        sys.argv=[str(ROOT/'tools/test_visible_window.py'),'--seconds',str(args.seconds),
                  '--output-dir',str(output)]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        timer.close();sys.argv=previous
        output.mkdir(parents=True,exist_ok=True)
        (output/'tracker_blocks.json').write_text(json.dumps(timer.report(),indent=2),encoding='utf-8')
        (output/'tracker_updates.json').write_text(json.dumps(timer.rows),encoding='utf-8')
        (output/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print('Tracker blocks: '+str(output),flush=True)


if __name__=='__main__':main()
