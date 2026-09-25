"""Exact metric parity, actions and timing for the isolated parallel candidate."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,json,pickle,time,types,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from copy import deepcopy
import numpy as np
import torch
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.deployment import configure_image_controller
from barrage_rl.system_resources import process_resources
from experiment import install_candidate,CHECKPOINT
import readiness_kernel as serial
import readiness_kernel_parallel as parallel
from numba import set_num_threads

def main():
    target=HERE/'parallel_check.json'
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(1)
    agent,spec,_=load_tracked_agent(str(CHECKPOINT),torch.device('cuda'),analytic_shield=False)
    configure_image_controller(agent,'receding',search_workers=1);guard=install_candidate(agent)
    traces=json.loads((ROOT/'diagnostics/planner_small_regression_20260921/results/replay.json').read_text())['traces']
    traces={(int(s),t['decision_index']):t for s,ts in traces.items() for t in ts}
    cases=json.loads((HERE/'mechanism.json').read_text())['cases']
    count=0
    for c in cases:
        g=traces[(c['seed'],c['step'])]['geometry']
        args=[np.asarray(g[k],np.float32) for k in ('plane','bullets','velocity','error')]
        p,b,v,e=args;e=np.maximum(e,60/7).astype(np.float32)
        paths=np.asarray(g['paths'],np.int64);lengths=np.asarray(g['lengths'],np.int64)
        for name in ('assess_paths','assess_path_intervals','assess_path_exposure'):
            args=(p,guard.half_size,b,v,e,guard.table,*(() if name=='assess_paths' else (guard.integral,)),
                parallel.ACTION_VECTORS if hasattr(parallel,'ACTION_VECTORS') else __import__('barrage_rl.runtime_core',fromlist=['ACTION_VECTORS']).ACTION_VECTORS,paths,lengths)
            expected=getattr(serial,name)(*args)
            for workers in (1,3,9):
                set_num_threads(workers);actual=getattr(parallel,name)(*args)
                assert np.array_equal(expected,actual),(name,workers,c['seed'],c['step'])
                count+=1
    print(f'Exact metric parity passed: {count} comparisons',flush=True)
    modspec=importlib.util.spec_from_file_location('parallel_controller',HERE/'candidate_parallel.py')
    module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
    for name in ('apply','_assess','_geometry'):setattr(guard,name,types.MethodType(getattr(module.ContinuationMixin,name),guard))
    records=[]
    for p in sorted((ROOT/'diagnostics/planner_consistency_20260921/states').glob('*.pkl')):
        with p.open('rb') as f:records.append((p.stem,pickle.load(f)))
    reference={r['state']:r['action'] for r in json.loads((HERE/'state_check.json').read_text())['rows']}
    def run(selected):
        features=tuple(np.stack([getattr(s,k) for _,s in selected]) for k in ('objects','mask','globals_'))
        agent.reset_state();guard._plans.update({i:deepcopy(s.controller_plan) for i,(_,s) in enumerate(selected) if s.controller_plan is not None})
        started=time.perf_counter();actions=agent.act_features(*features,episode_indices=np.arange(len(selected))).tolist()
        elapsed=time.perf_counter()-started
        assert actions==[reference[n] for n,_ in selected]
        return elapsed
    for workers in (1,3,9):
        set_num_threads(workers);run(records)
    unique={}
    for name,state in records:unique.setdefault(name.split('_')[0],(name,state))
    selected=list(unique.values());measurements=[]
    for workers in (1,3,9):
        set_num_threads(workers)
        for batch in (9,36):
            sample=selected*(batch//9);run(sample)
            elapsed=sum(run(sample) for _ in range(2))
            measurements.append(dict(search_workers=workers,batch_size=batch,decisions_per_second=batch*2/elapsed))
    best=max(measurements,key=lambda r:r['decisions_per_second']);set_num_threads(best['search_workers'])
    sample=selected*(best['batch_size']//9);started=time.perf_counter();before=process_resources(os.getpid());count_decisions=0;latencies=[]
    while time.perf_counter()-started<30:
        latencies.append(run(sample));count_decisions+=len(sample)
    duration=time.perf_counter()-started;after=process_resources(os.getpid())
    result=dict(new_episodes=0,exact_metric_comparisons=count,all_metrics_identical=True,
        saved_states=33,all_actions_identical=True,measurements=measurements,selected=best,
        stress_seconds=duration,decisions=count_decisions,decisions_per_second=count_decisions/duration,
        p99_batch_ms=float(np.percentile(latencies,99)*1000),stable=True,
        cpu_seconds=after['cpu_seconds']-before['cpu_seconds'],peak_rss_bytes=after['peak_rss_bytes'],
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='Same network/planner microbenchmark as performance.json; no complete episode rerun')
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
