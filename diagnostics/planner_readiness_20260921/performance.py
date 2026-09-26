"""Bounded same-state planner throughput and deterministic worker checks."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,json,pickle,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from copy import deepcopy
import numpy as np
import torch
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.deployment import configure_image_controller
from barrage_rl.system_resources import process_resources,memory_status
from experiment import install_candidate,CHECKPOINT

def main():
    target=HERE/'performance.json'
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(1);states={}
    for p in sorted((ROOT/'diagnostics/planner_consistency_20260921/states').glob('*.pkl')):
        seed=p.stem.split('_')[0]
        if seed not in states:
            with p.open('rb') as f:states[seed]=pickle.load(f)
    states=list(states.values());assert len(states)==9
    rows=[]
    for variant in ('baseline','candidate'):
        agent,spec,_=load_tracked_agent(str(CHECKPOINT),torch.device('cuda'),analytic_shield=False)
        configure_image_controller(agent,'receding',search_workers=1)
        if variant=='candidate':install_candidate(agent)
        guard=agent._receding_pixel_guard
        from numba import set_num_threads
        reference=None;measurements=[]
        def run(batch):
            chosen_states=states*(batch//9)
            features=tuple(np.stack([getattr(s,k) for s in chosen_states]) for k in ('objects','mask','globals_'))
            agent.reset_state()
            guard._plans.update({i:deepcopy(s.controller_plan) for i,s in enumerate(chosen_states) if s.controller_plan is not None})
            started=time.perf_counter()
            actions=agent.act_features(*features,episode_indices=np.arange(batch)).tolist()
            elapsed=time.perf_counter()-started
            assert all(actions[i:i+9]==actions[:9] for i in range(0,batch,9))
            return actions[:9],elapsed
        for workers in (1,3,9):
            set_num_threads(workers)
            for batch in (9,36):
                actions,_=run(batch)
                if reference is None:reference=actions
                assert actions==reference
                durations=[]
                for _ in range(2):
                    actions,elapsed=run(batch);assert actions==reference;durations.append(elapsed)
                measurements.append(dict(search_workers=workers,batch_size=batch,decisions_per_second=batch*2/sum(durations)))
        best=max(measurements,key=lambda r:r['decisions_per_second']);set_num_threads(best['search_workers'])
        started=time.perf_counter();before=process_resources(os.getpid());count=0;latencies=[]
        while time.perf_counter()-started<30:
            actions,elapsed=run(best['batch_size']);assert actions==reference
            count+=best['batch_size'];latencies.append(elapsed)
        duration=time.perf_counter()-started;after=process_resources(os.getpid())
        row=dict(variant=variant,measurements=measurements,selected=best,stress_seconds=duration,
            decisions=count,decisions_per_second=count/duration,p99_batch_ms=float(np.percentile(latencies,99)*1000),
            cpu_seconds=after['cpu_seconds']-before['cpu_seconds'],peak_rss_bytes=after['peak_rss_bytes'],
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),worker_batch_actions_identical=True,stable=True)
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(new_episodes=0,unique_saved_states=9,variants=rows,memory=memory_status(),
        fallback=dict(search_workers=1,batch_size=9),
        scope='Frozen network and planner only; excludes collection, simulation, RGB extraction and optimizer; no training path was changed')
    with target.open('x') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
