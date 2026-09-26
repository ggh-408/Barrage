"""Check 33 existing image snapshots; no new complete episodes."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import json,pickle,time
from copy import deepcopy
import numpy as np
import torch
from experiment import install_candidate,CHECKPOINT
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.deployment import configure_image_controller
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.task_spec import TARGET_TASK

def main():
    target=HERE/'state_check.json'
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(1)
    agent,spec,_=load_tracked_agent(str(CHECKPOINT),torch.device('cuda'),analytic_shield=False)
    configure_image_controller(agent,'receding',search_workers=9);guard=install_candidate(agent)
    original=guard._assess;latest={};events={}
    def assess(*args):
        value=original(*args)
        latest.update(paths=args[-2].copy(),lengths=args[-1].copy(),metrics=value.copy())
        return value
    guard._assess=assess
    guard._decision_observer=lambda key,event:events.update({int(key):event})
    env=BarrageVisionEnv(**TARGET_TASK.env_kwargs());env.reset(seed=1)
    records=[]
    for path in sorted((ROOT/'diagnostics/planner_consistency_20260921/states').glob('*.pkl')):
        with path.open('rb') as f:records.append((path.stem,pickle.load(f)))
    rows=[];times=[]
    for name,state in records:
        agent.reset_state();latest.clear();events.clear()
        if state.controller_plan is not None:guard._plans[0]=deepcopy(state.controller_plan)
        started=time.perf_counter()
        action=agent.act_features(state.objects[None],state.mask[None],state.globals_[None],episode_indices=np.array([0]))
        times.append(time.perf_counter()-started)
        e=events[0];chosen=e.get('chosen')
        if chosen is not None:
            m=e['metrics'];ok=e['eligible']
            assert not np.any(m[ok,5]>=1) or m[chosen,5]>=1
            assert not (np.any(m[ok,3]>=1) and m[chosen,5]<1) or m[chosen,3]>=1
            route=latest['paths'][chosen]
        elif e['reason']=='model_full_horizon':route=np.full(18,int(action[0]))
        elif e['reason']=='retained_route_certified':
            route=np.concatenate(([int(action[0])],guard._plans[0]['path']))
        else:raise RuntimeError(e['reason'])
        env.restore_state(state.env_snapshot);dead=False
        for a in route:
            _,_,dead,truncated,_=env.step(int(a))
            if dead or truncated:break
        metric=e['metrics'][chosen] if chosen is not None else e['metrics']
        rows.append(dict(state=name,action=int(action[0]),incumbent=int(e['incumbent']),reason=e['reason'],
            survived_fixed_route=not dead,route_seconds=(env.physics_steps-state.env_snapshot.physics_steps)/120,
            metrics=np.asarray(metric).tolist(),saved_remaining=guard._plans.get(0,{}).get('remaining'),
            original_remaining=(state.controller_plan or {}).get('remaining')))
    result=dict(new_complete_episodes=0,saved_states=len(rows),short_route_replays=len(rows),rows=rows,
        survived_fixed_routes=sum(r['survived_fixed_route'] for r in rows),
        cold_first_seconds=times[0],warm_mean_ms=float(np.mean(times[1:])*1000),
        limitation='Failure-biased snapshots; short open-loop routes include unknowable future respawns; no success-rate claim')
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
