"""Deterministic planner contract checks without environment episodes."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,json,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from candidate_controller import ContinuationMixin

@dataclass
class Selection:
    actions: object
    immediate_risk: object
    all_unsafe: object
    counter_values: object
    raw_actions: object

def main():
    # Incumbent goes left toward a wall, but retained route goes right and has
    # a fully revalidated extension. Expiring historical metadata must not
    # erase the newly validated 17-step suffix.
    guard=ContinuationMixin.__new__(ContinuationMixin)
    guard.recovery_config=types.SimpleNamespace(preserve_plans=True,interval_consensus=True,search_depth=18,
        trigger_safe_actions=2,wall_reserve_pixels=48,velocity_error_floor=2)
    guard._context_indices=np.array([0]);guard._context_steps=1
    guard.half_size=np.array([10,10],np.float32)
    plane=np.array([65,410],np.float32)
    guard._geometry=lambda *args:(plane.copy(),np.zeros((0,2),np.float32),np.zeros((0,2),np.float32),np.zeros(0,np.float32))
    clear=np.zeros((1,9),bool)
    guard.hazards=lambda *args:(clear.copy(),clear.copy())
    guard.commit_guard=types.SimpleNamespace(hazards=guard.hazards)
    guard._plans={0:dict(path=np.full(17,2,np.int64),expected_plane=plane.copy(),remaining=1)}
    guard.counters=defaultdict(int);guard.elapsed_seconds=0;events=[]
    guard._decision_observer=lambda key,event:events.append(event)
    metrics=np.array([9,9,9,1,100,1,0],float)
    guard._assess=lambda *args:np.repeat(metrics[None],len(args[-1]),axis=0)
    selection=Selection(torch.tensor([1]),torch.zeros((1,9)),torch.tensor([False]),torch.zeros(5,dtype=torch.long),torch.tensor([1]))
    objects=np.zeros((1,1,16),np.float32);masks=np.zeros((1,1),bool);globals_=np.zeros((1,16),np.float32)
    out=guard.apply(selection,objects,masks,globals_)
    assert out.actions.tolist()==[2]
    assert events[-1]['reason']=='retained_route_certified'
    assert guard._plans[0]['remaining']==17 and len(guard._plans[0]['path'])==17
    # Each renewal maintains a fixed, fully assessed horizon; no old expiry.
    for _ in range(30):
        guard._plans[0]['expected_plane']=plane.copy()
        guard.apply(selection,objects,masks,globals_)
        assert guard._plans[0]['remaining']==17
    guard.reset([0]);assert not guard._plans
    result=dict(new_episodes=0,own_route_wall_eligibility=True,renewed_horizon_lifetime=True,
        repeated_renewals=31,episode_reset_clears_history=True,
        limitation='Synthetic control-flow contracts; physical safety requires replay and complete episode checks')
    with (HERE/'contracts.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
