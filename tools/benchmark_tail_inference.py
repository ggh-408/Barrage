"""Choose a search thread count on identical saved image features."""
from __future__ import annotations
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import argparse
from pathlib import Path
import sys
import time
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.deployment import configure_image_controller
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.artifacts import atomic_write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stress-seconds',type=float,default=120.)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1)
    with np.load(ROOT/'runs/visual_set_v50/continuation_labels.npz') as z:
        take=np.flatnonzero(~z['validation'])[:120]
        features=tuple(z[k][take].copy() for k in ('objects','masks','globals'))
    device=torch.device('cuda')
    rows=[];reference=None
    for workers in (1,3,9):
        agent,_,_=load_tracked_agent(str(a.checkpoint),device,analytic_shield=True,
                                   analytic_shield_gate='learned_all_unsafe')
        configure_image_controller(agent,'receding',search_workers=workers)
        def passes(record=False):
            actions=[];latencies=[]
            agent.reset_state()
            for start in range(0,len(take),10):
                batch=tuple(x[start:start+10] for x in features)
                t=time.perf_counter()
                result=agent.act_features(*batch,episode_indices=np.arange(len(batch[0])))
                latencies.append(time.perf_counter()-t)
                actions.extend(result.tolist())
            return actions,latencies
        passes()
        actions=[];latencies=[]
        for _ in range(3):
            act,elapsed=passes();actions.extend(act);latencies.extend(elapsed)
        if reference is None: reference=actions
        if reference!=actions: raise RuntimeError('Search threads changed actions')
        row=dict(search_workers=workers,decisions=len(actions),
            decisions_per_second=len(actions)/sum(latencies),
            mean_batch_ms=float(np.mean(latencies)*1000),
            p95_batch_ms=float(np.percentile(latencies,95)*1000),actions_equal=True)
        rows.append(row);print(json.dumps(row),flush=True)
    selected=max(rows,key=lambda r:r['decisions_per_second'])['search_workers']
    agent,_,_=load_tracked_agent(str(a.checkpoint),device,analytic_shield=True,
                               analytic_shield_gate='learned_all_unsafe')
    configure_image_controller(agent,'receding',search_workers=selected)
    passes()
    started=time.perf_counter();count=0
    while time.perf_counter()-started<a.stress_seconds:
        actions,_=passes()
        if actions!=reference[:len(actions)]: raise RuntimeError('Stress actions changed')
        count+=len(actions)
    elapsed=time.perf_counter()-started
    report=dict(checkpoint=str(a.checkpoint),batch_size=10,torch_threads=1,rows=rows,
        selected_search_workers=selected,stress_seconds=elapsed,stress_decisions=count,
        stress_decisions_per_second=count/elapsed,
        scope='GPU model and image-derived guard; simulation and RGB detection excluded',
        formal_evaluation=False)
    atomic_write_json(a.output,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
