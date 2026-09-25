"""Paired CPU inference timings for the existing guard horizon parameter."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import time
import json
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import argparse
    import numpy as np
    import torch
    from barrage_rl.evaluate_tracked_policy import load_tracked_agent
    from barrage_rl.window_inference import enable_window_inference
    from tools.pixel_guard_refined import install_refined_guard
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(6)
    agent,_,_=load_tracked_agent(str(ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'),torch.device('cpu'),analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    enable_window_inference(agent.model)
    guard=install_refined_guard(agent)
    z=np.load(ROOT/'diagnostics/success_plateau_audit_20260905/current_terminal_features.npz')
    samples=len(z['objects'])
    times={4:[],16:[]};actions={}
    for iteration in range(120):
        i=iteration%samples
        for steps in ([4,16] if iteration%2==0 else [16,4]):
            guard.config=replace(guard.config,physics_steps=steps)
            started=time.perf_counter()
            a=agent.act_features(z['objects'][i:i+1],z['masks'][i:i+1],z['globals'][i:i+1])
            elapsed=(time.perf_counter()-started)*1000
            assert a.shape==(1,) and 0<=int(a[0])<9
            key=(i,steps)
            if key in actions:assert actions[key]==int(a[0])
            actions[key]=int(a[0])
            if iteration>=20:times[steps].append(elapsed)
    report={'cpu_threads':6,'scope':'model and guard on identical saved image features; excludes capture, tracker and rendering',
            'results':{str(k):{'mean_ms':float(np.mean(v)),'p99_ms':float(np.percentile(v,99)),'max_ms':max(v),'samples':len(v)} for k,v in times.items()}}
    guard.config=replace(guard.config,physics_steps=16)
    started=time.perf_counter();count=0
    while time.perf_counter()-started<60:
        i=count%samples
        a=agent.act_features(z['objects'][i:i+1],z['masks'][i:i+1],z['globals'][i:i+1])
        assert int(a[0])==actions[(i,16)]
        count+=1
    report.update(stress_seconds=time.perf_counter()-started,stress_decisions=count)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
