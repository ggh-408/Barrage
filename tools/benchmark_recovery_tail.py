"""Alternating sustained CPU thread comparison on the same single agent."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import time
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.window_inference import enable_window_inference
from barrage_rl.artifacts import atomic_write_json
from tools.pixel_guard_candidate import PixelGuardConfig,install_guard


def main():
    output=ROOT/'diagnostics/pixel_guard_recovery/settings_tail.json'
    if output.exists():raise FileExistsError(output)
    data=np.load(ROOT/'diagnostics/success_plateau_audit_20260905/current_terminal_features.npz')
    torch.set_num_threads(4)
    agent,_,_=load_tracked_agent(str(ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'),
        torch.device('cpu'),analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    enable_window_inference(agent.model)
    install_guard(agent,PixelGuardConfig(allow_imminent_escape=True,recovery_search=True,compiled_search=True))
    def act(j):return int(agent.act_features(data['objects'][j:j+1],data['masks'][j:j+1],data['globals'][j:j+1])[0])
    reference=[act(i) for i in range(7)]
    records=[]
    for repeat in range(4):
        for threads in ((4,6) if repeat%2==0 else (6,4)):
            torch.set_num_threads(threads)
            for i in range(21):act(i%7)
            start=time.perf_counter();lat=[];count=0
            while time.perf_counter()-start<15:
                tick=time.perf_counter();action=act(count%7)
                lat.append((time.perf_counter()-tick)*1000)
                assert action==reference[count%7]
                count+=1
            r=dict(threads=threads,repeat=repeat,seconds=time.perf_counter()-start,
                samples=count,mean_ms=float(np.mean(lat)),p99_ms=float(np.percentile(lat,99)),
                max_ms=max(lat),above_control_period=int(np.count_nonzero(np.asarray(lat)>1000/30)),latency_ms=lat)
            records.append(r)
            print(json.dumps({k:v for k,v in r.items() if k!='latency_ms'}),flush=True)
    summary=[]
    for threads in (4,6):
        rows=[r for r in records if r['threads']==threads]
        lat=np.concatenate([r['latency_ms'] for r in rows])
        summary.append(dict(threads=threads,seconds=sum(r['seconds'] for r in rows),samples=len(lat),
            mean_ms=float(lat.mean()),p99_ms=float(np.percentile(lat,99)),max_ms=float(lat.max()),
            above_control_period=int(np.count_nonzero(lat>1000/30))))
    atomic_write_json(output,dict(summary=summary,records=records,
        scope='Alternating 15 second blocks, four blocks per setting; same saved image features and single agent.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
