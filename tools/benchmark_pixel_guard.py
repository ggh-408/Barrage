"""CPU latency and sustained stability for the optional image guard."""

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from pathlib import Path
import sys
import time
import json
import platform
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.window_inference import enable_window_inference
from barrage_rl.artifacts import atomic_write_json
from tools.pixel_guard_candidate import PixelGuardConfig,install_guard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['interval','escape','search','search_jit'],default='interval')
    args=parser.parse_args()
    config=PixelGuardConfig(allow_imminent_escape=args.variant in ('escape','search','search_jit'),
                           recovery_search=args.variant in ('search','search_jit'),
                           compiled_search=args.variant=='search_jit')
    z=np.load(ROOT/'diagnostics/success_plateau_audit_20260905/current_terminal_features.npz')
    path=ROOT/f'diagnostics/pixel_guard_validation/latency_{args.variant}.json'
    if path.exists():raise FileExistsError(path)
    results=[]
    for threads in (1,4,10):
        torch.set_num_threads(threads)
        for mode in ('baseline',args.variant):
            agent,spec,metadata=load_tracked_agent(
                str(ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'),
                torch.device('cpu'),analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
            enable_window_inference(agent.model)
            if mode!='baseline': install_guard(agent,config)
            latencies=[]
            for i in range(230):
                j=i%len(z['seeds']);started=time.perf_counter()
                result=agent.act_features(z['objects'][j:j+1],z['masks'][j:j+1],z['globals'][j:j+1])
                elapsed=(time.perf_counter()-started)*1000
                assert result.shape==(1,) and 0<=result[0]<9
                if i>=30:latencies.append(elapsed)
            row={'threads':threads,'mode':mode,'samples':len(latencies),
                 'mean_ms':float(np.mean(latencies)),'p95_ms':float(np.percentile(latencies,95)),
                 'p99_ms':float(np.percentile(latencies,99)),'max_ms':max(latencies)}
            results.append(row);print(row,flush=True)
    best=min([r for r in results if r['mode']==args.variant],key=lambda r:r['mean_ms'])
    torch.set_num_threads(best['threads'])
    agent,spec,metadata=load_tracked_agent(
        str(ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'),torch.device('cpu'),
        analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    enable_window_inference(agent.model);guard=install_guard(agent,config)
    started=time.perf_counter();count=0
    while time.perf_counter()-started<60:
        j=count%len(z['seeds'])
        action=agent.act_features(z['objects'][j:j+1],z['masks'][j:j+1],z['globals'][j:j+1])
        assert action.shape==(1,) and 0<=action[0]<9
        count+=1
    elapsed=time.perf_counter()-started
    report={'platform':platform.platform(),'cpu':platform.processor(),'logical_processors':os.cpu_count(),
            'torch':torch.__version__,'results':results,'selected_threads':best['threads'],
            'stress_seconds':elapsed,'stress_decisions':count,'stress_decisions_per_second':count/elapsed,
            'limitation':'model and guard on saved RGB-derived features; screen detection excluded',
            'guard':guard.manifest()}
    atomic_write_json(path,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
