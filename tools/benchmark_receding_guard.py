"""Before/after CPU inference timing on development-only image features."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import argparse
import json
import time
import platform
import ctypes
from dataclasses import asdict,replace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.artifacts import atomic_write_json
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.window_inference import enable_window_inference
from tools.pixel_guard_refined import install_refined_guard
from tools.pixel_guard_receding import RecedingGuardConfig,install_receding_guard


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]


def hardware():
    memory=MemoryStatus()
    memory.length=ctypes.sizeof(memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    return dict(platform=platform.platform(),logical_processors=os.cpu_count(),physical_memory_bytes=memory.total,
                available_memory_bytes=memory.available,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                vram_bytes=torch.cuda.get_device_properties(0).total_memory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--stress-seconds',type=float,default=60)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=RecedingGuardConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    z=np.load(ROOT/'diagnostics/current_tail_audit_20260908/terminal_features.npz')
    # Original audit indices 4 and 6 are the reserved validation episodes.
    indices=np.flatnonzero(np.isin(z['episode_indices'],[0,1,2,3,5]))
    # Cover the last 12 frames of each development failure.
    indices=indices[z['history_indices'][indices]>=24]
    reference={}
    rows=[]
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    def make(mode,threads,workers=9):
        torch.set_num_threads(threads)
        agent,_,_=load_tracked_agent(str(checkpoint),torch.device('cpu'),analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
        enable_window_inference(agent.model)
        guard=install_refined_guard(agent) if mode=='baseline' else install_receding_guard(agent,replace(cfg,search_workers=workers))
        return agent,guard
    def run(agent,mode,i):
        j=indices[i%len(indices)]
        action=agent.act_features(z['objects'][j:j+1],z['masks'][j:j+1],z['globals'][j:j+1])
        assert action.shape==(1,) and 0<=action[0]<9
        key=(mode,int(j))
        if key in reference:assert reference[key]==int(action[0]),key
        reference[key]=int(action[0])
    for threads in (1,4,6,10):
        for mode,workers in [('baseline',1)]+[('candidate',w) for w in (1,3,6,9)]:
            agent,guard=make(mode,threads,workers)
            latencies=[]
            for i in range(20+len(indices)*2):
                started=time.perf_counter()
                run(agent,mode,i)
                if i>=20:latencies.append((time.perf_counter()-started)*1000)
            row=dict(mode=mode,threads=threads,search_workers=workers,mean_ms=float(np.mean(latencies)),p95_ms=float(np.percentile(latencies,95)),
                     p99_ms=float(np.percentile(latencies,99)),max_ms=max(latencies),samples=len(latencies))
            rows.append(row)
            print(json.dumps(row),flush=True)
    best=min((r for r in rows if r['mode']=='candidate'),key=lambda r:r['mean_ms'])
    agent,guard=make('candidate',best['threads'],best['search_workers'])
    started=time.perf_counter()
    cpu_started=time.process_time()
    count=0
    while time.perf_counter()-started<args.stress_seconds:
        run(agent,'candidate',count)
        count+=1
    elapsed=time.perf_counter()-started
    payload=dict(hardware=hardware(),config=asdict(replace(cfg,search_workers=best['search_workers'])),results=rows,selected_threads=best['threads'],
                 stress_seconds=elapsed,stress_decisions=count,stress_decisions_per_second=count/elapsed,
                 process_cpu_percent=100*(time.process_time()-cpu_started)/elapsed,
                 guard=guard.manifest(),validation_features_used=False,
                 measurement_scope='model plus guard on saved development image features; detection, rendering, collection, and optimizer excluded')
    atomic_write_json(args.output,payload)
    print(json.dumps({'output':str(args.output),'selected_threads':best['threads']}),flush=True)


if __name__=='__main__':main()
