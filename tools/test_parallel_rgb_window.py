"""Visible-window RGB worker sweep with frame-aligned CPU resource readings."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--breakdown',action='store_true')
    a=p.parse_args()
    import pygame
    start=None;cpu=None;samples=[];last=0.;peak_memory=0
    original=pygame.time.Clock
    class Clock:
        def __init__(self):self.clock=original()
        def tick_busy_loop(self,fps):
            nonlocal start,cpu,last
            now=time.perf_counter()
            if start is None:start=now;cpu=time.process_time();last=now
            if now-last>=1:
                samples.append(dict(seconds=now-start,cpu_seconds=time.process_time()-cpu))
                last=now
            return self.clock.tick_busy_loop(fps)
    pygame.time.Clock=Clock
    previous=sys.argv
    sys.argv=[str(ROOT/'tools/test_visible_window.py'),'--seconds',str(a.seconds),
              '--rgb-workers',str(a.workers),'--output-dir',str(a.output)]
    if a.breakdown:sys.argv.append('--breakdown')
    try:
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        pygame.time.Clock=original;sys.argv=previous
    # Use frame time, excluding report serialization after pygame.quit().
    elapsed=samples[-1]['seconds'];used=samples[-1]['cpu_seconds']
    r=dict(rgb_workers=a.workers,logical_cpus=os.cpu_count(),measured_seconds=elapsed,
        process_cpu_seconds=used,average_busy_logical_cpus=used/elapsed,
        machine_cpu_percent=100*used/elapsed/(os.cpu_count() or 1),samples=samples,
        note='Process CPU includes precise frame pacing, Torch, Numba, simulation and rendering. One-second samples exclude initialization; no competing benchmark is launched.')
    from tools.window_process_memory import process_memory
    r.update(process_memory())
    (a.output/'cpu_resources.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k!='samples'}),flush=True)

if __name__=='__main__':main()
