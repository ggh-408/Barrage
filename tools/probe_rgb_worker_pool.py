"""Confirm requested RGB workers execute separate rows and restore the mask."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from barrage_rl.foreground_kernel import njit
from barrage_rl.rgb_parallel import rgb_workers,dispatch
from numba import prange,get_thread_id,get_num_threads,set_num_threads,config

@njit(cache=False,parallel=True,fastmath=False)
def identify():
    rows=np.empty((820,2),np.float64)
    for row in prange(820):rows[row]=get_thread_id()
    for row in prange(820):
        total=0.
        for column in range(820):total+=np.sqrt(row+column+1.)
        rows[row,0]=get_thread_id()
        rows[row,1]=total
    return rows

if __name__=='__main__':
    set_num_threads(min(9,config.NUMBA_NUM_THREADS))
    before=get_num_threads()
    seen=set()
    with rgb_workers(4):
        for _ in range(10):
            result=dispatch(identify,identify)
            seen.update(np.unique(result[:,0]).astype(int).tolist())
    ids=sorted(seen)
    print(dict(ids=ids,before=before,after=get_num_threads()),flush=True)
    assert len(ids)==4 and get_num_threads()==before
    report=dict(requested_rgb_workers=4,executing_worker_ids=ids,
        planner_threads_before=before,planner_threads_after=get_num_threads(),mask_restored=True)
    p=ROOT/'diagnostics/parallel_rgb_deployment_20260925/worker_pool_probe.json'
    p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
