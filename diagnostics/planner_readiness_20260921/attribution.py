"""Compare frozen ranking factors on 13 already-labelled route sets."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import numpy as np
from tools.pixel_guard_candidate import PixelGuard,PixelGuardConfig
from barrage_rl.runtime_core import ACTION_VECTORS
from readiness_kernel import assess_paths,assess_path_intervals,assess_path_exposure

def main():
    source=ROOT/'diagnostics/planner_small_regression_20260921/results'
    replay=json.loads((source/'replay.json').read_text())
    trace={(int(s),t['decision_index']):t for s,ts in replay['traces'].items() for t in ts}
    back={(int(f['seed']),t['decision_index']):t for f in replay['failures'] for t in f['backtrace']}
    route={(r['seed'],r['decision_index']):r for r in json.loads((source/'route_audit.json').read_text())['rows']}
    cases=json.loads((HERE/'mechanism.json').read_text())['cases']
    guard=PixelGuard(PixelGuardConfig());rows=[]
    for c in cases:
        key=(c['seed'],c['step']);t=trace[key];g=t['geometry'];r=route[key]
        p=np.asarray(g['plane'],np.float32);b=np.asarray(g['bullets'],np.float32);v=np.asarray(g['velocity'],np.float32)
        error=np.asarray(g['error'],np.float32);paths=np.asarray(g['paths'],np.int64);lengths=np.asarray(g['lengths'],np.int64)
        root=paths[:,0];inc=t['event']['incumbent'];prior_index=18 if len(paths)==27 else -1
        first=np.clip(p+ACTION_VECTORS[root]*8,guard.half_size,820-guard.half_size)
        walls=np.minimum(first-guard.half_size,820-guard.half_size-first).min(axis=1)
        end=np.clip(p+ACTION_VECTORS[inc]*32,guard.half_size,820-guard.half_size)
        wallgate=np.minimum(end-guard.half_size,820-guard.half_size-end).min()<48
        selections={}
        for cap in (False,True):
            err=np.maximum(error,2/(7/30)).astype(np.float32) if cap else error
            # The new lower bound leaves unknown velocity's 240 unchanged.
            m=assess_paths(p,guard.half_size,b,v,err,guard.table,ACTION_VECTORS,paths,lengths)
            interval=assess_path_intervals(p,guard.half_size,b,v,err,guard.table,guard.integral,ACTION_VECTORS,paths,lengths)
            exposure=assess_path_exposure(p,guard.half_size,b,v,err,guard.table,guard.integral,ACTION_VECTORS,paths,lengths)
            ok=np.flatnonzero(m[:,3]>=1/18-1e-12)
            if not len(ok):ok=np.arange(len(paths))
            for integrate in (False,True):
                def rank(i):
                    return (-int(interval[i]>=1),-int(m[i,3]>=1),*( (float(exposure[i]),) if integrate else ()),
                        -interval[i],-m[i,2],-m[i,1],-m[i,3],-min(m[i,4],48),-walls[i] if wallgate else 0,
                        int(i!=9+inc),int(prior_index<0 or i<prior_index),int(root[i]))
                chosen=int(min(ok,key=rank));name=f'window_cap_{cap}_exposure_{integrate}'
                if not cap and not integrate:assert chosen==r['chosen']
                selections[name]=dict(index=chosen,root=int(root[chosen]),fixed_route_survived=r['candidates'][chosen]['survived_full_path'])
        teacher=back[key]
        rows.append(dict(seed=key[0],step=key[1],teacher_action=teacher['teacher_action'],raw_policy_action=teacher['raw_policy_action'],
            executed_action=teacher['executed_action'],teacher_regret=teacher['executed_action_regret'],
            selections=selections))
    counts={name:sum(r['selections'][name]['fixed_route_survived'] for r in rows) for name in rows[0]['selections']}
    result=dict(new_episodes=0,existing_route_sets=len(rows),survived_selected_routes=counts,rows=rows,
        limitation='Same existing candidate paths only; does not measure changes in beam search or closed-loop success')
    with (HERE/'attribution.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(counts))

if __name__=='__main__':main()
