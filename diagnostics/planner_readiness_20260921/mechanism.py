"""Inspect existing snapshots only; no complete episodes or policy truth inputs."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import json
import pickle
import numpy as np
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.runtime_core import ACTION_VECTORS
from tools.pixel_guard_candidate import PixelGuard,PixelGuardConfig

def main():
    source=ROOT/'diagnostics/planner_small_regression_20260921/results'
    replay=json.loads((source/'replay.json').read_text())
    audit=json.loads((source/'route_audit.json').read_text())['rows']
    traces={(int(seed),t['decision_index']):t for seed,ts in replay['traces'].items() for t in ts}
    guard=PixelGuard(PixelGuardConfig())
    env=BarrageVisionEnv(**TARGET_TASK.env_kwargs());env.reset(seed=1)
    cases=[];coverage=[]
    for row in audit:
        key=(row['seed'],row['decision_index']);t=traces[key];g=t['geometry']
        with (source/'worlds'/f'{key[0]}_{key[1]}.pkl').open('rb') as f:state=pickle.load(f)
        b=np.asarray(g['bullets'],np.float32);v=np.asarray(g['velocity'],np.float32);err=np.asarray(g['error'])
        dist=np.linalg.norm(b[:,None]-state.bullet_positions[None],axis=2)
        nearest=dist.argmin(axis=1)
        for j,k in enumerate(nearest):
            if dist[j,k]>1:continue
            # Close competing bullets make nearest-neighbour identity ambiguous.
            if np.count_nonzero(dist[j]<1)>1:continue
            delta=float(np.max(np.abs(v[j]-state.bullet_velocities[k])))
            coverage.append(dict(seed=key[0],step=key[1],track=j,velocity_component_error=delta,bound=float(err[j]),outside=bool(delta>err[j])))
        if row['chosen_result']['survived_full_path'] or not row['surviving_alternative_roots']:continue
        chosen=row['chosen'];route=g['paths'][chosen];env.restore_state(state)
        for a in route:
            _,_,dead,truncated,_=env.step(int(a))
            if dead or truncated:break
        elapsed=(env.physics_steps-state.physics_steps)/120
        plane=np.asarray(g['plane'],np.float32).copy()
        for n in range(env.physics_steps-state.physics_steps):
            plane=np.clip(plane+ACTION_VECTORS[route[n//4]]*2,guard.half_size,820-guard.half_size)
        fatal=[]
        for k in env.colliding_bullet_indices():
            j=int(nearest.tolist().index(int(k))) if int(k) in nearest else int(np.argmin(np.linalg.norm(b-state.bullet_positions[k],axis=1)))
            predicted=b[j]+v[j]*elapsed
            actual_relative=env.bullet_positions[k]-env.plane_position
            predicted_relative=predicted-plane
            fatal.append(dict(bullet=int(k),track=j,initial_distance=float(dist[j,k]),
                initial_plane_error=(np.asarray(g['plane'])-state.plane_position).tolist(),
                initial_bullet_error=(b[j]-state.bullet_positions[k]).tolist(),
                velocity_error=(v[j]-state.bullet_velocities[k]).tolist(),velocity_bound=float(err[j]),
                relative_prediction_error=(predicted_relative-actual_relative).tolist(),
                modeled_relative_bound=float(1.5+err[j]*elapsed),
                linear_world_error=(env.bullet_positions[k]-(state.bullet_positions[k]+state.bullet_velocities[k]*elapsed)).tolist()))
        cases.append(dict(seed=key[0],step=key[1],chosen=chosen,actual_route_seconds=elapsed,
            selected_metrics=t['event']['metrics'][chosen],fatal=fatal,
            alternatives=[c for c in row['candidates'] if c['survived_full_path']]))
    output=dict(new_complete_episodes=0,short_saved_route_replays=len(cases),cases=cases,coverage=coverage,
        coverage_limitation='Failure-tail, unique nearest match within 1 pixel; no population calibration claim')
    target=HERE/'mechanism.json'
    with target.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(dict(cases=len(cases),matched_tracks=len(coverage),outside=sum(r['outside'] for r in coverage),
        max_error=max(r['velocity_component_error'] for r in coverage))))
    for c in cases:
        print(json.dumps({k:v for k,v in c.items() if k!='alternatives'}))

if __name__=='__main__':main()
