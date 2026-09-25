"""Explicit three-episode diagnostic smoke replay, using the existing rollout."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import json
import time
from collections import deque
from dataclasses import replace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.parallel_evaluation import run_parallel_rollout
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.artifacts import atomic_write_json, sha256_file
from tools.pixel_guard_candidate import install_guard, PixelGuardConfig


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-test',action='store_true',required=True)
    parser.add_argument('--variant',choices=['conflict','young'],default='conflict')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2131058160,2118059892,2098867754])
    parser.add_argument('--seed-file',type=Path)
    parser.add_argument('--bullets',type=int,default=300)
    parser.add_argument('--threshold',type=float)
    parser.add_argument('--quick',action='store_true',help='skip attribution when screening parameters')
    parser.add_argument('--guard-steps',type=int,help='override the installed guard horizon')
    parser.add_argument('--search-depth',type=int,default=9)
    args=parser.parse_args()
    out=Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():raise FileExistsError(out)
    seeds=json.loads(args.seed_file.read_text()) if args.seed_file else args.seeds
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    torch.set_num_threads(1)
    agent,spec,_=load_tracked_agent(str(checkpoint),torch.device('cuda'),
        analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    if args.threshold is not None:
        agent.safety_threshold=args.threshold
    if args.variant=='young':
        from tools.pixel_guard_refined import install_refined_guard
        guard=install_refined_guard(agent)
    else:
        guard=install_guard(agent,PixelGuardConfig(allow_imminent_escape=True,resolve_interval_conflicts=True,recovery_search=True,compiled_search=True))
    if args.guard_steps is not None:
        guard.config=replace(guard.config,physics_steps=args.guard_steps)
    if args.search_depth != 9:
        from functools import partial
        from tools import pixel_recovery_planner, pixel_recovery_refined
        pixel_recovery_planner.recovery_action=partial(pixel_recovery_planner.recovery_action,depth=args.search_depth)
        pixel_recovery_refined.recovery_action=partial(pixel_recovery_refined.recovery_action,depth=args.search_depth)
    original=agent.act_features_with_diagnostics
    terminal={}
    young_samples=deque(maxlen=60)
    traces={}
    def capture(objects,masks,globals_,**kwargs):
        result=original(objects,masks,globals_,**kwargs)
        nominal,possible=guard.hazards(objects,masks,globals_)
        for row,index in enumerate(kwargs['episode_indices']):
            item={'nominal_collisions':nominal[row].tolist(), 'possible_collisions':possible[row].tolist(), 'action':int(result[0][row]), 'diagnostics':{k:np.asarray(v)[row].tolist() for k,v in result[1].items()}}
            terminal[int(index)]=item
            young=masks[row] & (objects[row,:,15]<.5) & (objects[row,:,9]==0) & (objects[row,:,8]>=.15)
            young &= np.linalg.norm(objects[row,:,:2]*820,axis=1)<40
            if young.any(): young_samples.append((objects[row].copy(),masks[row].copy(),globals_[row].copy(),seeds[int(index)]))
            traces.setdefault(int(index),deque(maxlen=36)).append(item)
        return result
    if not args.quick:
        agent.act_features_with_diagnostics=capture
    source_paths=[*sorted((ROOT/'barrage_rl').glob('*.py')),ROOT/'tools/pixel_guard_candidate.py',ROOT/'tools/pixel_recovery_planner.py',ROOT/'tools/pixel_search_kernel.py',ROOT/'tools/pixel_guard_refined.py',ROOT/'tools/pixel_recovery_refined.py',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in source_paths}
    started=time.perf_counter()
    rollout=run_parallel_rollout(agent=agent,spec=spec,episodes=len(seeds),workers=min(10,len(seeds)),
        seed=seeds[0],episode_seeds=seeds,env_kwargs=replace(TARGET_TASK,bullet_count=args.bullets).env_kwargs(),
        wall_threshold=40,rendered_rgb=True,causal_action_delay_steps=0,
        progress_callback=lambda completed,total: print(f'replay_progress completed={completed}/{total}',flush=True),
        collect_failure_diagnostics=not args.quick,failure_lookback_decisions=0 if args.quick else 36)
    payload={'variant':args.variant,'bullets':args.bullets,'search_depth':args.search_depth,'threshold':float(agent.safety_threshold),'formal_evaluation':False,'smoke_test':True,'seeds':seeds,
             'checkpoint_sha256':sha256_file(checkpoint),'source_hashes':hashes,'source_files_changed_during_run':[p for p,h in hashes.items() if sha256_file(ROOT/p)!=h],
             'elapsed_seconds':time.perf_counter()-started,
             'survival_seconds':rollout.survival_times.tolist(),
             'decision_traces':{str(k):list(v) for k,v in traces.items()},'terminal_image_hazards':terminal,'failures':rollout.failure_diagnostics,
             'guard':guard.manifest(),
             'privileged_diagnostic_labels_are_policy_input':False}
    atomic_write_json(out,payload)
    if young_samples:
        np.savez_compressed(out.with_suffix('.npz'),objects=np.stack([x[0] for x in young_samples]),masks=np.stack([x[1] for x in young_samples]),globals=np.stack([x[2] for x in young_samples]),seeds=np.array([x[3] for x in young_samples]))
    print(json.dumps({'survival_seconds':payload['survival_seconds'],'output':str(out)}),flush=True)


if __name__=='__main__':main()
