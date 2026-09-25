"""Explicit development smoke and frozen validation using the project rollout."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import argparse
from dataclasses import asdict,replace
import json
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.parallel_evaluation import run_parallel_rollout
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.artifacts import atomic_write_json,sha256_file
from tools.pixel_guard_receding import RecedingGuardConfig,install_receding_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=('development','validation','controls'),default='development')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seed-file',type=Path)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    parser.add_argument('--attribution',action='store_true')
    parser.add_argument('--safety-threshold',type=float,default=.18)
    args=parser.parse_args()
    if not 0 < args.safety_threshold < 1:
        parser.error('safety threshold must be in (0, 1)')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    split=json.loads((ROOT/'diagnostics/seven_seed_repair_20260908/split.json').read_text())
    if args.split=='controls':
        if args.seed_file is None: parser.error('controls require --seed-file')
        seeds=json.loads(args.seed_file.read_text())
        if set(seeds)&set(split['development_seeds']+split['validation_seeds']):
            parser.error('controls must be disjoint from development and validation')
    else:
        if args.seed_file is not None: parser.error('development and validation use the frozen split')
        seeds=split[args.split+'_seeds']
    config=RecedingGuardConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    checkpoint_hash=sha256_file(checkpoint)
    if checkpoint_hash!=split['checkpoint_sha256']:
        parser.error('checkpoint differs from the repair baseline')
    paths=[*sorted((ROOT/'barrage_rl').glob('*.py')),*[ROOT/'tools'/p for p in ('pixel_guard_candidate.py','pixel_guard_refined.py','pixel_recovery_planner.py','pixel_recovery_refined.py','pixel_search_kernel.py','pixel_guard_receding.py','pixel_receding_kernel.py','evaluate_receding_guard.py','evaluate_pixel_guard_refined.py','measure_visible_window.py','profile_visible_policy.py')],ROOT/'Barrage.py',ROOT/'image/plane(0).gif',ROOT/'image/bullet(5).gif']
    paths.append(ROOT/'tools/pixel_guard_continuation.py')
    hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in paths}
    if args.split=='validation':
        if args.freeze is None: parser.error('validation requires a frozen candidate manifest')
        frozen=json.loads(args.freeze.read_text())
        if frozen['config']!=asdict(config) or frozen['source_hashes']!=hashes or frozen['checkpoint_sha256']!=checkpoint_hash or frozen['safety_threshold']!=args.safety_threshold:
            parser.error('candidate differs from frozen configuration or sources')
    torch.set_num_threads(1)
    agent,spec,_=load_tracked_agent(str(checkpoint),torch.device(args.device),analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    agent.safety_threshold=args.safety_threshold
    agent._action_selector.safety_threshold=args.safety_threshold
    guard=install_receding_guard(agent,config)
    started=time.perf_counter()
    result=run_parallel_rollout(agent=agent,spec=spec,episodes=len(seeds),workers=min(10,len(seeds)),
        seed=seeds[0],episode_seeds=seeds,env_kwargs=replace(TARGET_TASK,bullet_count=300).env_kwargs(),
        wall_threshold=40,rendered_rgb=True,causal_action_delay_steps=0,
        collect_failure_diagnostics=args.attribution,failure_lookback_decisions=36 if args.attribution else 0,
        progress_callback=lambda completed,total: print(f'completed={completed}/{total}',flush=True))
    changed=[p for p,h in hashes.items() if sha256_file(ROOT/p)!=h]
    payload=dict(split=args.split,smoke_test=True,formal_evaluation=False,config=asdict(config),safety_threshold=args.safety_threshold,
        source_hashes=hashes,source_files_changed_during_run=changed,seeds=seeds,
        checkpoint_sha256=sha256_file(checkpoint),bullet_count=300,targeted_bullet_probability=.1,
        evaluation_episode_limit_seconds=120,survival_seconds=result.survival_times.tolist(),
        success_at_limit=float((result.survival_times>=120).mean()),
        elapsed_seconds=time.perf_counter()-started,guard=guard.manifest(),failures=result.failure_diagnostics,
        agent_counters={key:getattr(agent,key) for key in ('decision_count','filtered_action_count','all_unsafe_count','overridden_decision_count','analytic_gate_decision_count')})
    atomic_write_json(args.output,payload)
    print(json.dumps({k:payload[k] for k in ('survival_seconds','success_at_limit','elapsed_seconds')}),flush=True)
    if changed or sha256_file(checkpoint)!=checkpoint_hash:
        raise RuntimeError(f'Checkpoint or source changed during rollout: {changed}')


if __name__=='__main__':main()
