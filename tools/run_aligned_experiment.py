"""Run a paired deployed baseline, aligned DAgger, and continuation experiment."""
from __future__ import annotations
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json
import csv
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from barrage_rl.artifacts import atomic_write_json
from barrage_rl.train_tracked_policy import TrackedDAggerConfig, _resolve_run_seeds, train_tracked_policy
from barrage_rl.evaluate_tracked_policy import evaluate_tracked_checkpoint


def paired_failures(baseline_path, candidate_path, episode_limit):
    def read(path):
        with path.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        return {int(r['seed']):float(r['model_survival_seconds']) for r in rows}
    baseline,candidate=read(baseline_path),read(candidate_path)
    if len(baseline)!=200 or baseline.keys()!=candidate.keys():
        raise ValueError('Paired comparison requires the same 200 held-out seeds')
    old_failed={seed for seed,t in baseline.items() if t<episode_limit}
    new_failed={seed for seed,t in candidate.items() if t<episode_limit}
    return dict(rescued_failure_seeds=sorted(old_failed-new_failed),
        introduced_failure_seeds=sorted(new_failed-old_failed),
        shared_failure_seeds=sorted(old_failed & new_failed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Validate paths and print settings without starting evaluation or training')
    args=parser.parse_args()
    torch.set_num_threads(1)
    report=ROOT/'diagnostics/aligned_training_20260919'
    runs=[ROOT/'runs/visual_set_v49',ROOT/'runs/visual_set_v49_eval',ROOT/'runs/visual_set_v50']
    for folder in runs:
        if folder.exists() and any(folder.iterdir()): raise FileExistsError(folder)
    if (report/'experiment.json').exists(): raise FileExistsError(report/'experiment.json')
    config=TrackedDAggerConfig(output_dir=str(runs[0]),initial_checkpoint=str(ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'),
        evaluate_initial_checkpoint=False,rounds=1,samples_per_round=129600,replay_capacity=160000,
        num_envs=36,cpu_workers=9,batch_size=512,evaluation_workers=9,evaluation_batch_size=10,
        record_branch_snapshots=True,pixel_guard='receding',search_workers=1,
        fixed_safety_threshold=.18,random_action_probability=0.,failure_tail_decisions=60)
    if not Path(config.initial_checkpoint).is_file(): raise FileNotFoundError(config.initial_checkpoint)
    if args.dry_run:
        print(json.dumps(dict(status='ready',starts_training=False,output_directories=[str(p) for p in runs],
            source_checkpoint=config.initial_checkpoint,bullet_count=config.bullet_count,
            slots=config.max_objects,teacher_reaction_seconds=config.teacher_reaction_seconds,
            samples_per_round=config.samples_per_round,replay_capacity=config.replay_capacity,
            num_envs=config.num_envs,cpu_workers=config.cpu_workers,batch_size=config.batch_size,
            epochs_per_round=config.epochs_per_round,learning_rate=config.learning_rate,
            evaluation_episodes=config.evaluation_episodes,device=config.device),indent=2))
        return
    _resolve_run_seeds(config)
    report.mkdir(parents=True,exist_ok=True)
    started=time.time()
    source_paths=list((ROOT/'barrage_rl').glob('*.py'))+list((ROOT/'tools').glob('pixel*.py'))
    source_paths += [Path(__file__),ROOT/'tools/train_continuation_value.py']
    original_sources={p:p.read_bytes() for p in source_paths}
    atomic_write_json(report/'experiment.json',dict(source_checkpoint=config.initial_checkpoint,
        aligned_run=str(runs[0]),continuation_run=str(runs[2]),
        fixed_heldout_seeds=list(range(config.evaluation_seed,config.evaluation_seed+200)),
        collection_seed=config.collection_seed,training_seed=config.seed,
        task=dict(bullets=300,tracked_slots=384,targeted_probability=.10,episode_limit_seconds=120),
        teacher_reaction_seconds=.10,fresh_replay=True,samples_per_round=config.samples_per_round,
        controller='receding_continuation',source_files=[str(p.relative_to(ROOT)) for p in source_paths]))
    def status(phase,**extra):
        changed=[str(p.relative_to(ROOT)) for p,content in original_sources.items() if p.read_bytes()!=content]
        if changed: raise RuntimeError('Source content changed during experiment: '+', '.join(changed))
        atomic_write_json(report/'status.json',dict(phase=phase,elapsed_seconds=time.time()-started,**extra))
        print('aligned_experiment_phase='+phase,flush=True)
    try:
        status('deployed_baseline_evaluation')
        baseline=evaluate_tracked_checkpoint(config.initial_checkpoint,episodes=200,workers=9,
            seed=config.evaluation_seed,output_dir=str(runs[1]),pixel_guard='receding',search_workers=1)
        status('aligned_training',baseline_success_at_limit=baseline['success_at_limit'])
        train_tracked_policy(config)
        status('continuation_training')
        torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'tools/train_continuation_value.py'),
            '--source-run',str(runs[0]),'--output',str(runs[2]),'--max-states','160',
            '--workers','9','--states-per-batch','4','--search-workers','1'],check=True,cwd=ROOT)
        aligned=json.loads((runs[0]/'best_summary.json').read_text())
        continuation=json.loads((runs[2]/'evaluation/evaluation_summary.json').read_text())
        scores=[baseline['success_at_limit'],aligned['success_at_limit'],continuation['success_at_limit']]
        # Stable maximum retains the earlier checkpoint for exactly tied scores.
        selected=max(range(3),key=lambda i:scores[i])
        checkpoints=[config.initial_checkpoint,str(runs[0]/'best.pt'),str(runs[2]/'candidate.pt')]
        baseline_csv=runs[1]/'evaluation_episodes.csv'
        paired=dict(aligned=paired_failures(baseline_csv,
                runs[0]/'round1/evaluation/evaluation_episodes.csv',config.evaluation_episode_limit_seconds),
            continuation=paired_failures(baseline_csv,
                runs[2]/'evaluation/evaluation_episodes.csv',config.evaluation_episode_limit_seconds))
        atomic_write_json(report/'comparison.json',dict(deployed_baseline=baseline,aligned=aligned,
            continuation=continuation,selected_checkpoint=checkpoints[selected],
            paired_failures=paired,
            selected_success_at_limit=scores[selected],acceptance_met=scores[selected]==1.,
            selection_metric='success_at_limit',earlier_retained_on_tie=True))
        status('complete',success_at_limit=scores,selected_checkpoint=checkpoints[selected])
    except BaseException as error:
        atomic_write_json(report/'status.json',dict(phase='failed',elapsed_seconds=time.time()-started,error=str(error)))
        raise

if __name__=='__main__':main()
