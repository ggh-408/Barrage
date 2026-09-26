"""Write a launchable training plan without collecting or optimizing."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json
from pathlib import Path
from dataclasses import asdict
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from tools.train_targeted_dagger import TargetedConfig,validate_config
from barrage_rl.train_tracked_policy import _resolve_run_seeds

def main():
    previous=json.loads((ROOT/'diagnostics/planner_readiness_20260921/design.json').read_text())
    config=TargetedConfig(output_dir='runs/visual_set_v52',rounds=1,samples_per_round=144000,
        replay_capacity=160000,epochs_per_round=2,batch_size=512,num_envs=36,cpu_workers=9,
        search_workers=9,evaluation_workers=9,evaluation_batch_size=36,learning_rate=1e-5,
        priority_mode='behavior_regret',collection_seed_list=tuple(previous['known_failure_seeds']),
        repeat_collection_seeds=True,evaluate_initial_checkpoint=True,
        bootstrap_with_teacher_behavior=False,random_action_probability=0.0)
    _resolve_run_seeds(config);validate_config(config)
    with (HERE/'training_config.json').open('x',encoding='utf-8') as f:json.dump(asdict(config),f,indent=2)
    print('Prepared one round; 9 hard-seed slots plus 27 fresh-seed slots; training was not started.')

if __name__=='__main__':main()
