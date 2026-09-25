"""One bounded candidate run against the identical saved baseline; no queues."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import argparse,csv,json,time
import torch
from experiment import install_candidate,CHECKPOINT
from barrage_rl import deployment
from barrage_rl.evaluate_tracked_policy import evaluate_tracked_checkpoint
from barrage_rl.artifacts import contents_equal,atomic_write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke-test',action='store_true',required=True);p.parse_args()
    design=json.loads((HERE/'design.json').read_text());seeds=design['paired_seeds']
    assert len(seeds)==len(set(seeds))==20
    output=HERE/'candidate'
    if output.exists():raise FileExistsError(output)
    previous=ROOT/'diagnostics/planner_small_regression_20260921/baseline'
    snapshot=ROOT/'diagnostics/planner_consistency_20260921/source_snapshot'
    for name in design['source_files']:
        assert (ROOT/name).read_bytes()==(snapshot/name).read_bytes(),name
    old_model=torch.load(previous/'evaluated_model.pt',map_location='cpu',weights_only=False)
    model=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert contents_equal(old_model['model'],model['model'])
    config=json.loads((previous/'evaluation_config.json').read_text())
    assert config['episode_seeds']==seeds
    before={p:p.read_bytes() for p in [CHECKPOINT,*(ROOT/n for n in design['source_files']),
        HERE/'candidate_controller.py',HERE/'readiness_kernel.py',HERE/'experiment.py']}
    with (previous/'evaluation_episodes.csv').open(newline='') as f:baseline={int(r['seed']):r for r in csv.DictReader(f)}
    original=deployment.configure_image_controller
    def configure(agent,*args,**kwargs):
        config=original(agent,*args,**kwargs);guard=install_candidate(agent)
        return {**config,'experimental_candidate':guard.manifest(),'production_default_changed':False}
    deployment.configure_image_controller=configure
    torch.set_num_threads(1);started=time.perf_counter()
    print('Reused baseline validated by source/model contents and seeds; starting exactly 20 candidate episodes',flush=True)
    evaluate_tracked_checkpoint(str(CHECKPOINT),episodes=20,episode_seeds=seeds,workers=9,
        evaluation_batch_size=20,output_dir=str(output),device_name='cuda',smoke_test=True,
        episode_limit_seconds=120,bullet_count=300,targeted_bullet_probability=.10,
        rendered_rgb=True,causal_action_delay_steps=0,analytic_shield=False,pixel_guard='receding',search_workers=9)
    assert all(p.read_bytes()==b for p,b in before.items())
    with (output/'evaluation_episodes.csv').open(newline='') as f:candidate={int(r['seed']):r for r in csv.DictReader(f)}
    rows=[]
    for seed in seeds:
        b=baseline[seed];c=candidate[seed];bs=b['termination_reason']=='time_limit';cs=c['termination_reason']=='time_limit'
        rows.append(dict(seed=seed,baseline_seconds=float(b['model_survival_seconds']),candidate_seconds=float(c['model_survival_seconds']),
            baseline_success=bs,candidate_success=cs,outcome='rescued' if cs and not bs else 'regression' if bs and not cs else 'both_success' if bs else 'both_failure'))
    result=dict(new_complete_episode_executions=20,reused_baseline_episodes=20,source_model_unchanged=True,
        baseline_success_count=sum(r['baseline_success'] for r in rows),candidate_success_count=sum(r['candidate_success'] for r in rows),
        rescued=[r['seed'] for r in rows if r['outcome']=='rescued'],new_failures=[r['seed'] for r in rows if r['outcome']=='regression'],
        retained_failures=[r['seed'] for r in rows if r['outcome']=='both_failure'],rows=rows,
        elapsed_seconds=time.perf_counter()-started,formal_evaluation=False)
    atomic_write_json(HERE/'paired_summary.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
