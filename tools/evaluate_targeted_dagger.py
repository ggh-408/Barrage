"""Supplemental regression/fresh-seed evaluation with the trained experimental planner."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,csv,random,time
from datetime import datetime
from dataclasses import replace,asdict
import torch
from barrage_rl import evaluate_tracked_policy as evaluation
from barrage_rl import train_tracked_policy as training
from barrage_rl.artifacts import atomic_write_json,prepare_new_output
from tools import train_targeted_dagger as entry

OPTIMIZED_SOURCES = (
    'barrage_rl/window_runtime.py', 'barrage_rl/window_planner.py',
    'barrage_rl/window_planner_kernel.py', 'barrage_rl/window_pixel_geometry.py',
    'barrage_rl/window_tracker.py', 'barrage_rl/window_tracker_kernel.py',
)


def install_evaluation_runtime(search_depth=15, implementation='reference'):
    """Use the shared 0.5-second default; explicit historical overrides stay local."""
    from barrage_rl import deployment
    if implementation not in ('reference','optimized'):
        raise ValueError('Unknown evaluation runtime implementation')
    entry.install_runtime()
    original=deployment.configure_image_controller
    def configure(agent,kind='receding',*,search_workers=9):
        if implementation=='optimized':
            from barrage_rl.window_runtime import configure_window_controller
            if kind!='receding' or agent.model.continuation_head is not None:
                raise ValueError('Optimized evaluation requires the current policy without a continuation head; use --runtime reference')
            result=configure_window_controller(agent,search_workers=search_workers)
            result={**result,'source_files':list(OPTIMIZED_SOURCES),
                    'feature_expected_bullet_count':agent.model.spec.expected_bullet_count}
        else:
            result=original(agent,kind,search_workers=search_workers)
        guard=agent._receding_pixel_guard
        guard.recovery_config=replace(guard.recovery_config,search_depth=search_depth)
        return {**result,'config':asdict(guard.recovery_config),
                'evaluation_runtime':implementation,
                'evaluation_search_depth':search_depth,
                'planning_horizon_seconds':search_depth/30,
                'evaluation_horizon_override':True}
    deployment.configure_image_controller=configure

def csv_seeds(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows or 'seed' not in rows[0]:raise ValueError(f'Missing episode seeds: {path}')
    return [int(r['seed']) for r in rows]

def exclusion_sets(config):
    training_seeds=training._collection_seed_footprint(training.TrackedDAggerConfig(**{
        k:v for k,v in config.items() if k in training.TrackedDAggerConfig.__dataclass_fields__
    }),int(config['collection_seed']))
    heldout=set(range(int(config['evaluation_seed']),int(config['evaluation_seed'])+int(config['evaluation_episodes'])))
    return training_seeds,heldout

def select_seeds(mode,count,random_seed,regression_csv,excluded):
    if mode=='regression':
        pool=csv_seeds(regression_csv)
        if len(pool)!=len(set(pool)):raise ValueError('Regression CSV contains duplicate seeds')
        if count>len(pool):raise ValueError(f'Regression pool contains only {len(pool)} episodes')
        return pool[:count]
    rng=random.Random(random_seed);result=[];seen=set(excluded)
    if count>147000000-len({s for s in seen if 2000000000<=s<2147000000}):raise ValueError('Fresh seed space exhausted')
    while len(result)<count:
        seed=rng.randrange(2000000000,2147000000)
        if seed not in seen:seen.add(seed);result.append(seed)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'best.pt')
    p.add_argument('--mode',choices=('fresh','regression'),default='fresh')
    p.add_argument('--episodes',type=int,help='Fresh default 3000; regression default full CSV pool')
    p.add_argument('--seeds-csv',type=Path,help='Explicit regression pool with a seed column')
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--runtime',choices=('optimized','reference'),default='optimized',
                   help='Optimized window planner/tracker (default); reference restores the historical implementation')
    p.add_argument('--workers',type=int,default=None,help='Default: 10 optimized, 9 reference')
    p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--search-workers',type=int,default=9)
    p.add_argument('--search-depth',type=int,choices=(15,18),default=15,
                   help='Planner horizon: 15 decisions = 0.5s (default); 18 = previous 0.6s')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--smoke-episodes',type=int,default=2)
    p.add_argument('--smoke-limit-seconds',type=float,default=5)
    a=p.parse_args()
    if a.workers is None:a.workers=10 if a.runtime=='optimized' else 9
    if a.mode=='regression' and a.seeds_csv is None:p.error('--mode regression requires --seeds-csv')
    count=a.smoke_episodes if a.smoke_test else a.episodes
    if count is None:count=3000 if a.mode=='fresh' else len(csv_seeds(a.seeds_csv))
    if min(count,a.workers,a.batch_size,a.search_workers)<=0:p.error('Counts must be positive')
    if a.smoke_test and count>50:p.error('Smoke tests are limited to 50 episodes')
    if not 0<a.smoke_limit_seconds<=120:p.error('Smoke limit must be in (0,120]')
    checkpoint=a.checkpoint.resolve()
    checkpoint_bytes=checkpoint.read_bytes()
    metadata=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if metadata.get('experimental_controller')!=entry.CONTROLLER:raise ValueError('Checkpoint must belong to the prepared targeted DAgger controller')
    config=metadata['config'];prepared=entry.load_config()
    if metadata['tracked_policy_spec']['expected_bullet_count']!=prepared.feature_expected_bullet_count:raise ValueError('Feature normalization differs from the controller preparation')
    if config['bullet_count']!=300 or config['targeted_bullet_probability']!=.10:raise ValueError('Checkpoint task differs from the current 300-bullet task')
    if any('collision_head' in name for name in metadata['model']):raise ValueError('Retired learned risk head found')
    sources=list(dict.fromkeys((*training._MANIFEST_SOURCE_FILES,*entry.EXTRA_SOURCES)))
    sources.extend(('tools/evaluate_targeted_dagger.py','diagnostics/dagger_ready_20260921/training_config.json'))
    if a.runtime=='optimized':sources.extend(OPTIMIZED_SOURCES)
    before={name:(ROOT/name).read_bytes() for name in sources}
    train_seeds,heldout=exclusion_sets(config)
    excluded=train_seeds|heldout
    seeds=select_seeds(a.mode,count,a.seed,a.seeds_csv,excluded)
    if a.mode=='fresh':assert not set(seeds)&excluded
    output=(a.output or ROOT/'diagnostics'/f'targeted_dagger_{a.mode}_{a.seed}_{datetime.now():%H%M%S_%f}').resolve()
    prepare_new_output(output)
    limit=a.smoke_limit_seconds if a.smoke_test else 120.0
    atomic_write_json(output/'experiment_manifest.json',dict(
        checkpoint=str(checkpoint),checkpoint_round=metadata.get('round'),controller=entry.CONTROLLER,
        historical_evaluation_results_used=False,
        evaluation_search_depth=a.search_depth,planning_horizon_seconds=a.search_depth/30,
        evaluation_runtime=a.runtime,tracker_implementation='window' if a.runtime=='optimized' else 'reference',
        mode=a.mode,episodes=count,episode_seeds=seeds,random_seed=a.seed,source_files=sources,
        smoke_test=a.smoke_test,supplemental_test=not a.smoke_test,formal_evaluation=False,
        checkpoint_selection_performed=False,training_seed_overlap=sorted(set(seeds)&train_seeds),
        heldout_seed_overlap=sorted(set(seeds)&heldout),excluded_seeds=sorted(excluded) if a.mode=='fresh' else [],
        excluded_training_seed_count=len(train_seeds),excluded_heldout_seed_count=len(heldout),
        bullet_count=300,targeted_bullet_probability=.10,evaluation_episode_limit_seconds=limit,
        regression_source=str(a.seeds_csv.resolve()) if a.mode=='regression' else None))
    torch.set_num_threads(1);install_evaluation_runtime(a.search_depth,a.runtime);started=time.perf_counter()
    print(f'Starting {a.mode} {count} episodes, limit={limit:g}s, checkpoint round={metadata.get("round")}',flush=True)
    result=evaluation.evaluate_tracked_checkpoint(str(checkpoint),episodes=count,episode_seeds=seeds,
        seed=a.seed,workers=min(a.workers,count),evaluation_batch_size=min(a.batch_size,count),output_dir=str(output/'evaluation'),
        device_name=a.device,episode_limit_seconds=limit,bullet_count=300,targeted_bullet_probability=.10,
        rendered_rgb=True,causal_action_delay_steps=0,analytic_shield=False,pixel_guard='receding',
        search_workers=a.search_workers,smoke_test=a.smoke_test,supplemental_test=not a.smoke_test,
        tracker_implementation='window' if a.runtime=='optimized' else 'reference')
    assert result['controller']['experimental_controller']==entry.CONTROLLER
    with (output/'evaluation/evaluation_episodes.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert sorted(int(r['seed']) for r in rows)==sorted(seeds)
    assert checkpoint.read_bytes()==checkpoint_bytes,'Checkpoint changed during evaluation'
    assert all((ROOT/name).read_bytes()==contents for name,contents in before.items()),'Sources changed during evaluation'
    failures=[int(r['seed']) for r in rows if r['termination_reason']!='time_limit']
    atomic_write_json(output/'result.json',dict(mode=a.mode,episodes=len(rows),success_at_limit=result['success_at_limit'],
        success_count=len(rows)-len(failures),failure_seeds=failures,smoke_test=a.smoke_test,
        evaluation_episode_limit_seconds=limit,controller=result['controller'],checkpoint_round=metadata.get('round'),
        source_contents_unchanged=True,checkpoint_contents_unchanged=True,elapsed_seconds=time.perf_counter()-started,
        interpretation='Development regression; may include training seeds' if a.mode=='regression' else 'Excluded training and fixed held-out seeds; independent of historical evaluation results'))
    print(f'Completed {len(rows)-len(failures)}/{len(rows)}: {output}',flush=True)

if __name__=='__main__':main()
