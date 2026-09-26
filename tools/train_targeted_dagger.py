"""Prepared, explicit-start DAgger run using the isolated planner candidate."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from dataclasses import asdict,dataclass
import importlib.util
import json
import types
from barrage_rl import deployment
from barrage_rl import train_tracked_policy as training
from barrage_rl import evaluate_tracked_policy as evaluation

BUNDLE=ROOT/'diagnostics/dagger_ready_20260921'
PLANNER=ROOT/'diagnostics/planner_readiness_20260921'
CONTROLLER='planner_readiness_20260921_parallel'
EXTRA_SOURCES=(
    'tools/train_targeted_dagger.py',
    'tools/commit_safe_ranking.py',
    'diagnostics/planner_readiness_20260921/candidate_parallel.py',
    'diagnostics/planner_readiness_20260921/readiness_kernel_parallel.py',
    'barrage_rl/deployment.py', 'barrage_rl/artifacts.py', 'barrage_rl/timing.py',
    'barrage_rl/metrics.py', 'barrage_rl/plot.py',
    'tools/pixel_guard_receding.py', 'tools/pixel_guard_continuation.py',
    'tools/pixel_guard_refined.py', 'tools/pixel_guard_candidate.py',
    'tools/pixel_search_kernel.py', 'tools/pixel_receding_kernel.py',
    'image/plane(0).gif', 'image/bullet(5).gif',
)

@dataclass
class TargetedConfig(training.TrackedDAggerConfig):
    initial_checkpoint: str = "best.pt"
    experimental_controller: str = CONTROLLER
    feature_expected_bullet_count: int = 250

def load_config():
    data=json.loads((BUNDLE/'training_config.json').read_text(encoding='utf-8'))
    for name in ('collection_seed_list','safety_horizons'):data[name]=tuple(data[name])
    return TargetedConfig(**data)

def validate_config(config):
    if config.experimental_controller!=CONTROLLER:raise ValueError('Unknown experimental controller')
    if config.smoke_test or config.resume:raise ValueError('Prepared training requires a fresh formal run')
    if config.evaluation_episodes!=200:raise ValueError('Formal evaluation requires exactly 200 episodes')
    if config.selection_mode!='success_at_limit':raise ValueError('Selection must use success_at_limit')
    if config.bullet_count!=300 or config.evaluation_bullet_count!=300:raise ValueError('Expected 300 bullets')
    if config.targeted_bullet_probability!=.10 or config.teacher_reaction_seconds!=.10:raise ValueError('Task probabilities/reaction changed')
    if config.max_objects!=384 or config.tracker_capacity!=384:raise ValueError('Expected 384 slots')
    if not config.deployment_rgb_observation or config.pixel_guard!='receding':raise ValueError('Expected RGB receding controller')
    if config.collection_causal_action_delay_steps or config.evaluation_causal_action_delay_steps:raise ValueError('Expected zero delay')
    if config.samples_per_round<config.num_envs*3600:raise ValueError('Collection must cover complete episodes')
    if config.initial_replay or config.resume_replay:raise ValueError('Do not reuse legacy replay')
    output=ROOT/config.output_dir
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Output is non-empty: {output}')
    if not (ROOT/config.initial_checkpoint).is_file():raise FileNotFoundError(config.initial_checkpoint)
    collection=training._collection_seed_footprint(config,config.collection_seed)
    heldout=training._evaluation_seed_footprint(config,config.evaluation_seed)
    if collection & heldout:raise ValueError('Training and held-out seeds overlap')
    if config.feature_expected_bullet_count!=250:raise ValueError('Preserve the evaluated warm-start feature normalization')

def prepared_source_files():
    """Cover the current training dependencies and the validation programs."""
    return tuple(dict.fromkeys((*training._MANIFEST_SOURCE_FILES, *EXTRA_SOURCES,
        *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'barrage_rl').glob('*.py'))),
        *(f'diagnostics/dagger_ready_20260921/{name}' for name in
          ('training_config.json', 'start_training.ps1', 'prepare.py', 'preflight.py', 'bootstrap_check.py')))))


def validate_launch(config, *, evidence_dir=None):
    evidence = BUNDLE if evidence_dir is None else Path(evidence_dir)
    validate_config(config)
    approval=json.loads((evidence/'preflight.json').read_text(encoding='utf-8'))
    bootstrap=json.loads((evidence/'bootstrap_check.json').read_text(encoding='utf-8'))
    if not all(r.get('passed') and r.get('optimizer_steps')==0 for r in (approval,bootstrap)):
        raise RuntimeError('Prepared tests have not passed')
    required = set(prepared_source_files())
    if any(set(record.get('source_files', ())) != required for record in (approval, bootstrap)):
        raise RuntimeError('Preflight source coverage changed; revalidate before training')
    for name in bootstrap['source_files']:
        if (ROOT/name).read_bytes()!=(evidence/'launch_source_snapshot'/name).read_bytes():
            raise RuntimeError(f'Code or configuration changed since preflight: {name}; revalidate before training')
    import torch
    from barrage_rl.artifacts import contents_equal
    current=torch.load(ROOT/config.initial_checkpoint,map_location='cpu',weights_only=False)
    reference=torch.load(evidence/'initial_model_reference.pt',map_location='cpu',weights_only=False)
    from barrage_rl.checkpoint_loader import validate_model_version
    validate_model_version(current)
    for key in ('model', 'model_version', 'model_hparams', 'tracked_policy_spec', 'experimental_controller'):
        if not contents_equal(current.get(key),reference.get(key)):
            raise RuntimeError(f'Warm-start {key} changed since preflight')

def install_runtime():
    """Install only in this explicitly invoked process; deployment default stays unchanged."""
    if getattr(deployment.configure_image_controller,'_targeted_dagger',False):return
    prepared=load_config()
    original_warm_start=training._validate_plain_dagger_warm_start
    def validate_warm_start(checkpoint, checkpoint_path):
        if checkpoint.get('experimental_controller') != CONTROLLER:
            raise ValueError('Prepared warm-start requires its matching experimental controller')
        # This entry installs the matching planner; every other composite marker
        # must still pass the generic trainer's rejection checks.
        original_warm_start({key:value for key,value in checkpoint.items()
                             if key != 'experimental_controller'}, checkpoint_path)
    training._validate_plain_dagger_warm_start=validate_warm_start
    original_spec=training.TrackedPolicySpec
    def tracked_spec(*args,**kwargs):
        # Retain the checkpoint's feature denominator while the environment
        # task stays at 300 bullets. Return the original dataclass, not a new
        # model architecture or a new observation channel.
        kwargs['expected_bullet_count']=prepared.feature_expected_bullet_count
        return original_spec(*args,**kwargs)
    training.TrackedPolicySpec=tracked_spec
    sys.path.insert(0,str(PLANNER))
    spec=importlib.util.spec_from_file_location('targeted_parallel_controller',PLANNER/'candidate_parallel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=deployment.configure_image_controller
    def configure(agent,kind='receding',*,search_workers=9):
        if kind!='receding':raise ValueError('Targeted candidate requires receding planning')
        config=original(agent,kind,search_workers=search_workers)
        guard=agent._receding_pixel_guard
        for name in ('apply','_assess','_geometry'):
            setattr(guard,name,types.MethodType(getattr(module.ContinuationMixin,name),guard))
        module.warmup_parallel_controller(guard)
        return {**config,'experimental_controller':CONTROLLER,'source_files':list(EXTRA_SOURCES),
            'ranking_variant':'commit_safe_ranking',
            'ranking_safety_basis':'nominal_pixel',
            'feature_expected_bullet_count':prepared.feature_expected_bullet_count,
            'uncertainty_semantics':'observed-track sensitivity; excludes future births and association errors'}
    configure._targeted_dagger=True
    deployment.configure_image_controller=configure
    original_load=evaluation.load_tracked_agent
    def load(*args,**kwargs):
        kwargs['experimental_controller']=CONTROLLER
        return original_load(*args,**kwargs)
    evaluation.load_tracked_agent=load
    original_checkpoint=training._checkpoint
    def checkpoint(*args,**kwargs):
        result=original_checkpoint(*args,**kwargs)
        result['experimental_controller']=CONTROLLER
        return result
    training._checkpoint=checkpoint
    training._MANIFEST_SOURCE_FILES=tuple(dict.fromkeys((*training._MANIFEST_SOURCE_FILES,*EXTRA_SOURCES)))
    original_write=training.atomic_write_json
    def write(path,data):
        if Path(path).name=='run_manifest.json':
            data={**data,'experimental_controller':CONTROLLER}
            snapshot=Path(path).parent/'source_snapshot'
            for name in training._MANIFEST_SOURCE_FILES:
                source=ROOT/name;destination=snapshot/name
                if not destination.exists():
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with destination.open('xb') as f:f.write(source.read_bytes())
                elif destination.read_bytes()!=source.read_bytes():
                    raise RuntimeError(f'Run source changed: {name}')
        return original_write(path,data)
    training.atomic_write_json=write

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true',help='Print prepared configuration; never start training')
    mode.add_argument('--train',action='store_true',help='Explicitly start the prepared formal DAgger run')
    mode.add_argument('--evaluate',metavar='CHECKPOINT',help='Evaluate with the same experimental controller, fixed 200 episodes')
    args=parser.parse_args();os.chdir(ROOT)
    config=load_config()
    if args.evaluate:
        install_runtime()
        evaluation.evaluate_tracked_checkpoint(args.evaluate,episodes=200,workers=config.evaluation_workers,
            seed=config.evaluation_seed,output_dir=str(ROOT/'runs/visual_set_v52_eval'),device_name=config.device,
            bullet_count=300,targeted_bullet_probability=.10,episode_limit_seconds=120,rendered_rgb=True,
            evaluation_batch_size=config.evaluation_batch_size,pixel_guard='receding',search_workers=config.search_workers)
        return
    validate_config(config)
    if not args.train:
        print(json.dumps(asdict(config),indent=2,ensure_ascii=False));return
    validate_launch(config)
    install_runtime()
    print('Starting one DAgger round: initial fixed-200 evaluation, collection, optimization, fixed-200 round evaluation.',flush=True)
    training.train_tracked_policy(config)

if __name__=='__main__':main()
