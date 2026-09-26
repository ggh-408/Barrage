"""Check the actual warm-start construction without calling training or taking optimizer steps."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,json,argparse
from pathlib import Path
from dataclasses import asdict
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tools import train_targeted_dagger as entry
from barrage_rl import train_tracked_policy as training
from barrage_rl.artifacts import contents_equal,atomic_write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.resolve()
    target=output/'bootstrap_check.json'
    if target.exists():raise FileExistsError(target)
    tested=json.loads((output/'preflight.json').read_text())
    assert tested['passed'] and tested['optimizer_steps']==0
    config=entry.load_config();entry.validate_config(config)
    def forbidden(*args,**kwargs):raise RuntimeError('Optimizer updates are forbidden during bootstrap checks')
    torch.optim.AdamW.step=forbidden
    for name in tested['source_files']:
        assert (ROOT/name).read_bytes()==(output/'launch_source_snapshot'/name).read_bytes()
    entry.install_runtime()
    # Mirror the existing trainer's constructor arguments and migration helper.
    spec=training.TrackedPolicySpec(max_objects=config.max_objects,object_features=config.object_features,
        global_features=config.global_features,tracker_capacity=config.tracker_capacity,expected_bullet_count=config.bullet_count)
    model=training.ActionQueryPolicy(spec,width=config.model_width,attention_layers=config.attention_layers,
        attention_heads=config.attention_heads,safety_horizons=config.safety_horizons)
    source=torch.load(ROOT/config.initial_checkpoint,map_location='cpu',weights_only=False)
    original=torch.load(output/'untrained_probe.pt',map_location='cpu',weights_only=False)
    training.validate_model_version(source)
    training._validate_plain_dagger_warm_start(source,config.initial_checkpoint)
    migration=training._load_migrated_model_state(model,source['model'])
    assert not migration['expanded'] and not migration['skipped']
    assert contents_equal(original['model'],model.state_dict())
    assert asdict(spec)==original['tracked_policy_spec']
    optimizer=torch.optim.AdamW(training._configure_trainable_scope(model,config.trainable_scope),lr=config.learning_rate,weight_decay=config.weight_decay)
    assert not optimizer.state
    result_checkpoint=training._checkpoint(model,optimizer,config,spec,0,{})
    assert result_checkpoint['experimental_controller']==entry.CONTROLLER
    assert result_checkpoint['tracked_policy_spec']==original['tracked_policy_spec']
    assert result_checkpoint['config']['feature_expected_bullet_count']==250
    assert result_checkpoint['config']['bullet_count']==300
    # Verify manifest source copying into a diagnostic fixture, never into runs.
    manifest_path=output/'manifest_fixture/run_manifest.json'
    training.atomic_write_json(manifest_path,dict(source_files=list(training._MANIFEST_SOURCE_FILES)))
    recorded=json.loads(manifest_path.read_text())
    assert recorded['experimental_controller']==entry.CONTROLLER
    assert all((manifest_path.parent/'source_snapshot'/name).read_bytes()==(ROOT/name).read_bytes() for name in recorded['source_files'])
    files=entry.prepared_source_files()
    assert set(files)==set(tested['source_files'])
    result=dict(initial_checkpoint=config.initial_checkpoint,passed=True,optimizer_steps=0,real_training_function_called=False,source_files=list(files),
        bootstrap_model_exactly_matches_evaluated_model=True,bootstrap_features_exactly_match_evaluated_spec=True,
        migration_exact_tensors=len(migration['exact']),migration_expanded=0,migration_skipped=0,
        feature_expected_bullet_count=250,actual_environment_bullets=300,fresh_optimizer_empty=True,
        checkpoint_marker_preserved=True,manifest_controller_and_sources_recorded=True,
        training_output_created=(ROOT/config.output_dir).exists(),complete_evaluation_episodes=0)
    atomic_write_json(target,result)
    entry.validate_launch(config,evidence_dir=output)
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'}))

if __name__=='__main__':main()
