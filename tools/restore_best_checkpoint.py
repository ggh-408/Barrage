"""Rebuild the deployment checkpoint from retained model and configuration evidence."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from barrage_rl.artifacts import atomic_torch_save, atomic_write_json, contents_equal
from barrage_rl.checkpoint_loader import load_tracked_agent
from barrage_rl.live_screen import LiveVisualController
from barrage_rl.window_runtime import CONTROLLER, configure_window_controller
from tools.evaluate_targeted_dagger import exclusion_sets


def main():
    folder = ROOT / 'diagnostics/best_recovery_20260925'
    folder.mkdir(exist_ok=False)
    long_dir = ROOT / 'diagnostics/targeted_dagger_fresh_20260924_101738_193311'
    short_dir = ROOT / 'diagnostics/targeted_dagger_fresh_20260924_210036_920113'
    model_path = long_dir / 'evaluation/evaluated_model.pt'
    template_path = ROOT / 'diagnostics/dagger_ready_20260921/untrained_probe.pt'
    load = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    saved = load(model_path)
    short = load(short_dir / 'evaluation/evaluated_model.pt')
    assert contents_equal(saved['model'], short['model'])
    assert contents_equal(saved['model'], load(ROOT / 'best.pt')['model'])
    template = load(template_path)
    manifest = json.loads((long_dir / 'experiment_manifest.json').read_text())
    evaluation = json.loads((long_dir / 'evaluation/evaluation_config.json').read_text())
    original = json.loads((short_dir / 'experiment_manifest.json').read_text())
    checkpoint = {key: copy.deepcopy(template[key]) for key in (
        'tracked_policy_spec', 'model_hparams', 'observation_size',
        'inference_head', 'policy_architecture')}
    checkpoint.update(model=saved['model'], model_version=saved['model_version'],
                      experimental_controller=CONTROLLER, round=manifest['checkpoint_round'])
    config = copy.deepcopy(template['config'])
    # Recovered from the exact seed exclusion set recorded by the original best.pt short test.
    config.update(output_dir='runs/visual_set_v53', collection_seed=841322446,
                  evaluation_seed=1722311744)
    train, heldout = exclusion_sets(config)
    assert len(train) == original['excluded_training_seed_count']
    assert len(heldout) == original['excluded_heldout_seed_count']
    assert sorted(train | heldout) == original['excluded_seeds']
    assert config['bullet_count'] == evaluation['bullet_count'] == 300
    assert config['targeted_bullet_probability'] == evaluation['targeted_bullet_probability'] == .10
    assert config['evaluation_causal_action_delay_steps'] == evaluation['causal_action_delay_steps'] == 0
    assert checkpoint['tracked_policy_spec']['expected_bullet_count'] == evaluation['controller']['feature_expected_bullet_count']
    assert evaluation['controller']['experimental_controller'] == CONTROLLER
    checkpoint['config'] = config
    checkpoint['recovery'] = {
        'kind': 'deployment_metadata_reconstruction',
        'model_source': str(model_path.relative_to(ROOT)),
        'metadata_template': str(template_path.relative_to(ROOT)),
        'seed_evidence': str((short_dir / 'experiment_manifest.json').relative_to(ROOT)),
        'original_checkpoint': manifest['checkpoint'],
        'optimizer_recovered': False, 'exact_training_resume_available': False,
        'training_config_note': 'Inherited preparation settings; v53 output path and seed exclusions recovered from evaluation evidence. Other training-only settings are not verified as the original v53 configuration.',
    }
    candidate = folder / 'restored_candidate.pt'
    atomic_torch_save(checkpoint, candidate)
    agent, spec, metadata = load_tracked_agent(str(candidate), torch.device('cpu'),
                                               experimental_controller=CONTROLLER)
    assert contents_equal(agent.model.state_dict(), saved['model'])
    torch.set_num_threads(1)
    controller = LiveVisualController(str(candidate), device_name='cpu',
                                     experimental_controller=CONTROLLER)
    runtime = configure_window_controller(controller.agent, search_workers=1)
    assert runtime['ranking_safety_basis'] == 'nominal_pixel'
    backup = folder / 'best_parameters_before_recovery.pt'
    shutil.copyfile(ROOT / 'best.pt', backup)
    atomic_torch_save(checkpoint, ROOT / 'best.pt')
    assert contents_equal(load(ROOT / 'best.pt')['model'], saved['model'])
    atomic_write_json(folder / 'recovery_report.json', {
        **checkpoint['recovery'], 'model_tensors_equal_long_test': True,
        'model_tensors_equal_original_best_short_test': True,
        'original_seed_exclusions_match': True, 'strict_model_load_passed': True,
        'window_controller_load_passed': True, 'window_runtime': runtime,
        'restored_checkpoint': str(ROOT / 'best.pt'), 'backup': str(backup),
    })
    print('Restored best.pt: exact model tensors, validated seed exclusions, strict model and window controller loading passed.')


if __name__ == '__main__':
    main()
