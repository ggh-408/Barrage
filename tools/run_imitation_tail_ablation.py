"""Paired replay ablation of teacher target semantics and fixed-200 evaluation."""
from __future__ import annotations

import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from barrage_rl import train_tracked_policy as training
from barrage_rl.artifacts import atomic_torch_save, atomic_write_json, atomic_copy, prepare_new_output
from barrage_rl.evaluate_tracked_policy import evaluate_tracked_checkpoint
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from tools.run_aligned_experiment import paired_failures

REPORT = ROOT / 'diagnostics/imitation_tail_audit_20260920'
SOURCE = ROOT / 'runs/visual_set_v49'
OUTPUT = ROOT / 'runs/visual_set_v51'


def legacy_target(regrets, collisions, temperature):
    logits = -regrets / max(float(temperature), 1e-4)
    unsafe = collisions[:, 0].bool()
    masked = unsafe & (~unsafe).any(dim=1, keepdim=True)
    logits = torch.where(masked, torch.full_like(logits, -80.), logits)
    return torch.nn.functional.softmax(logits, dim=1)


def train(continue_preflight=False, candidate_only=False):
    if continue_preflight:
        # Continue only the preflight created by this experiment. Never replace
        # an existing candidate or an unrelated run.
        existing=json.loads((OUTPUT/'config.json').read_text())
        if existing.get('training_stage')!='paired_replay_target_ablation' or set(
                p.name for p in OUTPUT.iterdir()) != {'config.json'}:
            raise ValueError('Only the unfinished preflight can be continued')
    else:
        prepare_new_output(OUTPUT)
    settings = json.loads((SOURCE / 'config.json').read_text())
    if settings.pop('physics_fps') != 120:
        raise ValueError('The paired replay must use the current 120 Hz timebase')
    settings['output_dir'] = str(OUTPUT)
    config = training.TrackedDAggerConfig(**settings)
    config.safety_horizons = tuple(config.safety_horizons)
    spec = TrackedPolicySpec(expected_bullet_count=300)
    replay = training.TrackedReplay.load(SOURCE / 'replay_latest.npz', config.replay_capacity,
                                         spec, len(config.safety_horizons))
    initial = torch.load(config.initial_checkpoint, map_location='cpu', weights_only=False)
    reference = torch.load(SOURCE / 'best.pt', map_location='cpu', weights_only=False)
    device = torch.device(config.device)
    current_target = training._teacher_distribution
    if not continue_preflight:
        atomic_write_json(OUTPUT / 'config.json', {**asdict(config),
            'training_stage': 'paired_replay_target_ablation',
            'replay_source': str(SOURCE / 'replay_latest.npz'),
            'new_collection_performed': False})
    report = json.loads((REPORT/'training_ablation.json').read_text()) if candidate_only else dict(source_run=str(SOURCE), replay=str(SOURCE / 'replay_latest.npz'),
        evaluation_seed=config.evaluation_seed, changed_factor='policy teacher distribution only',
        unchanged=['replay', 'episode split', 'priorities', 'initial weights', 'risk labels',
                   'optimizer', 'training seed', 'epochs', 'controller', 'task', 'threshold'])
    for variant, target in [('legacy', legacy_target), ('continuation_cost', current_target)]:
        if candidate_only and variant=='legacy': continue
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        model = ActionQueryPolicy(spec, width=config.model_width,
            attention_layers=config.attention_layers, attention_heads=config.attention_heads,
            safety_horizons=config.safety_horizons).to(device)
        training._load_migrated_model_state(model, initial['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        training._teacher_distribution = target
        started = time.perf_counter()
        try:
            _, metrics = training._train_round(model, optimizer, replay, config, device, 1)
        finally:
            training._teacher_distribution = current_target
        elapsed = time.perf_counter() - started
        differences = [name for name, value in model.state_dict().items()
                       if not torch.equal(value.cpu(), reference['model'][name].cpu())]
        report[variant] = dict(seconds=elapsed, metrics=metrics,
            reference_tensors_different=len(differences), reference_tensors_equal=not differences,
            maximum_parameter_difference=max(float((value.cpu()-reference['model'][name].cpu()).abs().max())
                for name,value in model.state_dict().items()),
            reference_tensors_close=all(torch.allclose(value.cpu(),reference['model'][name].cpu(),
                rtol=0.,atol=1e-6) for name,value in model.state_dict().items()))
        if variant == 'legacy':
            atomic_torch_save(training._checkpoint(model, optimizer, config, spec, 1, metrics),
                              REPORT / 'legacy_reproduction.pt')
        if variant == 'legacy' and not report[variant]['reference_tensors_equal']:
            report['reproduction_limitation']='Floating point parameter differences prevent exact historical reproduction. Saved historical scores remain historical references, not scores of this reproduced checkpoint.'
        if variant == 'continuation_cost':
            candidate=training._checkpoint(model, optimizer, config, spec, 1, metrics)
            candidate['config'].update(training_stage='paired_replay_target_ablation',
                initial_replay=str(SOURCE/'replay_latest.npz'),new_collection_performed=False)
            atomic_torch_save(candidate, OUTPUT / 'candidate.pt')
        atomic_write_json(REPORT / 'training_ablation.json', report)
        print(json.dumps({variant: report[variant]}), flush=True)
        del optimizer, model
        torch.cuda.empty_cache()


def evaluate():
    training_report = json.loads((REPORT / 'training_ablation.json').read_text())
    if 'continuation_cost' not in training_report:
        raise RuntimeError('Complete candidate training is required')
    cfg = json.loads((OUTPUT / 'config.json').read_text())
    frozen_paths=list((ROOT/'barrage_rl').glob('*.py'))+list((ROOT/'tools').glob('pixel*.py'))
    frozen={path:path.read_bytes() for path in frozen_paths}
    performance_path=REPORT/'inference_performance.json'
    search_workers=(json.loads(performance_path.read_text())['selected_search_workers']
                    if performance_path.is_file() else 1)
    summary = evaluate_tracked_checkpoint(str(OUTPUT / 'candidate.pt'), episodes=200,
        workers=9, seed=cfg['evaluation_seed'], output_dir=str(OUTPUT / 'evaluation'),
        pixel_guard='receding', search_workers=search_workers, evaluation_batch_size=10)
    changed=[str(path.relative_to(ROOT)) for path,content in frozen.items() if path.read_bytes()!=content]
    if changed: raise RuntimeError('Sources changed during evaluation: '+', '.join(changed))
    baseline = json.loads((SOURCE / 'best_summary.json').read_text())
    deployed = ROOT / 'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    deployed_summary = json.loads((ROOT / 'runs/visual_set_v49_eval/evaluation_summary.json').read_text())
    promote = summary['success_at_limit'] > deployed_summary['success_at_limit']
    atomic_copy(OUTPUT / 'candidate.pt' if promote else deployed, OUTPUT / 'best.pt')
    atomic_write_json(OUTPUT / 'best_summary.json', summary if promote else deployed_summary)
    comparison = dict(legacy=baseline['success_at_limit'], candidate=summary['success_at_limit'],
        deployed=deployed_summary['success_at_limit'],
        paired_legacy=paired_failures(SOURCE / 'round1/evaluation/evaluation_episodes.csv',
            OUTPUT / 'evaluation/evaluation_episodes.csv', 120.),
        paired_deployed=paired_failures(ROOT / 'runs/visual_set_v49_eval/evaluation_episodes.csv',
            OUTPUT / 'evaluation/evaluation_episodes.csv', 120.),
        candidate_selected=promote, selection_metric='success_at_limit',
        earlier_retained_on_tie=True, game_default_changed=False,
        legacy_reproduction_exact=training_report['legacy']['reference_tensors_equal'],
        source_files=[str(p.relative_to(ROOT)) for p in frozen_paths],source_files_unchanged=True,
        search_workers=search_workers,
        acceptance_met=summary['success_at_limit'] == 1.)
    atomic_write_json(REPORT / 'evaluation_comparison.json', comparison)
    # Use the canonical dashboard without changing panel content or styling.
    from barrage_rl.plot import save_round_summary_plot
    history=[{**deployed_summary, 'round':0., 'phase':'deployed_baseline'},
             {**summary, 'round':1., 'phase':'teacher_target_repair'}]
    training._write_history(OUTPUT / 'round_summaries.csv', history)
    save_round_summary_plot(OUTPUT / 'round_summaries.csv', OUTPUT / 'results.png', OUTPUT / 'config.json')
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'continue-preflight', 'continue-candidate', 'evaluate'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase=='evaluate': evaluate()
    else: train(continue_preflight=args.phase.startswith('continue-'),
                candidate_only=args.phase=='continue-candidate')
