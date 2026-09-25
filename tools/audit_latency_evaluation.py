"""Archive direct per-worker evidence at completed formal evaluation boundaries."""
import json
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def preserved_records(output, pattern, live_paths):
    # Compare direct paired contents: equal prediction counts can belong to
    # different batches when Windows reuses a process identifier.
    retained = []
    records = []
    for path in [*sorted((output / 'audit_boundaries').glob('completed_*/' + pattern)), *live_paths]:
        contents = path.read_bytes()
        companion = path.with_name(path.name.replace('prediction_', 'worker_', 1))
        identity = (path.name, contents, companion.read_bytes())
        if identity not in retained:
            retained.append(identity)
            records.append(json.loads(contents))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'diagnostics/combined_latency_20260925/fixed200'
    progress = json.loads((output / 'evaluation/evaluation_progress.json').read_text())
    completed = int(progress['completed_episodes'])
    if args.require_complete:
        assert progress['complete'] and completed == 200, 'Formal evaluation is still incomplete'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert all((ROOT / name).read_bytes() == (output / 'source_snapshot' / name).read_bytes()
               for name in manifest['source_files'])
    velocity_paths = sorted((output / 'worker_checks').glob('worker_*.json'))
    prediction_paths = sorted((output / 'worker_checks').glob('prediction_*.json'))
    velocity = preserved_records(output, 'worker_*.json', velocity_paths)
    prediction = preserved_records(output, 'prediction_*.json', prediction_paths)
    assert all(item['completed'] and item['all_velocities_bitwise_equal'] for item in velocity)
    assert all(item['all_exact'] for item in prediction)
    result = dict(completed_episodes=completed, success_count=progress['success_count'],
        checked_fit_calls=sum(item['calls'] for item in velocity),
        checked_tracks=sum(item['tracks'] for item in velocity),
        checked_hint_calls=sum(item['hint_calls'] for item in prediction),
        reused_predictions=sum(item['reused_predictions'] for item in prediction),
        velocity_worker_records=len(velocity), prediction_worker_records=len(prediction),
        source_contents_unchanged=True)
    if args.require_complete:
        read = lambda name: json.loads((output / name).read_text())
        config = read('evaluation/evaluation_config.json')
        summary = read('evaluation/evaluation_summary.json')
        final = read('result.json')
        combined = read('combined_checks.json')
        with (output / 'evaluation/evaluation_episodes.csv').open(newline='') as stream:
            episodes = list(csv.DictReader(stream))
        seeds = [int(row['seed']) for row in episodes]
        assert len(episodes) == len(set(seeds)) == 200
        assert seeds == manifest['episode_seeds'] == config['episode_seeds']
        assert all(row['termination_reason'] == 'time_limit' and
                   float(row['model_survival_seconds']) == 120. for row in episodes)
        assert summary['success_at_limit'] == final['success_at_limit'] == 1.
        assert not final['smoke_test'] and final['checkpoint_and_sources_unchanged']
        assert config['bullet_count'] == 300 and config['targeted_bullet_probability'] == .10
        assert config['rendered_rgb'] and not config['analytic_shield']
        assert config['causal_action_delay_steps'] == 0
        assert config['controller']['config']['search_depth'] == 15
        assert config['controller']['ranking_safety_basis'] == 'nominal_pixel'
        assert config['controller']['ranking_variant'] == 'commit_safe_ranking'
        assert combined['planner']['all_actions_and_state_exact']
        assert combined['all_prediction_checks_exact']
        live_velocity = [json.loads(path.read_bytes()) for path in velocity_paths]
        live_prediction = [json.loads(path.read_bytes()) for path in prediction_paths]
        # The running evaluator summarizes its current PID-named files. Verify
        # that report against those files, then require complete archived totals.
        assert combined['hint_calls'] == sum(item['hint_calls'] for item in live_prediction)
        assert combined['reused_predictions'] == sum(item['reused_predictions'] for item in live_prediction)
        assert combined['planner']['decisions'] == summary['controller_decisions'] == 720000
        assert (result['checked_fit_calls'] == result['checked_hint_calls'] ==
                result['reused_predictions'] == 719800)
        assert result['velocity_worker_records'] == result['prediction_worker_records'] == 90
        assert final['checked_fit_calls'] == sum(item['calls'] for item in live_velocity)
        assert final['checked_tracks'] == sum(item['tracks'] for item in live_velocity)
        result['pid_reuse_records_recovered'] = len(velocity) - len(live_velocity)
        result['live_report_totals_verified'] = True
        import torch
        checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        evaluated = torch.load(output / 'evaluation/evaluated_model.pt', map_location='cpu', weights_only=False)
        assert evaluated['model_version'] == 12
        assert checkpoint['model'].keys() == evaluated['model'].keys()
        for name, value in checkpoint['model'].items():
            actual = evaluated['model'][name]
            assert value.dtype == actual.dtype and value.shape == actual.shape, name
            # Compare storage bytes so signed zero and NaN payloads stay visible.
            expected_bytes = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            actual_bytes = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            assert expected_bytes == actual_bytes, name
        assert not any('collision_head' in name for name in checkpoint['model'])
        result['full_acceptance_audit_passed'] = True
        result['all_200_seeds_and_model_weights_verified'] = True
        (output / 'acceptance_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    archive = output / 'audit_boundaries' / f'completed_{completed:03d}'
    if not archive.exists():
        archive.mkdir(parents=True)
        for path in (*velocity_paths, *prediction_paths):
            (archive / path.name).write_bytes(path.read_bytes())
        (archive / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
