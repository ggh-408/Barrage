import csv
import json
from types import SimpleNamespace

import pytest

from tools import evaluate_targeted_dagger as entry


def test_fresh_run_needs_no_manifest_or_historical_results(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'test checkpoint')
    for name in ('tools/evaluate_targeted_dagger.py',
                 'diagnostics/dagger_ready_20260921/training_config.json',
                 *entry.OPTIMIZED_SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('current source')
    monkeypatch.setattr(entry.training, '_MANIFEST_SOURCE_FILES', ())
    monkeypatch.setattr(entry.entry, 'EXTRA_SOURCES', ())
    monkeypatch.setattr(entry.entry, 'load_config', lambda: SimpleNamespace(feature_expected_bullet_count=250))
    monkeypatch.setattr(entry.torch, 'load', lambda *a, **k: {
        'experimental_controller': entry.entry.CONTROLLER,
        'tracked_policy_spec': {'expected_bullet_count': 250},
        'config': {'bullet_count': 300, 'targeted_bullet_probability': .10,
                   'collection_seed': 123, 'evaluation_seed': 456, 'evaluation_episodes': 2},
        'model': {},
    })
    monkeypatch.setattr(entry.training, '_collection_seed_footprint', lambda *a: {123})
    monkeypatch.setattr(entry, 'install_evaluation_runtime', lambda *a: None)
    calls = []

    def evaluate(checkpoint_path, **kwargs):
        calls.append(kwargs)
        output = entry.Path(kwargs['output_dir'])
        output.mkdir(parents=True)
        with (output / 'evaluation_episodes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['seed', 'termination_reason'])
            writer.writeheader()
            writer.writerows({'seed': seed, 'termination_reason': 'time_limit'}
                            for seed in kwargs['episode_seeds'])
        return {'controller': {'experimental_controller': entry.entry.CONTROLLER},
                'success_at_limit': 1.0}

    monkeypatch.setattr(entry.evaluation, 'evaluate_tracked_checkpoint', evaluate)
    for index in range(2):
        if index:
            for directory in ('runs', 'diagnostics'):
                path = tmp_path / directory / 'old/evaluation_episodes.partial.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('invalid historical CSV')
        output = tmp_path / f'output{index}'
        monkeypatch.setattr(entry.sys, 'argv', ['evaluate', '--episodes', '3', '--workers', '2',
                                              '--output', str(output)])
        entry.main()
        manifest = json.loads((output / 'experiment_manifest.json').read_text())
        assert manifest['historical_evaluation_results_used'] is False
        assert manifest['excluded_seeds'] == [123, 456, 457]
        assert 'approved_source_updates' not in manifest
    assert calls[0]['episode_seeds'] == calls[1]['episode_seeds']
    assert calls[0]['workers'] == 2
    assert not (tmp_path / 'run_manifest.json').exists()


def test_regression_requires_explicit_seed_pool(monkeypatch):
    monkeypatch.setattr(entry.sys, 'argv', ['evaluate', '--mode', 'regression'])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


def test_seed_pool_and_training_exclusions(tmp_path):
    baseline = entry.select_seeds('fresh', 10, 123, None, set())
    filtered = entry.select_seeds('fresh', 10, 123, None, {baseline[0]})
    assert baseline[0] not in filtered
    assert len(set(filtered)) == 10
    pool = tmp_path / 'seeds.csv'
    pool.write_text('seed\n31\n17\n')
    assert entry.select_seeds('regression', 2, 123, pool, set()) == [31, 17]
