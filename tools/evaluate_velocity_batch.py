"""Use the existing evaluator with paired exact checks inside rollout workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from barrage_rl import parallel_evaluation
from barrage_rl.artifacts import atomic_write_json, prepare_new_output
from tools.validate_velocity_batch import candidate

_original_worker = parallel_evaluation._rollout_worker
ADDITIONAL_SOURCES = ()
CANDIDATE_NAME = 'batch velocity normalization'


def checked_worker(*args, **kwargs):
    from barrage_rl.image_oracle import PersistentImageTracker
    original, modified = candidate()
    stats = dict(pid=os.getpid(), calls=0, tracks=0, original_seconds=0., candidate_seconds=0.,
                 all_velocities_bitwise_equal=True, completed=False)
    def checked(self, tracks):
        initial = [(t.velocity, t.velocity_known) for t in tracks]
        start = time.perf_counter()
        original(self, tracks)
        stats['original_seconds'] += time.perf_counter() - start
        expected = [(t.velocity, t.velocity_known) for t in tracks]
        for t, (velocity, known) in zip(tracks, initial):
            t.velocity, t.velocity_known = velocity, known
        start = time.perf_counter()
        modified(self, tracks)
        stats['candidate_seconds'] += time.perf_counter() - start
        for t, (velocity, known) in zip(tracks, expected):
            if (t.velocity.dtype != velocity.dtype or t.velocity.shape != velocity.shape
                    or t.velocity.tobytes() != velocity.tobytes() or t.velocity_known != known):
                stats['all_velocities_bitwise_equal'] = False
                raise AssertionError(f'Velocity mismatch in track {t.track_id}')
        stats['calls'] += 1
        stats['tracks'] += len(tracks)
    PersistentImageTracker._fit_velocities = checked
    try:
        _original_worker(*args, **kwargs)
        stats['completed'] = True
    finally:
        PersistentImageTracker._fit_velocities = original
        atomic_write_json(Path(os.environ['BARRAGE_VELOCITY_AUDIT_DIR']) / f'worker_{os.getpid()}.json', stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--workers', type=int, default=9)
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error('workers and batch size must be positive')
    output = args.output.resolve()
    prepare_new_output(output)
    checkpoint = ROOT / 'best.pt'
    checkpoint_contents = checkpoint.read_bytes()
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = metadata['config']
    episodes = 2 if args.smoke_test else 200
    assert config['evaluation_episodes'] == 200
    seed = int(config['evaluation_seed'])
    from barrage_rl import train_tracked_policy as training
    from tools.train_targeted_dagger import install_runtime
    from barrage_rl.evaluate_tracked_policy import evaluate_tracked_checkpoint
    typed = training.TrackedDAggerConfig(**{k: v for k, v in config.items()
        if k in training.TrackedDAggerConfig.__dataclass_fields__})
    heldout = sorted(training._evaluation_seed_footprint(typed, seed))
    assert len(heldout) == 200
    seeds = heldout[:episodes]
    assert not set(seeds) & training._collection_seed_footprint(typed, typed.collection_seed)
    sources = ['barrage_rl/image_oracle.py', 'barrage_rl/tracked_policy.py',
               'barrage_rl/parallel_evaluation.py', 'tools/validate_velocity_batch.py',
               'tools/evaluate_velocity_batch.py', *ADDITIONAL_SOURCES]
    source_contents = {name: (ROOT / name).read_bytes() for name in sources}
    for name, content in source_contents.items():
        destination = output / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    audit = output / 'worker_checks'
    audit.mkdir()
    os.environ['BARRAGE_VELOCITY_AUDIT_DIR'] = str(audit)
    atomic_write_json(output / 'manifest.json', dict(checkpoint=str(checkpoint),
        episodes=episodes, episode_seeds=seeds, smoke_test=args.smoke_test,
        episode_limit_seconds=5. if args.smoke_test else 120.,
        candidate=CANDIDATE_NAME, source_files=sources,
        input_boundary='existing rendered RGB evaluation path',
        comparison='original and candidate velocity bytes at every fitting call',
        deployment_modified=False, workers=min(args.workers, episodes),
        batch_size=min(args.batch_size, episodes),
        timing_caveat='Reference runs first; worker fit times are diagnostic, not an unbiased speed benchmark.'))
    torch.set_num_threads(1)
    install_runtime()
    parallel_evaluation._rollout_worker = checked_worker
    started = time.perf_counter()
    try:
        result = evaluate_tracked_checkpoint(str(checkpoint), episodes=episodes,
            episode_seeds=seeds, seed=seed, workers=min(args.workers, episodes),
            evaluation_batch_size=min(args.batch_size, episodes), output_dir=str(output / 'evaluation'),
            device_name='cuda', episode_limit_seconds=5. if args.smoke_test else 120.,
            bullet_count=300, targeted_bullet_probability=.10, rendered_rgb=True,
            causal_action_delay_steps=0, analytic_shield=False, pixel_guard='receding',
            search_workers=9, smoke_test=args.smoke_test)
    finally:
        parallel_evaluation._rollout_worker = _original_worker
    checks = [json.loads(path.read_text()) for path in audit.glob('worker_*.json')]
    assert checks and all(c['completed'] and c['all_velocities_bitwise_equal'] for c in checks)
    assert sum(c['calls'] for c in checks) > 0
    assert checkpoint.read_bytes() == checkpoint_contents
    assert all((ROOT / name).read_bytes() == content for name, content in source_contents.items())
    atomic_write_json(output / 'result.json', dict(episodes=episodes,
        success_at_limit=result['success_at_limit'], smoke_test=args.smoke_test,
        elapsed_seconds=time.perf_counter() - started, checked_fit_calls=sum(c['calls'] for c in checks),
        checked_tracks=sum(c['tracks'] for c in checks), worker_records=len(checks),
        all_velocities_bitwise_equal=True, checkpoint_and_sources_unchanged=True))
    print((output / 'result.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
