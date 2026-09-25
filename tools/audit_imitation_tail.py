"""Audit saved imitation supervision without changing historical artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from barrage_rl.artifacts import atomic_write_json
from barrage_rl.train_tracked_policy import _episode_validation_mask


def audit_replay(path, temperature=.08):
    with np.load(path, allow_pickle=False) as data:
        regrets = torch.from_numpy(data['regrets'].astype(np.float32))
        collisions = data['collisions'].astype(bool)
        actions = data['actions'].astype(int)
        episodes = data['episode_ids']
        priorities = data['priorities'].astype(float)
    unsafe = torch.from_numpy(collisions[:, 0])
    masked = unsafe & (~unsafe).any(dim=1, keepdim=True)
    logits = -regrets / temperature
    old = torch.softmax(logits.masked_fill(masked, -80.), dim=1)
    strict_mask = torch.softmax(logits.masked_fill(masked, -torch.inf), dim=1)
    corrected = torch.softmax(logits, dim=1)
    leak = (old * masked).sum(dim=1).numpy()
    old_action = old.argmax(dim=1).numpy()
    new_action = corrected.argmax(dim=1).numpy()
    n = len(actions)
    result = dict(path=str(path), samples=n, source_episodes=len(np.unique(episodes)),
        all_actions_unsafe_states=int(unsafe.all(dim=1).sum()),
        teacher_fixed_hold_unsafe_states=int(collisions[np.arange(n), 0, actions].sum()),
        masked_mass_over_half=int(np.sum(leak > .5)),
        masked_mass_over_one_percent=int(np.sum(leak > .01)),
        maximum_masked_mass=float(leak.max()),
        changed_target_argmax=int(np.sum(old_action != new_action)),
        strict_mask_changed_target_argmax=int(np.sum(old_action != strict_mask.argmax(1).numpy())),
        strict_mask_excludes_teacher_minimum=int(np.sum(
            strict_mask.argmax(1).numpy() != regrets.argmin(1).numpy())),
        corrected_target_matches_minimum_regret=bool(torch.equal(corrected.argmax(1), regrets.argmin(1))),
        corrected_finite=bool(torch.isfinite(corrected).all()),
        example_indices=np.flatnonzero(leak > .01)[:20].tolist())
    validation = _episode_validation_mask(episodes)
    result['splits'] = {}
    for name, selected in [('training', ~validation), ('validation', validation)]:
        result['splits'][name] = dict(states=int(selected.sum()),
            episodes=len(np.unique(episodes[selected])),
            masked_mass_over_half=int(np.sum(leak[selected] > .5)),
            priority_fraction=float(priorities[selected & (leak > .5)].sum() /
                                    max(priorities[selected].sum(), 1)))
    return result


def saved_results(folder):
    with (folder / 'evaluation_episodes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    failed = [dict(seed=int(r['seed']), seconds=float(r['model_survival_seconds']),
                   reason=r['termination_reason']) for r in rows
              if r['termination_reason'] != 'time_limit']
    return dict(path=str(folder), episodes=len(rows), failures=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    start = time.perf_counter()
    replay = audit_replay(ROOT / 'runs/visual_set_v49/replay_latest.npz')
    labels_path = ROOT / 'runs/visual_set_v50/continuation_labels.npz'
    with np.load(labels_path, allow_pickle=False) as data:
        labels = data['labels']
        validation = data['validation'].astype(bool)
    manifest = json.loads((labels_path.parent / 'branch_manifest.json').read_text())
    reasons = {reason: sum(r['reason'] == reason for r in manifest)
               for reason in sorted({r['reason'] for r in manifest})}
    result = dict(replay=replay, continuation=dict(states=len(labels), reasons=reasons,
        source_episodes=len({r['group'] for r in manifest}),
        training_episodes=len({r['group'] for r in manifest if not r['validation']}),
        validation_episodes=len({r['group'] for r in manifest if r['validation']}),
        validation_success_fraction=float(labels[validation].mean())),
        saved_evaluations=[saved_results(ROOT / p) for p in (
            'runs/visual_set_v49_eval', 'runs/visual_set_v49/round1/evaluation',
            'runs/visual_set_v50/evaluation')],
        elapsed_seconds=time.perf_counter()-start)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
