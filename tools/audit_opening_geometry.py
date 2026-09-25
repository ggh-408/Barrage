"""Replay image-planned paths in saved worlds, strictly after policy execution."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import sys
import json
import pickle
import argparse
from collections import deque
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barrage_rl import parallel_evaluation
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.deployment import configure_image_controller
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.artifacts import atomic_write_json

_reset = BarrageVisionEnv.reset
def diagnostic_reset(self, **kwargs):
    self._audit_seed = kwargs.get('seed')
    return _reset(self, **kwargs)
BarrageVisionEnv.reset = diagnostic_reset
_teacher = parallel_evaluation.privileged_planner_supervision
def diagnostic_teacher(env, **kwargs):
    folder = Path(os.environ['BARRAGE_OPENING_AUDIT_DIR'])
    path = folder / f'{env._audit_seed}_{env.episode_steps}.pkl'
    snapshot = env.capture_state()
    if path.exists():
        with path.open('rb') as stream:
            saved = pickle.load(stream)
        assert saved.physics_steps == snapshot.physics_steps
        for name in ('plane_position', 'bullet_positions', 'bullet_velocities'):
            assert np.array_equal(getattr(saved, name), getattr(snapshot, name))
    else:
        with path.open('xb') as stream:
            pickle.dump(snapshot, stream)
    kwargs.setdefault('reaction_seconds', .10)
    return _teacher(env, **kwargs)
parallel_evaluation.privileged_planner_supervision = diagnostic_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-preflight', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and (not args.resume_preflight or (args.output/'report.json').exists()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume_preflight)
    worlds = args.output / 'worlds'
    worlds.mkdir(exist_ok=args.resume_preflight)
    os.environ['BARRAGE_OPENING_AUDIT_DIR'] = str(worlds.resolve())
    torch.set_num_threads(1)
    seeds = [282467554, 282467667]
    agent, spec, _ = load_tracked_agent(str(ROOT/'runs/visual_set_v51/candidate.pt'),
        torch.device('cuda'), analytic_shield=True, analytic_shield_gate='learned_all_unsafe')
    configure_image_controller(agent, 'receding', search_workers=1)
    guard = agent._receding_pixel_guard
    original_assess = guard._assess
    latest = {}
    traces = {i: deque(maxlen=30) for i in range(len(seeds))}
    counts = {i: 0 for i in traces}
    def assess(plane, half_size, bullets, velocity, error, paths, lengths):
        latest.clear()
        latest.update(plane=plane.copy(), bullets=bullets.copy(), velocity=velocity.copy(),
                      error=error.copy(), paths=paths.copy(), lengths=lengths.copy())
        return original_assess(plane, half_size, bullets, velocity, error, paths, lengths)
    guard._assess = assess
    def observe(key, event):
        key = int(key)
        traces[key].append(dict(decision_index=counts[key], event=event, geometry=latest.copy()))
        counts[key] += 1
    guard._decision_observer = observe
    kwargs = TARGET_TASK.env_kwargs()
    kwargs['max_episode_seconds'] = 4.
    result = parallel_evaluation.run_parallel_rollout(agent, spec, 2, 2, seeds[0], kwargs, 40.,
        episode_seeds=seeds, rendered_rgb=True, causal_action_delay_steps=0,
        collect_failure_diagnostics=True, failure_lookback_decisions=15)
    # All world state below is offline diagnosis and never reaches the policy.
    rows = []
    env = BarrageVisionEnv(**kwargs)
    for index, seed in enumerate(seeds):
        env.reset(seed=seed)
        fatal = result.failure_diagnostics[index]['collision_bullets'][0]['bullet_index']
        for item in traces[index]:
            step = item['decision_index']
            world_path = worlds / f'{seed}_{step}.pkl'
            event = item['event']
            if not world_path.exists() or event['reason'] != 'candidate_ranking':
                continue
            with world_path.open('rb') as stream:
                snapshot = pickle.load(stream)
            geometry = item['geometry']
            candidates = []
            for candidate, (path, length) in enumerate(zip(geometry['paths'], geometry['lengths'])):
                env.restore_state(snapshot)
                elapsed = 0
                terminated = False
                for action in path[:length]:
                    _, _, terminated, truncated, _ = env.step(int(action))
                    elapsed += 1
                    if terminated or truncated:
                        break
                candidates.append(dict(index=candidate, root=int(path[0]),
                    nominal_full=bool(event['metrics'][candidate, 3] == 1),
                    interval_full=bool(event['metrics'][candidate, 5] == 1),
                    eligible=bool(event['eligible'][candidate]),
                    survived_full_path=not terminated and elapsed == int(length),
                    survived_seconds=(env.physics_steps-snapshot.physics_steps)/120))
            position = snapshot.bullet_positions[fatal]
            distance = np.linalg.norm(geometry['bullets']-position[None], axis=1)
            nearest = int(np.argmin(distance))
            truth_velocity = snapshot.bullet_velocities[fatal]
            selected = int(event['chosen'])
            rows.append(dict(seed=seed, decision_index=step, seconds=step/30,
                chosen=selected, chosen_result=candidates[selected],
                surviving_alternative_roots=sorted(set(c['root'] for c in candidates
                    if c['survived_full_path'] and c['root'] != int(event['action']))),
                fatal_bullet_index=fatal, nearest_track_distance=float(distance[nearest]),
                estimated_velocity=geometry['velocity'][nearest].tolist(),
                actual_velocity=truth_velocity.tolist(),
                velocity_error= float(np.linalg.norm(geometry['velocity'][nearest]-truth_velocity)),
                modeled_velocity_error_bound=float(geometry['error'][nearest]),
                candidates=candidates))
    env.close()
    report = dict(formal_evaluation=False, scope='Two known opening failures, offline fixed image-path replay',
        policy_received_privileged_state=False, survival_seconds=result.survival_times.tolist(), rows=rows)
    atomic_write_json(args.output/'report.json', report)
    print(json.dumps(dict(survival_seconds=report['survival_seconds'], states=len(rows),
        chosen_failed_with_surviving_alternative=sum(not r['chosen_result']['survived_full_path'] and
            bool(r['surviving_alternative_roots']) for r in rows))), flush=True)


if __name__ == '__main__':
    main()
