"""Validate an isolated normalization candidate against current tracker code."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import textwrap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barrage_rl.image_oracle import PersistentImageTracker
from tools import benchmark_tracker_exact as validation


def boundaries(original, modified):
    from barrage_rl.image_oracle import BulletTrack
    rng = np.random.default_rng(92517)
    cases = 0
    for speed in (0., 120., 240., 480.):
        for refit in (False, True):
            tracker = PersistentImageTracker(bullet_speed=speed, refit_known_velocity=refit)
            tracks = []
            for i in range(192):
                length = i % 13
                history = []
                for j in range(length):
                    point = rng.normal(0, 100, 2).astype(np.float32)
                    if i % 11 == 0:
                        point[:] = [0., -0.]
                    if i % 17 == 0:
                        point[0] = np.nan
                    if i % 19 == 0:
                        point[1] = np.inf
                    history.append((0 if i % 7 == 0 else j * (1 + i % 3), point))
                tracks.append(BulletTrack(i, np.zeros(2, np.float32),
                    np.array([-0., 1.], np.float32), history=history,
                    velocity_known=bool(i % 2)))
            old, new = copy.deepcopy(tracks), copy.deepcopy(tracks)
            with np.errstate(all='ignore'):
                original(tracker, old)
                modified(tracker, new)
            for a, b in zip(old, new):
                validation.exact(vars(a), vars(b))
                assert a.velocity.flags.owndata == b.velocity.flags.owndata
                cases += 1
    result = dict(track_cases=cases, state_bitwise_equal=True,
                  includes=['short and long histories', 'repeated timestamps',
                            'nonfinite positions', 'signed zeros', 'alternate speeds',
                            'refit disabled', 'velocity array ownership'])
    (validation.BASE / 'boundary_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def candidate():
    original = PersistentImageTracker._fit_velocities
    source = textwrap.dedent(inspect.getsource(original))
    marker = '        # Keep Python scalar normalization identical to the scalar path.\n'
    assert source.count(marker) == 1
    prefix, fallback = source.split(marker)
    replacement = '''        if np.isfinite(estimates).all() and self.bullet_speed == 240.0:
            wide = magnitudes.astype(np.float64)
            valid = wide >= 0.20 * self.bullet_speed
            factors = (self.bullet_speed / np.maximum(wide[valid], 1e-6)).astype(np.float32)
            velocities = estimates[valid] * factors[:, None]
            for index, velocity in zip(np.flatnonzero(valid), velocities):
                track = group[index][0]
                track.velocity = velocity.copy()
                track.velocity_known = True
            continue
'''
    namespace = dict(original.__globals__)
    exec(compile(prefix + replacement + fallback, '<velocity_batch_candidate>', 'exec'), namespace)
    return original, namespace['_fit_velocities']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['synthetic', 'replay', 'boundaries', 'live-pair'])
    args = parser.parse_args()
    validation.BASE = ROOT / 'diagnostics/velocity_batch_20260925'
    validation.BASE.mkdir(exist_ok=True)
    original, modified = candidate()
    items = [(PersistentImageTracker, '_fit_velocities', original, modified)]
    try:
        if args.mode == 'live-pair':
            validation.BASE = validation.BASE / 'visible'
            validation.live(items, 'before')
            validation.live(items, 'after')
        elif args.mode == 'boundaries':
            boundaries(original, modified)
        elif args.mode == 'synthetic':
            validation.synthetic(items)
        else:
            validation.replay(items, ROOT / 'diagnostics/tracker_exact_20260925/after/image_detections.pkl')
    finally:
        PersistentImageTracker._fit_velocities = original


if __name__ == '__main__':
    main()
