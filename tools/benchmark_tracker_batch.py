"""Visible scalar/batch timing; paired replay uses only recorded RGB detections."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scalar_fit(self, tracks):
    cache = {}
    for track in tracks:
        self._fit_velocity(track, regression_cache=cache)


def main():
    from barrage_rl.image_oracle import PersistentImageTracker
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    from barrage_rl.artifacts import contents_equal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['scalar', 'batch'], required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--skip-replay', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / 'diagnostics' / ('tracker_' + args.variant + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    batch_fit = PersistentImageTracker._fit_velocities
    if args.variant == 'scalar':
        PersistentImageTracker._fit_velocities = scalar_fit
    original = TrackedFeatureExtractor.step_detections
    rows, inputs = [], []
    initial = None
    def measured(extractor, bullets, plane, **kwargs):
        nonlocal initial
        if initial is None:
            initial = copy.deepcopy(extractor)
        inputs.append((bullets.copy(), None if plane is None else plane.copy(), kwargs.copy()))
        started = time.perf_counter()
        try:
            return original(extractor, bullets, plane, **kwargs)
        finally:
            rows.append((time.perf_counter() - started) * 1000)
    TrackedFeatureExtractor.step_detections = measured
    previous = sys.argv
    try:
        sys.argv = [str(ROOT/'tools/diagnose_window_pacing.py'), '--seconds', str(args.seconds),
                    '--record-window-state', '--output-dir', str(output)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        TrackedFeatureExtractor.step_detections = original
        PersistentImageTracker._fit_velocities = batch_fit
        sys.argv = previous
    stats = lambda values: dict(count=len(values), mean_ms=float(np.mean(values)),
                               p95_ms=float(np.percentile(values,95)), p99_ms=float(np.percentile(values,99)))
    result = dict(variant=args.variant, tracking_features=stats(rows))
    (output/'tracker_timing.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Tracker timing: ' + json.dumps(result), flush=True)
    if args.variant == 'batch' and not args.skip_replay:
        # Alternate order on every update, excluding state comparisons from timing.
        old, new = copy.deepcopy(initial), copy.deepcopy(initial)
        old_times, new_times = [], []
        for index, (bullets, plane, kwargs) in enumerate(inputs):
            results = {}
            order = ('scalar','batch') if index % 2 == 0 else ('batch','scalar')
            for variant in order:
                PersistentImageTracker._fit_velocities = scalar_fit if variant == 'scalar' else batch_fit
                target, times = (old, old_times) if variant == 'scalar' else (new, new_times)
                started = time.perf_counter()
                results[variant] = original(target, bullets, plane, **kwargs)
                times.append((time.perf_counter()-started)*1000)
            assert contents_equal(results['scalar'], results['batch']), f'Feature mismatch at {index}'
            a, b = dict(old.tracker.__dict__), dict(new.tracker.__dict__)
            a['tracks'] = [vars(t) for t in a['tracks']]
            b['tracks'] = [vars(t) for t in b['tracks']]
            assert contents_equal(a,b), f'Tracker state mismatch at {index}'
            if index % 600 == 599:
                print(f'Paired replay: {index+1}/{len(inputs)} exact', flush=True)
        PersistentImageTracker._fit_velocities = batch_fit
        replay = dict(updates=len(inputs), features_and_state_exact=True,
                      scalar=stats(old_times), batch=stats(new_times))
        (output/'paired_replay.json').write_text(json.dumps(replay,indent=2),encoding='utf-8')
        print('Paired replay: '+json.dumps(replay),flush=True)
    print('Results: '+str(output),flush=True)


if __name__ == '__main__':
    main()
