"""Exact paired tracking replay against the saved pre-optimization source."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barrage_rl.image_oracle import PersistentImageTracker as Tracker
from barrage_rl.tracked_policy import TrackedFeatureExtractor
from tools import benchmark_tracker_exact as validation

OUT = ROOT/'diagnostics/tracker_hotblocks_20260925'


def variants():
    return [(Tracker, name, validation.previous_method(Tracker, name, OUT/'image_oracle_before.py'),
             getattr(Tracker, name)) for name in ('_fit_velocities', '_update_measurements')]


def replay(items, count, seconds):
    from barrage_rl.runtime_core import tracker_prediction_hints
    recording = ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl'
    with recording.open('rb') as f:
        records = pickle.load(f)[:count or None]
    extractors = [TrackedFeatureExtractor(), TrackedFeatureExtractor()]
    timings = [[], []]
    started = time.perf_counter()
    for i, (d, kw, _) in enumerate(records):
        features = {}
        for j in ([0, 1] if i % 2 else [1, 0]):
            validation.switch(items, j == 0)
            start = time.perf_counter()
            features[j] = extractors[j].step_detections(d.bullet_positions, d.plane_position, **kw)
            timings[j].append((time.perf_counter()-start)*1000)
        validation.exact(features[0], features[1])
        validation.exact(validation.state(extractors[0]), validation.state(extractors[1]))
        validation.exact(tracker_prediction_hints(extractors[0].tracker, (820, 820, 3)),
                         tracker_prediction_hints(extractors[1].tracker, (820, 820, 3)))
        if (i+1) % 600 == 0:
            print(f'{i+1}/{len(records)} exact features, full state, hints', flush=True)
    validation.switch(items, False)
    result = dict(updates=len(records), before=validation.stats(timings[0]),
                  after=validation.stats(timings[1]), direct_content_equal=True,
                  validation_wall_seconds=time.perf_counter()-started)
    result['speedup'] = result['before']['mean_ms']/result['after']['mean_ms']
    if seconds:
        start = time.perf_counter()
        cpu_start = time.process_time()
        updates = 0
        while time.perf_counter()-start < seconds:
            extractor = TrackedFeatureExtractor()
            for d, kw, _ in records:
                values = extractor.step_detections(d.bullet_positions, d.plane_position, **kw)
                assert all(np.isfinite(v).all() for v in values)
                updates += 1
                if time.perf_counter()-start >= seconds:
                    break
        wall = time.perf_counter()-start
        result['stress'] = dict(seconds=wall, updates=updates, updates_per_second=updates/wall,
                                process_cpu_seconds=time.process_time()-cpu_start,
                                finite_features=True)
    (OUT/f'tracker_replay_{len(records)}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['synthetic', 'replay', 'boundaries', 'controller', 'blocks'])
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--stress-seconds', type=float, default=0)
    args = parser.parse_args()
    validation.BASE = OUT
    items = variants()
    try:
        if args.mode == 'synthetic':
            validation.synthetic(items)
        elif args.mode == 'boundaries':
            from tools.validate_velocity_batch import boundaries
            boundaries(items[0][2], items[0][3])
        elif args.mode == 'controller':
            validation.replay(items, ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl',
                              verify_planner_state=True)
        elif args.mode == 'blocks':
            from tools.profile_tracker_blocks import BlockProfiler
            baseline_specs = runpy.run_path(str(OUT/'profile_tracker_blocks_before.py'))['SPECS']
            with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:
                records = pickle.load(f)[:args.count or None]
            for before in (True, False):
                validation.switch(items, before)
                timer = BlockProfiler(10)
                if before:
                    timer.specs = baseline_specs
                timer.install()
                try:
                    extractor = TrackedFeatureExtractor()
                    for d, kw, _ in records:
                        extractor.step_detections(d.bullet_positions, d.plane_position, **kw)
                finally:
                    timer.close()
                variant = 'before' if before else 'after'
                report = timer.report()
                (OUT/f'blocks_{variant}.json').write_text(json.dumps(report, indent=2))
                print(variant, report['sampled_mean_ms'], report['unsampled_mean_ms'], flush=True)
        else:
            replay(items, args.count, args.stress_seconds)
    finally:
        validation.switch(items, False)


if __name__ == '__main__':
    main()
