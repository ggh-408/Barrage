"""Validate the fused scan using existing captured RGB images and detector state."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import pickle
import sys
import textwrap
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barrage_rl.live_screen import DominantBackgroundSemanticizer as Detector
from tools.probe_fused_rgb_classification import classify
from tools.benchmark_tracker_exact import exact


def candidate_class():
    source = textwrap.dedent(inspect.getsource(Detector.detect))
    old = '    foreground = self._foreground_mask(\n        image, background, self.color_threshold\n    )\n'
    assert source.count(old) == 1
    source = source.replace(old, '')
    old = 'bullet_mask, plane_yx = self._classify_foreground(image, foreground)'
    assert source.count(old) == 1
    source = source.replace(old, 'bullet_mask, plane_yx = _classify_scan(image, background, self.color_threshold)')
    namespace = dict(Detector.detect.__globals__, _classify_scan=classify)
    exec(compile(source, '<fused_rgb_detect>', 'exec'), namespace)
    return type('FusedDetector', (Detector,), {'detect': namespace['detect']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'diagnostics/rgb_blocks_20260925_112807_783291/rgb_samples.pkl'
    with path.open('rb') as stream:
        samples = pickle.load(stream)
    detectors = Detector(), candidate_class()()
    cases = 0
    for sample in samples:
        for variant in ('recorded', 'full', 'semantic'):
            kwargs = copy.deepcopy(sample['kwargs'])
            if variant == 'full':
                kwargs = dict(include_semantic=False)
            elif variant == 'semantic':
                kwargs['include_semantic'] = True
            results, states = [], []
            for detector in detectors:
                detector.__dict__ = copy.deepcopy(sample['state'])
                results.append(vars(detector.detect(sample['image'], **kwargs)))
                states.append(copy.deepcopy(vars(detector)))
            exact(results[0], results[1])
            exact(states[0], states[1])
            if variant == 'recorded':
                exact(results[0], vars(sample['result']))
                exact(states[0], sample['after'])
            cases += 1
    result = dict(samples=len(samples), cases=cases, outputs_and_state_bitwise_equal=True,
                  recorded_outputs_bitwise_equal=True, deployment_modified=False)
    if args.timing:
        timings = [[], []]
        for repeat in range(20):
            for index, sample in enumerate(samples):
                for variant in ([0, 1] if (index + repeat) % 2 else [1, 0]):
                    detector = detectors[variant]
                    detector.__dict__ = copy.deepcopy(sample['state'])
                    start = time.perf_counter()
                    detector.detect(sample['image'], **sample['kwargs'])
                    timings[variant].append((time.perf_counter() - start) * 1000)
        for name, values in zip(('before', 'after'), timings):
            result[name] = dict(count=len(values), mean_ms=float(np.mean(values)),
                               p95_ms=float(np.percentile(values, 95)))
    output = ROOT / 'diagnostics/rgb_shared_scan_20260925'
    output.mkdir(exist_ok=True)
    filename = 'replay_timing.json' if args.timing else 'replay_validation.json'
    (output / filename).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
