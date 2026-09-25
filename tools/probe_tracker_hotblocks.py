"""Compare array assembly candidates using recorded image tracking histories."""
import json
from itertools import chain
from operator import itemgetter
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barrage_rl.tracked_policy import TrackedFeatureExtractor


def main():
    with (ROOT / 'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:
        records = pickle.load(f)
    extractor = TrackedFeatureExtractor()
    for d, kw, _ in records[:240]:
        extractor.step_detections(d.bullet_positions, d.plane_position, **kw)
    tracks = extractor.tracker.tracks
    histories = [t.history[-8:] for t in tracks if len(t.history) >= 8]
    arrays = [t.position for t in tracks]
    def history_buffers():
        points = list(map(itemgetter(1), chain.from_iterable(histories)))
        assert all(p.dtype == np.dtype('float32') for p in points)
        return np.frombuffer(b''.join(points), np.float32).reshape(-1, 8, 2)

    dtype32 = np.dtype('float32')
    def history_buffers_cached_dtype():
        points = list(map(itemgetter(1), chain.from_iterable(histories)))
        assert all(p.dtype == dtype32 for p in points)
        return np.frombuffer(b''.join(points), np.float32).reshape(-1, 8, 2)

    candidates = {
        'history_asarray': lambda: np.asarray([[p for _, p in h] for h in histories], np.float32),
        'history_concatenate': lambda: np.concatenate([p for h in histories for _, p in h]).reshape(-1, 8, 2),
        'history_concatenate_map': lambda: np.concatenate(list(map(itemgetter(1), chain.from_iterable(histories)))).reshape(-1, 8, 2),
        'history_fromiter': lambda: np.fromiter((v for h in histories for _, p in h for v in p), np.float32).reshape(-1, 8, 2),
        'history_vector_fromiter': lambda: np.fromiter(map(itemgetter(1), chain.from_iterable(histories)), dtype=np.dtype((np.float32, 2)), count=len(histories)*8).reshape(-1, 8, 2),
        'history_buffers': history_buffers,
        'history_buffers_cached_dtype': history_buffers_cached_dtype,
        'positions_asarray': lambda: np.asarray(arrays),
        'positions_concatenate': lambda: np.concatenate(arrays).reshape(-1, 2),
        'positions_vector_fromiter': lambda: np.fromiter(iter(arrays), dtype=np.dtype((np.float32, 2)), count=len(arrays)),
        'timestamps_generator': lambda: [tuple(item[0] for item in h) for h in histories],
        'timestamps_map': lambda: [tuple(map(itemgetter(0), h)) for h in histories],
    }
    for name, function in candidates.items():
        reference = candidates['history_asarray' if name.startswith('history') else 'positions_asarray' if name.startswith('positions') else 'timestamps_generator']()
        actual = function()
        if isinstance(actual, np.ndarray):
            assert actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes()
        else:
            assert actual == reference
    results = {}
    for repeat in range(6):
        for name in list(candidates)[::1 if repeat % 2 else -1]:
            start = time.perf_counter()
            for _ in range(1000):
                candidates[name]()
            results.setdefault(name, []).append((time.perf_counter()-start))
    result = {name: float(np.median(values)) for name, values in results.items()}
    (ROOT/'diagnostics/tracker_hotblocks_20260925/assembly_probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
