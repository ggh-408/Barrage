"""Measure an exact batched normalization candidate without changing deployment."""
import json
import time
from pathlib import Path

import numpy as np


def scalar(estimates, magnitudes, speed):
    result = estimates.copy()
    for i, (estimate, magnitude) in enumerate(zip(estimates, magnitudes)):
        magnitude = float(magnitude)
        if magnitude < .20 * speed:
            continue
        result[i] = (estimate * (speed / max(magnitude, 1e-6))).astype(np.float32)
    return result


def batched(estimates, magnitudes, speed):
    result = estimates.copy()
    valid = magnitudes >= .20 * speed
    # Python division uses double precision; NumPy casts the scalar to float32
    # before multiplying each float32 estimate in the reference implementation.
    scale = (speed / np.maximum(magnitudes[valid].astype(np.float64), 1e-6)).astype(np.float32)
    result[valid] = estimates[valid] * scale[:, None]
    return result


def main():
    rng = np.random.default_rng(925)
    samples = rng.normal(0, 300, (200000, 2)).astype(np.float32)
    samples[:4] = [[0., -0.], [48., 0.], [-48., -0.], [1e-20, 1e-20]]
    magnitudes = np.sqrt((samples * samples).sum(axis=1))
    old = scalar(samples, magnitudes, 240.)
    new = batched(samples, magnitudes, 240.)
    assert old.tobytes() == new.tobytes()
    timings = [[], []]
    sample = samples[:384]
    magnitude = magnitudes[:384]
    for repeat in range(10):
        for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            function = (scalar, batched)[index]
            start = time.perf_counter()
            for _ in range(500):
                function(sample, magnitude, 240.)
            timings[index].append((time.perf_counter() - start) * 1000 / 500)
    result = dict(exact_finite_vectors=len(samples), objects=384,
                  scalar_ms=float(np.median(timings[0])),
                  batched_ms=float(np.median(timings[1])),
                  scope='Finite float32 normalization microbenchmark only; deployment unchanged')
    output = Path(__file__).resolve().parents[1] / 'diagnostics/velocity_normalization_probe_20260925.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
