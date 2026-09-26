"""Ordinary-function staging of the validated velocity fitting candidate.

Not installed in runtime. Kept separate while the fixed-200 source snapshot runs.
"""
import numpy as np


def fit_velocities(self, tracks):
    groups = {}
    for track in tracks:
        if track.velocity_known and not self.refit_known_velocity:
            continue
        history = track.history[-8:]
        if len(history) >= 4:
            timestamps = tuple(item[0] for item in history)
            groups.setdefault(timestamps, []).append((track, history))
    for timestamps, group in groups.items():
        steps = np.asarray(timestamps, np.float32)
        centered = steps - float(steps.mean())
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0:
            continue
        positions = np.asarray(
            [[item[1] for item in history] for _, history in group],
            dtype=np.float32,
        )
        displacement = (centered[None, :, None] * positions).sum(axis=1) / denominator
        estimates = displacement / self.decision_dt
        magnitudes = np.sqrt((estimates * estimates).sum(axis=1))
        if np.isfinite(estimates).all() and self.bullet_speed == 240.0:
            # Python scalar division was float64, followed by float32 factors
            # during the original array multiplication. Preserve both stages.
            wide = magnitudes.astype(np.float64)
            valid = wide >= 0.20 * self.bullet_speed
            factors = (self.bullet_speed / np.maximum(wide[valid], 1e-6)).astype(np.float32)
            velocities = estimates[valid] * factors[:, None]
            for index, velocity in zip(np.flatnonzero(valid), velocities):
                track = group[index][0]
                track.velocity = velocity.copy()
                track.velocity_known = True
            continue
        for (track, _), estimate, magnitude in zip(group, estimates, magnitudes):
            magnitude = float(magnitude)
            if magnitude < 0.20 * self.bullet_speed:
                continue
            track.velocity = (
                estimate * (self.bullet_speed / max(magnitude, 1e-6))
            ).astype(np.float32)
            track.velocity_known = True
