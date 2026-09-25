"""Measure image-only tracker error against simulator state for diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from barrage_rl.timing import DECISION_DT
from barrage_rl.env import BarrageVisionEnv
from barrage_rl.image_oracle import PersistentImageTracker
from barrage_rl.task_spec import TARGET_TASK, tracking_capacity_for


def _greedy_matches(
    estimated: np.ndarray, actual: np.ndarray, gate: float
) -> list[tuple[int, int]]:
    if not len(estimated) or not len(actual):
        return []
    distances = np.linalg.norm(
        estimated[:, None, :] - actual[None, :, :], axis=2
    )
    used_estimated = np.zeros(len(estimated), dtype=np.bool_)
    used_actual = np.zeros(len(actual), dtype=np.bool_)
    result: list[tuple[int, int]] = []
    for flat in np.argsort(distances, axis=None):
        estimated_index, actual_index = divmod(int(flat), len(actual))
        if distances[estimated_index, actual_index] > gate:
            break
        if used_estimated[estimated_index] or used_actual[actual_index]:
            continue
        used_estimated[estimated_index] = True
        used_actual[actual_index] = True
        result.append((estimated_index, actual_index))
    return result


def benchmark(
    *,
    observation_size: int,
    episodes: int,
    decisions: int,
    seed: int,
    bullet_count: int,
) -> dict[str, float]:
    position_errors: list[float] = []
    velocity_errors: list[float] = []
    reverse: list[bool] = []
    matched = 0
    possible = 0
    known = 0
    detected = []
    for episode in range(episodes):
        env = BarrageVisionEnv(
            bullet_count=bullet_count,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            observation_size=observation_size,
            randomize_initial_phase=False,
        )
        try:
            observation, _ = env.reset(seed=seed + episode)
            tracker = PersistentImageTracker(
                target_track_count=tracking_capacity_for(bullet_count),
                expected_bullet_count=bullet_count,
            )
            tracker.initialize(observation)
            for _ in range(decisions):
                observation, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
                tracker.update(observation)
                tracks = tracker.tracks
                estimated = np.asarray([track.position for track in tracks], np.float32)
                matches = _greedy_matches(estimated, env.bullet_positions, gate=12.0)
                detected.append(len(tracks))
                possible += len(env.bullet_positions)
                matched += len(matches)
                for track_index, actual_index in matches:
                    track = tracks[track_index]
                    position_errors.append(float(np.linalg.norm(
                        track.position - env.bullet_positions[actual_index]
                    )))
                    if not track.velocity_known:
                        continue
                    known += 1
                    actual_velocity = env.bullet_velocities[actual_index]
                    velocity_errors.append(float(np.linalg.norm(
                        track.velocity - actual_velocity
                    )))
                    reverse.append(bool(np.dot(track.velocity, actual_velocity) < 0.0))
        finally:
            env.close()
    velocity_array = np.asarray(velocity_errors, np.float64)
    position_array = np.asarray(position_errors, np.float64)
    return {
        "observation_size": float(observation_size),
        "bullets": float(bullet_count),
        "episodes": float(episodes),
        "decisions": float(decisions),
        "mean_detected": float(np.mean(detected)),
        "position_mae_source_px": float(position_array.mean()),
        "position_p90_source_px": float(np.percentile(position_array, 90.0)),
        "match_recall": matched / max(possible, 1),
        "known_velocity_match_fraction": known / max(matched, 1),
        "velocity_mae_source_px_per_second": float(velocity_array.mean()),
        "velocity_p90_source_px_per_second": float(np.percentile(velocity_array, 90.0)),
        "velocity_mae_source_px_per_decision": float(
            velocity_array.mean() * DECISION_DT
        ),
        "reverse_velocity_fraction": float(np.mean(reverse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-size", type=int, default=192)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1_600_000)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    args = parser.parse_args()
    print(json.dumps(benchmark(
        observation_size=args.observation_size,
        episodes=args.episodes,
        decisions=args.decisions,
        seed=args.seed,
        bullet_count=args.bullets,
    ), indent=2))


if __name__ == "__main__":
    main()
