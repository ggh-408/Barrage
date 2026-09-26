"""Measure scheme-two RGB detection speed and fidelity on moving scenes.

Simulator coordinates are used only by this offline diagnostic to render exact
reference RGB frames.  Both detectors and the persistent tracker receive image-
derived detections only, matching the real deployment observation boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from barrage_rl.task_spec import TARGET_TASK, tracking_capacity_for


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "p99_ms": float(np.percentile(array, 99.0)),
    }


def _prediction_hints(tracker: "object") -> tuple[np.ndarray, np.ndarray]:
    tracks = tracker.tracks
    elapsed = tracker.decision_dt
    positions = np.stack([
        track.position + (track.velocity * elapsed if track.velocity_known else 0.0)
        for track in tracks
    ]).astype(np.float32)
    uncertainty = np.asarray(
        [track.position_uncertainty for track in tracks], dtype=np.float32
    )
    known = np.asarray([track.velocity_known for track in tracks], dtype=np.bool_)
    radii = np.where(
        known,
        np.clip(uncertainty + 4.0, 6.0, 24.0),
        np.clip(uncertainty + 8.0, 24.0, 64.0),
    )
    return positions / tracker.source_size, radii


def benchmark(bullets: int, seeds: int, decisions: int) -> dict[str, object]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    from barrage_rl.env import BarrageVisionEnv
    from barrage_rl.image_oracle import PersistentImageTracker
    from barrage_rl.live_screen import DominantBackgroundSemanticizer

    pygame.init()
    pygame.display.set_mode((1, 1))
    surface = pygame.Surface((820, 820))
    bullet = pygame.image.load(
        str(PROJECT_ROOT / "image" / "bullet(5).gif")
    ).convert_alpha()
    plane = pygame.image.load(
        str(PROJECT_ROOT / "image" / "plane(0).gif")
    ).convert_alpha()
    full_times: list[float] = []
    predictive_times: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    count_errors: list[int] = []
    frame_diagnostics: list[dict[str, float | int]] = []

    for seed in range(2_300_000, 2_300_000 + seeds):
        env = BarrageVisionEnv(
            bullet_count=bullets,
            bullet_size=5,
            bullet_speed_min=240.0,
            bullet_speed_max=240.0,
            targeted_bullet_probability=0.10,
            observation_size=384,
            randomize_initial_phase=False,
        )
        full_detector = DominantBackgroundSemanticizer(bullet_size=5)
        predictive_detector = DominantBackgroundSemanticizer(bullet_size=5)
        tracker = PersistentImageTracker(
            source_size=820.0,
            bullet_speed=240.0,
            target_track_count=tracking_capacity_for(bullets),
            expected_bullet_count=bullets,
        )
        try:
            env.reset(seed=seed)
            for decision in range(decisions):
                surface.fill((0, 0, 0))
                surface.blit(plane, plane.get_rect(center=tuple(env.plane_position)))
                for position in env.bullet_positions:
                    surface.blit(bullet, bullet.get_rect(center=tuple(position)))
                rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

                started = time.perf_counter()
                reference = full_detector.detect(rgb, include_semantic=False)
                full_times.append((time.perf_counter() - started) * 1000.0)
                if decision == 0:
                    predicted = predictive_detector.detect(
                        rgb, include_semantic=False
                    )
                    tracker.initialize_detections(
                        predicted.bullet_positions,
                        predicted.plane_position,
                        normalized=True,
                    )
                else:
                    positions, radii = _prediction_hints(tracker)
                    started = time.perf_counter()
                    predicted = predictive_detector.detect(
                        rgb,
                        include_semantic=False,
                        predicted_bullet_positions=positions,
                        prediction_search_radii=radii,
                    )
                    predictive_times.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    tracker.update_detections(
                        predicted.bullet_positions,
                        predicted.plane_position,
                        normalized=True,
                    )
                    reference_xy = reference.bullet_positions * 820.0
                    predicted_xy = predicted.bullet_positions * 820.0
                    if len(reference_xy) and len(predicted_xy):
                        distances = np.linalg.norm(
                            reference_xy[:, None, :] - predicted_xy[None, :, :],
                            axis=2,
                        )
                        recalls.append(float(np.mean(np.min(distances, axis=1) <= 1.5)))
                        precisions.append(float(np.mean(np.min(distances, axis=0) <= 1.5)))
                    else:
                        recalls.append(float(len(reference_xy) == 0))
                        precisions.append(float(len(predicted_xy) == 0))
                    count_errors.append(int(len(predicted_xy) - len(reference_xy)))
                    frame_diagnostics.append({
                        "seed": int(seed),
                        "decision": int(decision),
                        "reference_count": int(len(reference_xy)),
                        "predictive_count": int(len(predicted_xy)),
                        "recall": float(recalls[-1]),
                        "tracker_count": int(len(tracker.tracks)),
                        "known_velocity_fraction": float(
                            tracker.known_velocity_fraction
                        ),
                        "predicted_matches": int(
                            predictive_detector.last_predicted_match_count
                        ),
                        "recovery_detections": int(
                            predictive_detector.last_recovery_detection_count
                        ),
                    })
                for _ in range(env.action_repeat):
                    env._move_bullets()
        finally:
            env.close()

    pygame.quit()
    errors = np.asarray(count_errors, dtype=np.int32)
    return {
        "observation_boundary": "rendered RGB only",
        "bullets": int(bullets),
        "targeted_bullet_probability": 0.10,
        "seeds": int(seeds),
        "decisions_per_seed": int(decisions),
        "full_detector": _stats(full_times[5:]),
        "predictive_detector": _stats(predictive_times[5:]),
        "speedup": float(np.mean(full_times[5:]) / np.mean(predictive_times[5:])),
        "reference_match_recall_mean": float(np.mean(recalls)),
        "reference_match_recall_min": float(np.min(recalls)),
        "reference_match_precision_mean": float(np.mean(precisions)),
        "count_error_mean": float(errors.mean()),
        "count_error_abs_p95": float(np.percentile(np.abs(errors), 95.0)),
        "count_error_min": int(errors.min()),
        "count_error_max": int(errors.max()),
        "worst_frames": sorted(
            frame_diagnostics, key=lambda item: float(item["recall"])
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--decisions", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.bullets, args.seeds, args.decisions)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
