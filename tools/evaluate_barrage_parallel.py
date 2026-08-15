"""Parallel, resumable evaluation through the real Barrage.py game path."""

import argparse
import csv
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FIELDS = (
    "episode", "seed", "survival_seconds", "reached_limit", "worker",
    "checkpoint_sha256", "limit_seconds", "bullets", "bullet_size",
    "bullet_speed", "targeted_probability",
)


def _run_worker(worker_id, jobs, checkpoint, limit_seconds, settings, result_queue):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame

        from Barrage import Barrage, Bullet, Plane
        from barrage_rl.dynamics import ACTION_VECTORS
        from barrage_rl.live_screen import LiveVisualController

        pygame.init()
        pygame.display.set_mode((1, 1))
        Barrage.SCREEN_WIDTH = 820
        Barrage.SCREEN_HEIGHT = 820
        Barrage.BULLET_SIZE = settings["bullet_size"]
        Barrage.QUANTITY = settings["bullets"]
        Barrage.COLLISION = False
        Barrage.INVINCIBLE = True
        Barrage.MUSIC = False
        Barrage.PLANE_SPEED = settings["plane_speed"]
        Barrage.BULLET_SPEED = settings["bullet_speed"]
        Barrage.TARGETED_BULLET_PROBABILITY = settings["targeted_probability"]
        Barrage.TARGETED_PREDICTION_SCALE_MIN = settings["targeted_prediction_min"]
        Barrage.TARGETED_PREDICTION_SCALE_MAX = settings["targeted_prediction_max"]
        Barrage.TARGETED_ANGULAR_NOISE = settings["targeted_angular_noise"]
        Barrage.window = pygame.Surface((820, 820))
        Plane.SKIN = 0
        controller = LiveVisualController(checkpoint)
        Barrage.AI_CONTROLLER = controller
        physics_dt = 1.0 / Barrage.PHYSICS_FPS

        for episode, seed in jobs:
            started = time.monotonic()
            Barrage.RNG = __import__("numpy").random.default_rng(seed)
            Barrage.KEY = True
            Barrage.reset_game()
            while Barrage.KEY and Barrage.TimeNow < limit_seconds:
                direction = ACTION_VECTORS[Barrage.AI_ACTION]
                Barrage.advance_physics(direction, physics_dt)
                Barrage.window.fill("#000000")
                Barrage.PLANE.display()
                Barrage.BULLET.display()
                Barrage.AI_ACTION = controller.observe_surface(Barrage.window)

            survival = min(float(Barrage.TimeNow), float(limit_seconds))
            result_queue.put(
                {
                    "kind": "result",
                    "episode": episode,
                    "seed": seed,
                    "survival_seconds": survival,
                    "reached_limit": int(survival >= limit_seconds - 1e-9),
                    "worker": worker_id,
                    "checkpoint_sha256": settings["checkpoint_sha256"],
                    "limit_seconds": limit_seconds,
                    "bullets": settings["bullets"],
                    "bullet_size": settings["bullet_size"],
                    "bullet_speed": settings["bullet_speed"],
                    "targeted_probability": settings["targeted_probability"],
                    "wall_seconds": time.monotonic() - started,
                }
            )
        pygame.quit()
        result_queue.put({"kind": "done", "worker": worker_id})
    except BaseException:
        result_queue.put(
            {"kind": "error", "worker": worker_id, "traceback": traceback.format_exc()}
        )


def _read_completed(path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {int(row["episode"]): row for row in csv.DictReader(file)}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[index] for index in sorted(rows))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--limit-seconds", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bullets", type=int, default=50)
    parser.add_argument("--bullet-size", type=int, default=5)
    parser.add_argument("--plane-speed", type=float, default=240.0)
    parser.add_argument("--bullet-speed", type=float, default=240.0)
    parser.add_argument("--targeted-probability", type=float, default=0.35)
    parser.add_argument("--targeted-prediction-min", type=float, default=0.65)
    parser.add_argument("--targeted-prediction-max", type=float, default=1.0)
    parser.add_argument("--targeted-angular-noise", type=float, default=0.08)
    parser.add_argument(
        "--output", default="runs/barrage_actual_best_50_fixed/evaluation_episodes.csv"
    )
    args = parser.parse_args()
    output = Path(args.output)
    settings = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "bullets": args.bullets,
        "bullet_size": args.bullet_size,
        "plane_speed": args.plane_speed,
        "bullet_speed": args.bullet_speed,
        "targeted_probability": args.targeted_probability,
        "targeted_prediction_min": args.targeted_prediction_min,
        "targeted_prediction_max": args.targeted_prediction_max,
        "targeted_angular_noise": args.targeted_angular_noise,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        **settings, "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed, "episodes": args.episodes,
        "limit_seconds": args.limit_seconds,
    }
    if output.exists():
        if not manifest_path.exists():
            raise ValueError("existing CSV has no compatible evaluation manifest")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing evaluation manifest does not match this run")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    rows = {
        episode: row for episode, row in _read_completed(output).items()
        if 0 <= episode < args.episodes
    }
    jobs = [
        (episode, args.seed + episode)
        for episode in range(args.episodes)
        if episode not in rows
    ]
    if not jobs:
        print(f"already_complete={output.resolve()}", flush=True)
        return

    worker_count = max(1, min(args.workers, len(jobs)))
    chunks = [jobs[index::worker_count] for index in range(worker_count)]
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_run_worker,
            args=(
                index, chunk, args.checkpoint, args.limit_seconds, settings,
                result_queue,
            ),
        )
        for index, chunk in enumerate(chunks)
    ]
    started = time.monotonic()
    for process in processes:
        process.start()

    finished_workers = 0
    try:
        while finished_workers < worker_count:
            try:
                message = result_queue.get(timeout=30.0)
            except queue.Empty:
                completed = len(rows)
                print(
                    f"progress={completed}/{args.episodes} elapsed_minutes="
                    f"{(time.monotonic() - started) / 60.0:.1f}",
                    flush=True,
                )
                continue
            if message["kind"] == "error":
                raise RuntimeError(
                    f"worker {message['worker']} failed\n{message['traceback']}"
                )
            if message["kind"] == "done":
                finished_workers += 1
                continue
            row = {key: message[key] for key in FIELDS}
            rows[int(row["episode"])] = row
            _write_rows(output, rows)
            print(
                f"episode={row['episode']} seed={row['seed']} "
                f"survival={float(row['survival_seconds']):.3f} "
                f"limit={row['reached_limit']} worker={row['worker']} "
                f"episode_wall_seconds={message['wall_seconds']:.1f} "
                f"progress={len(rows)}/{args.episodes}",
                flush=True,
            )
    finally:
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
    print(f"complete={output.resolve()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
