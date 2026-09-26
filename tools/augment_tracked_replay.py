"""Append one pass over newly discovered failure seeds to a tracked replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from barrage_rl.timing import DECISION_DT
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.tracked_collection import ParallelTrackedDaggerEnv
from barrage_rl.train_tracked_policy import TrackedReplay
from barrage_rl.task_spec import TARGET_TASK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input_replay", type=Path)
    parser.add_argument("output_replay", type=Path)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--capacity", type=int, default=600_000)
    parser.add_argument("--limit-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rendered-rgb", action="store_true")
    parser.add_argument(
        "--action-repeat-choices",
        default="",
        help="comma-separated physics-step counts for timing randomization",
    )
    parser.add_argument("--failure-tail-priority", type=float, default=6.0)
    args = parser.parse_args()
    seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique values")
    if args.limit_seconds <= 0.0:
        raise ValueError("--limit-seconds must be positive")
    action_repeat_choices = tuple(
        int(value.strip())
        for value in args.action_repeat_choices.split(",")
        if value.strip()
    )
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    agent, spec, checkpoint = load_tracked_agent(str(args.checkpoint), device)
    replay = TrackedReplay.load(
        args.input_replay,
        args.capacity,
        spec,
        len(agent.model.safety_horizons),
    )
    env_kwargs = {
        "bullet_count": args.bullets,
        "bullet_size_min": 5,
        "bullet_size_max": 5,
        "bullet_speed_min": 240.0,
        "bullet_speed_max": 240.0,
        "targeted_bullet_probability": 0.10,
        "observation_size": int(checkpoint.get("observation_size", 384)),
        "max_episode_seconds": float(args.limit_seconds),
        "randomize_initial_phase": False,
        "wall_collision": False,
    }
    buffers = [
        {
            "objects": [], "masks": [], "globals": [], "actions": [],
            "regrets": [], "collisions": [], "steps": [], "priorities": [],
        }
        for _ in seeds
    ]
    completed = np.zeros(len(seeds), dtype=np.bool_)
    failures = np.zeros(len(seeds), dtype=np.bool_)
    survivals = np.zeros(len(seeds), dtype=np.float64)
    with ParallelTrackedDaggerEnv(
        env_count=len(seeds),
        workers=min(args.workers, len(seeds)),
        seed=0,
        initial_episode_seeds=seeds,
        env_kwargs=env_kwargs,
        spec=spec,
        safety_horizons=agent.model.safety_horizons,
        teacher_kind="exact",
        teacher_horizon_seconds=1.5,
        teacher_reaction_seconds=DECISION_DT,
        deployment_rgb_observation=args.rendered_rgb,
        action_repeat_choices=action_repeat_choices,
    ) as pipeline:
        while not bool(completed.all()):
            actions = agent.act_features(
                pipeline.objects, pipeline.masks, pipeline.globals
            )
            for index in np.flatnonzero(~completed):
                unsafe = bool(pipeline.collisions[index, 0, actions[index]])
                disagreement = bool(actions[index] != pipeline.teacher_actions[index])
                priority = 1.0 + 4.0 * disagreement + 8.0 * unsafe
                step = int(pipeline.episode_steps[index])
                priority *= 3.0 if step < 90 else (2.0 if step >= 900 else 1.0)
                buffer = buffers[index]
                buffer["objects"].append(pipeline.objects[index].copy())
                buffer["masks"].append(pipeline.masks[index].copy())
                buffer["globals"].append(pipeline.globals[index].copy())
                buffer["actions"].append(int(pipeline.teacher_actions[index]))
                buffer["regrets"].append(pipeline.regrets[index].copy())
                buffer["collisions"].append(pipeline.collisions[index].copy())
                buffer["steps"].append(step)
                buffer["priorities"].append(priority)
            done, truncated, survival = pipeline.step(actions)
            newly_completed = (~completed) & done
            failures[newly_completed] = ~truncated[newly_completed]
            survivals[newly_completed] = survival[newly_completed]
            completed |= newly_completed
    added = 0
    for index, buffer in enumerate(buffers):
        priorities = np.asarray(buffer["priorities"], dtype=np.float32)
        if failures[index]:
            priorities[-180:] *= float(args.failure_tail_priority)
        count = len(priorities)
        replay.add(
            np.asarray(buffer["objects"], dtype=np.float32),
            np.asarray(buffer["masks"], dtype=np.bool_),
            np.asarray(buffer["globals"], dtype=np.float32),
            np.asarray(buffer["actions"], dtype=np.int64),
            np.asarray(buffer["regrets"], dtype=np.float32),
            np.asarray(buffer["collisions"], dtype=np.bool_),
            np.full(count, 9 * 10**15 + index, dtype=np.int64),
            np.asarray(buffer["steps"], dtype=np.int32),
            priorities,
        )
        added += count
    replay.save(args.output_replay)
    result = {
        "seeds": list(seeds),
        "rendered_rgb": bool(args.rendered_rgb),
        "action_repeat_choices": list(action_repeat_choices),
        "limit_seconds": float(args.limit_seconds),
        "added_samples": added,
        "failures": int(failures.sum()),
        "survival_seconds": survivals.tolist(),
        "output_replay": str(args.output_replay.resolve()),
    }
    report = args.output_replay.with_suffix(".json")
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
