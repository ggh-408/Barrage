"""Export active policy/teacher-cost weights without retired risk parameters."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from barrage_rl.artifacts import atomic_torch_save, contents_equal
from barrage_rl.evaluate_tracked_policy import load_tracked_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    agent, spec, source = load_tracked_agent(str(args.source), torch.device("cpu"))
    model = agent.model
    active = model.state_dict()
    expected = {k: v for k, v in source["model"].items()
                if k not in {"collision_head.weight", "collision_head.bias"}}
    assert contents_equal(active, expected)
    config_names = ("bullet_size", "bullet_count", "targeted_bullet_probability",
        "observation_size", "action_repeat", "physics_fps", "teacher_reaction_seconds",
        "collection_causal_action_delay_steps", "evaluation_causal_action_delay_steps",
        "evaluation_episode_limit_seconds")
    config = {k: source.get("config", {})[k] for k in config_names if k in source.get("config", {})}
    config.update(physics_fps=120, bullet_count=300, targeted_bullet_probability=0.10,
        collection_causal_action_delay_steps=0, evaluation_causal_action_delay_steps=0)
    checkpoint = {"model": active, "model_version": model.model_version,
        "tracked_policy_spec": asdict(spec), "config": config,
        "model_hparams": {k: getattr(model, k) for k in ("action_count", "width",
            "attention_layers", "attention_heads", "safety_horizons", "geometry_statistics",
            "continuation_horizons", "continuation_weight")},
        "observation_size": int(source.get("observation_size", 192)),
        "inference_head": "policy", "policy_architecture": "policy_teacher_cost",
        "source_checkpoint": str(args.source.resolve()), "training_performed": False}
    atomic_torch_save(checkpoint, args.output)
    loaded, _, _ = load_tracked_agent(str(args.output), torch.device("cpu"))
    assert contents_equal(active, loaded.model.state_dict())
    print(f"Exported active weights: {args.output}")


if __name__ == "__main__":
    main()
