"""Replay historical model-test collision seeds with the deployed image policy."""
from __future__ import annotations
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def discover():
    sources = {}
    errors = []
    def add(seed, path):
        seed = int(seed)
        if seed >= 0:
            sources.setdefault(seed, set()).add(str(path.relative_to(ROOT)))
    for folder in (ROOT / "diagnostics", ROOT / "runs"):
        for path in sorted(folder.rglob("*")):
            if path.suffix not in (".csv", ".json") or "historical_deaths_" in str(path):
                continue
            if any(s in str(path).lower() for s in ("exact_teacher", "image_oracle", "teacher_episodes")):
                continue
            try:
                if path.suffix == ".csv":
                    with path.open(encoding="utf-8-sig", newline="") as f:
                        for r in csv.DictReader(f):
                            if not r.get("seed"):
                                continue
                            reason = r.get("termination_reason", r.get("replay_termination_reason", ""))
                            if reason == "collision" or ("collision_positions" in path.name):
                                add(r["seed"], path)
                else:
                    d = json.loads(path.read_text(encoding="utf-8-sig"))
                    if not isinstance(d, dict):
                        continue
                    for key in ("failures", "failure_seeds"):
                        for r in d.get(key, []) or []:
                            if isinstance(r, dict) and "seed" in r:
                                add(r["seed"], path)
                            elif isinstance(r, int):
                                add(r, path)
                    seeds, times = d.get("seeds", []), d.get("survival_seconds", [])
                    limit = d.get("evaluation_episode_limit_seconds", d.get("episode_limit_seconds", 120))
                    if isinstance(seeds, list) and isinstance(times, list) and len(seeds) == len(times):
                        for seed, seconds in zip(seeds, times):
                            if float(seconds) < float(limit):
                                add(seed, path)
            except (ValueError, TypeError, AttributeError) as exc:
                errors.append({"path": str(path.relative_to(ROOT)), "error": type(exc).__name__})
    return [{"seed": s, "sources": sorted(p)} for s, p in sorted(sources.items())], errors


def write_csv(path, rows, fields):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--seed-file", type=Path, help="Explicit diagnostic seeds; bypass broad discovery")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.seed_file:
        requested = json.loads(args.seed_file.read_text(encoding="utf-8"))
        assert len(requested) == len(set(requested)) and requested
        inventory = [{"seed": int(s), "sources": [str(args.seed_file)]} for s in requested]
        errors = []
    else:
        inventory, errors = discover()
    (args.output / "seed_sources.json").write_text(json.dumps({"seeds": inventory, "read_errors": errors}, indent=2), encoding="utf-8")
    print(f"Historical collision seeds: {len(inventory)}; read errors: {len(errors)}", flush=True)
    if args.inventory_only:
        return
    import torch
    import numba
    original_njit = numba.njit
    def uncached_njit(*a, **kw):
        kw["cache"] = False
        return original_njit(*a, **kw)
    numba.njit = uncached_njit
    from barrage_rl.evaluate_tracked_policy import load_tracked_agent, checkpoint_action_delay_steps
    from barrage_rl.parallel_evaluation import run_parallel_rollout
    from barrage_rl.task_spec import TARGET_TASK
    from tools.pixel_guard_receding import install_receding_guard
    from tools.plot_death_positions import plot
    checkpoint = ROOT / "diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt"
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, spec, metadata = load_tracked_agent(str(checkpoint), device,
        analytic_shield=True, analytic_shield_gate="learned_all_unsafe")
    seeds = [r["seed"] for r in inventory]
    config = {"checkpoint": str(checkpoint), "safety_threshold": agent.safety_threshold,
              "guard": "receding", "task": TARGET_TASK.manifest(), "device": str(device),
              "seed_count": len(seeds), "formal_evaluation": False,
              "scope": "Diagnostic replay of all recorded historical model-test collision seeds"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    episodes, deaths = [], []
    ef = ["seed", "survival_seconds", "termination_reason"]
    df = ["seed", "survival_seconds", "plane_center_x", "plane_center_y"]
    for start in range(0, len(seeds), 10):
        batch = seeds[start:start+10]
        install_receding_guard(agent)
        result = run_parallel_rollout(agent=agent, spec=spec, episodes=len(batch), workers=len(batch),
            seed=batch[0], episode_seeds=batch, env_kwargs=TARGET_TASK.env_kwargs(),
            wall_threshold=40, rendered_rgb=True, causal_action_delay_steps=checkpoint_action_delay_steps(metadata),
            collect_failure_diagnostics=True, failure_lookback_decisions=0)
        episodes.extend(dict(seed=s, survival_seconds=float(t), termination_reason=r)
                        for s, t, r in zip(batch, result.survival_times, result.termination_reasons))
        deaths.extend(dict(seed=int(r["seed"]), survival_seconds=float(r["survival_seconds"]),
                           plane_center_x=float(r["plane_center"][0]), plane_center_y=float(r["plane_center"][1]))
                      for r in result.failure_diagnostics)
        write_csv(args.output / "episodes.csv", episodes, ef)
        write_csv(args.output / "collision_positions.csv", deaths, df)
        print(f"Completed {len(episodes)}/{len(seeds)}; deaths {len(deaths)}", flush=True)
    plot(args.output / "collision_positions.csv", args.output / "death_map.png",
         "当前默认模型 · 历史死亡种子回放（300 发 / 120 秒）")


if __name__ == "__main__":
    main()
