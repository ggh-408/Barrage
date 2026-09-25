"""Observe slot pressure on known failures using the existing parallel RGB rollout."""
from __future__ import annotations

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from barrage_rl.image_oracle import PersistentImageTracker
from barrage_rl import parallel_evaluation

_original_trim = PersistentImageTracker._trim_tracks
_original_clear = PersistentImageTracker.clear
_original_attribution = parallel_evaluation._failure_attribution


def observed_clear(self):
    _original_clear(self)
    self._slot_audit = dict(updates=0, maximum_candidates=0, maximum_retained=0,
                           capacity_reached_updates=0, overflow_updates=0,
                           discarded_tracks=0)


def observed_trim(self, tracks):
    count = len(tracks)
    result = _original_trim(self, tracks)
    stats = self._slot_audit
    stats["updates"] += 1
    stats["maximum_candidates"] = max(stats["maximum_candidates"], count)
    stats["maximum_retained"] = max(stats["maximum_retained"], len(result))
    stats["capacity_reached_updates"] += int(count >= self.target_track_count)
    stats["overflow_updates"] += int(count > self.target_track_count)
    stats["discarded_tracks"] += max(0, count - len(result))
    return result


def observed_attribution(env, extractor, **kwargs):
    result = _original_attribution(env, extractor, **kwargs)
    result["slot_audit"] = dict(extractor.tracker._slot_audit)
    result["model_max_objects"] = extractor.spec.max_objects
    result["tracker_capacity"] = extractor.tracker.target_track_count
    result["terminal_model_token_count"] = int(extractor._features()[1].sum())
    return result


PersistentImageTracker.clear = observed_clear
PersistentImageTracker._trim_tracks = observed_trim
parallel_evaluation._failure_attribution = observed_attribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", required=True,
                        help="Required: targeted diagnosis, not a formal success estimate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    import numba
    original_njit = numba.njit
    def uncached_njit(*a, **kw):
        kw["cache"] = False
        return original_njit(*a, **kw)
    numba.njit = uncached_njit

    from dataclasses import asdict
    from barrage_rl.evaluate_tracked_policy import load_tracked_agent, checkpoint_action_delay_steps
    from barrage_rl.task_spec import TARGET_TASK
    from tools.pixel_guard_receding import install_receding_guard

    seeds = [1577244970, 1194734942, 890779774, 1482728399]
    checkpoint = ROOT / "diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt"
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, spec, metadata = load_tracked_agent(str(checkpoint), device,
        analytic_shield=True, analytic_shield_gate="learned_all_unsafe")
    guard = install_receding_guard(agent)
    started = time.perf_counter()
    print(f"Diagnostic rollout: {len(seeds)} episodes, {len(seeds)} workers, {device}", flush=True)
    result = parallel_evaluation.run_parallel_rollout(
        agent=agent, spec=spec, episodes=len(seeds), workers=len(seeds),
        seed=seeds[0], episode_seeds=seeds, env_kwargs=TARGET_TASK.env_kwargs(),
        wall_threshold=40, rendered_rgb=True,
        causal_action_delay_steps=checkpoint_action_delay_steps(metadata),
        collect_failure_diagnostics=True, failure_lookback_decisions=0)
    report = dict(scope="Four known current failures; observational smoke diagnosis",
        formal_evaluation=False, checkpoint=str(checkpoint), task=TARGET_TASK.manifest(),
        spec=asdict(spec), model_version=metadata.get("model_version"),
        model_hparams=metadata.get("model_hparams", {}),
        model_parameter_count=sum(p.numel() for p in agent.model.parameters()),
        safety_threshold=agent.safety_threshold, guard=guard.manifest(),
        device=str(device), workers=len(seeds), elapsed_seconds=time.perf_counter()-started,
        episodes=[dict(seed=s, survival_seconds=float(t), termination_reason=r)
                  for s, t, r in zip(seeds, result.survival_times, result.termination_reasons)],
        failures=result.failure_diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"episodes": report["episodes"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
