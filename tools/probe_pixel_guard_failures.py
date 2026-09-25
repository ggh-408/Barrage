"""Explicit three-episode diagnostic smoke replay, using the existing rollout."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.parallel_evaluation import run_parallel_rollout
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.artifacts import atomic_write_json, sha256_file
from tools.pixel_guard_candidate import install_guard, PixelGuardConfig


def main():
    if sys.argv[1:] != ['--smoke-test']:
        raise ValueError('explicit --smoke-test required')
    out=ROOT/'diagnostics/pixel_guard_validation/remaining_failures.json'
    if out.exists():raise FileExistsError(out)
    seeds=[877670566,782681748,419163929]
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    torch.set_num_threads(1)
    agent,spec,_=load_tracked_agent(str(checkpoint),torch.device('cuda'),
        analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    guard=install_guard(agent,PixelGuardConfig(allow_imminent_escape=True))
    original=agent.act_features_with_diagnostics
    terminal={}
    def capture(objects,masks,globals_,**kwargs):
        result=original(objects,masks,globals_,**kwargs)
        nominal,possible=guard.hazards(objects,masks,globals_)
        for row,index in enumerate(kwargs['episode_indices']):
            terminal[int(index)]={'nominal_collisions':nominal[row].tolist(),
                                  'possible_collisions':possible[row].tolist()}
        return result
    agent.act_features_with_diagnostics=capture
    started=time.perf_counter()
    rollout=run_parallel_rollout(agent=agent,spec=spec,episodes=3,workers=3,
        seed=seeds[0],episode_seeds=seeds,env_kwargs=TARGET_TASK.env_kwargs(),
        wall_threshold=40,rendered_rgb=True,causal_action_delay_steps=0,
        collect_failure_diagnostics=True,failure_lookback_decisions=36)
    payload={'formal_evaluation':False,'smoke_test':True,'seeds':seeds,
             'checkpoint_sha256':sha256_file(checkpoint),
             'elapsed_seconds':time.perf_counter()-started,
             'survival_seconds':rollout.survival_times.tolist(),
             'terminal_image_hazards':terminal,'failures':rollout.failure_diagnostics,
             'guard':guard.manifest(),
             'privileged_diagnostic_labels_are_policy_input':False}
    atomic_write_json(out,payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='failures'}),flush=True)


if __name__=='__main__':main()
