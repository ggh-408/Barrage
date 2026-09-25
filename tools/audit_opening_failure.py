"""Observe known opening failures; privileged state is diagnosis-only."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
import json
import argparse
from collections import deque
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from barrage_rl import parallel_evaluation
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.deployment import configure_image_controller
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.artifacts import atomic_write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-test',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1)
    seeds=[282467554,282467667]
    agent,spec,_=load_tracked_agent(str(ROOT/'runs/visual_set_v51/candidate.pt'),torch.device('cuda'),
        analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    configure_image_controller(agent,'receding',search_workers=1)
    guard=agent._receding_pixel_guard
    traces={i:deque(maxlen=30) for i in range(len(seeds))}
    def observe(key,event):
        traces[int(key)].append({k:v.tolist() if hasattr(v,'tolist') else v for k,v in event.items()})
    guard._decision_observer=observe
    kwargs=TARGET_TASK.env_kwargs();kwargs['max_episode_seconds']=4.
    result=parallel_evaluation.run_parallel_rollout(agent,spec,2,2,seeds[0],kwargs,40.,
        episode_seeds=seeds,rendered_rgb=True,causal_action_delay_steps=0,
        collect_failure_diagnostics=True,failure_lookback_decisions=15)
    report=dict(scope='Two known early failure seeds; diagnostic smoke, no model fitting',
        formal_evaluation=False,seeds=seeds,teacher_reaction_seconds=.10,
        survival_seconds=result.survival_times.tolist(),termination_reasons=result.termination_reasons,
        failures=result.failure_diagnostics,traces={str(seeds[i]):list(rows) for i,rows in traces.items()})
    atomic_write_json(a.output,report)
    print(json.dumps({k:report[k] for k in ['seeds','survival_seconds','termination_reasons']}),flush=True)


if __name__=='__main__':main()
