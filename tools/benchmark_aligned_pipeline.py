"""Measure the real 300-bullet collection and optimizer paths without digests."""
from __future__ import annotations
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys
import time
import subprocess
import threading
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, nargs='+', default=[8, 9, 10])
    p.add_argument('--envs', type=int, default=36)
    p.add_argument('--decisions', type=int, default=32)
    p.add_argument('--batches', type=int, nargs='+', default=[768, 1024, 1280])
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--guard', choices=['off', 'receding'], default='off')
    p.add_argument('--search-workers', type=int, default=1)
    p.add_argument('--stress-seconds', type=float, default=0)
    p.add_argument('--record-branches', action='store_true')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    from barrage_rl.system_resources import memory_status, process_resources
    import numba
    njit = numba.njit
    def uncached(*args, **kwargs):
        kwargs['cache'] = False
        return njit(*args, **kwargs)
    numba.njit = uncached
    from barrage_rl.evaluate_tracked_policy import load_tracked_agent
    from barrage_rl.tracked_collection import ParallelTrackedDaggerEnv
    from barrage_rl.task_spec import TARGET_TASK
    from barrage_rl.train_tracked_policy import TrackedDAggerConfig, _loss
    from tools.pixel_guard_receding import install_receding_guard, RecedingGuardConfig
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cp = ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    agent, old_spec, metadata = load_tracked_agent(str(cp),device,analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    spec = replace(old_spec, expected_bullet_count=300)
    agent.model.spec = spec
    if a.guard == 'receding':
        install_receding_guard(agent, RecedingGuardConfig(search_workers=a.search_workers))
    report = dict(hardware=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),
        memory_bytes=memory_status()['total_bytes'],python=sys.version,torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
        settings=vars(a).copy(), collection=[], optimizer=[])
    report['settings']['output']=str(a.output)
    resource_stop=threading.Event(); gpu_samples=[]
    def sample_gpu():
        while not resource_stop.is_set():
            result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,utilization.memory',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=0x08000000)
            if result.returncode==0:
                gpu_samples.append([float(x.strip()) for x in result.stdout.strip().split(',')])
            resource_stop.wait(2.)
    sampler=threading.Thread(target=sample_gpu,daemon=True);sampler.start()
    reference = None
    final_data = None
    for workers in a.workers:
        agent.reset_state()
        from barrage_rl.branch_records import DecisionRecorder
        folder=a.output.parent/(a.output.stem+'_snapshots')/str(workers)
        recorder=DecisionRecorder(folder) if a.record_branches else None
        start=time.perf_counter()
        with ParallelTrackedDaggerEnv(env_count=a.envs,workers=workers,seed=91001000,
            env_kwargs=TARGET_TASK.env_kwargs(),spec=spec,teacher_kind='exact',
            teacher_reaction_seconds=.10,deployment_rgb_observation=True,causal_action_delay_steps=0,
            branch_output_dir=str(folder) if recorder else '') as pipeline:
            startup=time.perf_counter()-start
            pids=[x.pid for x in pipeline._processes]
            before={pid:process_resources(pid)['cpu_seconds'] for pid in pids}
            frames=[]; elapsed_policy=0.; start=time.perf_counter(); steps=0
            while steps<a.decisions or time.perf_counter()-start<a.stress_seconds:
                t=time.perf_counter()
                ids=np.arange(a.envs)
                if recorder:
                    prior=recorder.before(agent,pipeline,ids)
                    actions,diag=agent.act_features_with_diagnostics(pipeline.objects,pipeline.masks.astype(bool),pipeline.globals,episode_indices=ids)
                    recorder.after(agent,pipeline,ids,prior,actions,np.zeros(a.envs,bool),diag)
                else:
                    actions=agent.act_features(pipeline.objects,pipeline.masks.astype(bool),pipeline.globals,episode_indices=ids)
                elapsed_policy+=time.perf_counter()-t
                if steps<a.decisions:
                    frames.append((actions.copy(),pipeline.objects.copy(),pipeline.masks.copy(),pipeline.globals.copy(),pipeline.collisions.copy()))
                final_data=tuple(np.array(x,copy=True) for x in (pipeline.objects,pipeline.masks,pipeline.globals,pipeline.teacher_actions,pipeline.regrets,pipeline.collisions))
                done,_,_=pipeline.step(actions)
                if done.any(): agent.reset_state(np.flatnonzero(done))
                steps+=1
            elapsed=time.perf_counter()-start
            cpu={str(pid):process_resources(pid)['cpu_seconds']-before[pid] for pid in pids}
            rss=sum(process_resources(pid)['rss_bytes'] for pid in pids)+process_resources(os.getpid())['rss_bytes']
        equal=True if reference is None else all(np.array_equal(x,y) for f,g in zip(frames,reference) for x,y in zip(f,g))
        if not equal: raise RuntimeError('Worker configuration changed the trajectory')
        if reference is None: reference=frames
        if recorder: recorder.close()
        row=dict(workers=workers,envs=a.envs,decisions=steps,startup_seconds=startup,elapsed_seconds=elapsed,
                 states_per_second=steps*a.envs/elapsed,policy_seconds=elapsed_policy,worker_cpu_seconds=cpu,
                 processes_rss_bytes=rss,direct_trajectory_equal=equal)
        report['collection'].append(row); print(json.dumps(row),flush=True)
    model=agent.model; model.train(); cfg=TrackedDAggerConfig()
    for batch_size in a.batches:
        ix=np.arange(batch_size)%len(final_data[0])
        tensors=tuple(torch.as_tensor(x[ix],device=device,dtype=dt) for x,dt in zip(final_data,
            (torch.float32,torch.bool,torch.float32,torch.long,torch.float32,torch.float32)))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
        def step():
            opt.zero_grad(set_to_none=True)
            loss,_=_loss(model,*tensors,cfg)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        for _ in range(2): step()
        if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter(); steps=0
        while steps<a.iterations or time.perf_counter()-start<a.stress_seconds:
            step(); steps+=1
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        row=dict(batch_size=batch_size,iterations=steps,seconds=elapsed,samples_per_second=steps*batch_size/elapsed,
            peak_vram_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0)
        report['optimizer'].append(row);print(json.dumps(row),flush=True)
        del tensors,opt
        if device.type=='cuda': torch.cuda.empty_cache()
    resource_stop.set();sampler.join(timeout=5.)
    report['gpu_samples']=[dict(utilization_percent=x[0],memory_used_mib=x[1],memory_controller_utilization_percent=x[2]) for x in gpu_samples]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)

if __name__=='__main__': main()
