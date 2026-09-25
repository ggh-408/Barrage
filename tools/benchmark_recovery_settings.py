"""Balanced, repeated single-image latency sweep; no policy changes."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import time
import json
import platform
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.window_inference import enable_window_inference
from barrage_rl.artifacts import atomic_write_json,sha256_file
from tools.pixel_guard_candidate import PixelGuardConfig,install_guard


def main():
    output=ROOT/'diagnostics/pixel_guard_recovery/settings_sweep.json'
    if output.exists():raise FileExistsError(output)
    data=np.load(ROOT/'diagnostics/success_plateau_audit_20260905/current_terminal_features.npz')
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    config=PixelGuardConfig(allow_imminent_escape=True,recovery_search=True,compiled_search=True)
    sources=[*sorted((ROOT/'barrage_rl').glob('*.py')),*[ROOT/'tools'/n for n in
        ('pixel_guard_candidate.py','pixel_recovery_planner.py','pixel_search_kernel.py')]]
    hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in sources}
    configs=[('cpu',i) for i in range(1,min(10,os.cpu_count())+1)]
    if torch.cuda.is_available():configs += [('cuda',1),('cuda',4)]
    agents={}
    for device,threads in configs:
        torch.set_num_threads(threads)
        agent,_,_=load_tracked_agent(str(checkpoint),torch.device(device),
            analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
        enable_window_inference(agent.model);install_guard(agent,config)
        agents[(device,threads)]=agent
    reference=[]
    def act(key,j):
        return int(agents[key].act_features(data['objects'][j:j+1],data['masks'][j:j+1],data['globals'][j:j+1])[0])
    torch.set_num_threads(4)
    for j in range(7):reference.append(act(('cpu',4),j))
    records=[]
    def block(key,phase,repeat,n):
        torch.set_num_threads(key[1])
        for i in range(21):act(key,i%7)
        if key[0]=='cuda':torch.cuda.synchronize()
        lat=[];mismatch=0
        for i in range(n):
            j=i%7;start=time.perf_counter()
            action=act(key,j)
            # act_features returns host actions, including transfer completion.
            lat.append((time.perf_counter()-start)*1000)
            assert 0<=action<9
            mismatch += action!=reference[j]
        row=dict(device=key[0],threads=key[1],phase=phase,repeat=repeat,
            samples=n,mean_ms=float(np.mean(lat)),p99_ms=float(np.percentile(lat,99)),
            max_ms=max(lat),action_mismatches=mismatch,latency_ms=lat)
        records.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='latency_ms'}),flush=True)
    rng=np.random.default_rng(5905)
    for repeat in range(3):
        for index in rng.permutation(len(configs)):
            block(configs[index],'sweep',repeat,140)
    def stats(key,phase):
        r=[r for r in records if (r['device'],r['threads'])==key and r['phase']==phase]
        lat=np.concatenate([r['latency_ms'] for r in r])
        return dict(device=key[0],threads=key[1],samples=len(lat),
            mean_ms=float(lat.mean()),p95_ms=float(np.percentile(lat,95)),
            p99_ms=float(np.percentile(lat,99)),max_ms=float(lat.max()),
            round_mean_ms=[r['mean_ms'] for r in r],
            action_mismatches=sum(r['action_mismatches'] for r in r))
    sweep=[stats(k,'sweep') for k in configs]
    cpu_best=min([r for r in sweep if r['device']=='cpu'],key=lambda r:r['mean_ms'])
    keys=[('cpu',4),('cpu',cpu_best['threads'])]
    gpu=[r for r in sweep if r['device']=='cuda']
    if gpu:
        best_gpu=min(gpu,key=lambda r:r['mean_ms']);keys.append(('cuda',best_gpu['threads']))
    keys=list(dict.fromkeys(keys))
    for repeat in range(3):
        for index in rng.permutation(len(keys)):block(keys[index],'confirm',repeat,350)
    confirmation=[stats(k,'confirm') for k in keys]
    eligible=[r for r in confirmation if r['action_mismatches']==0]
    best=min(eligible,key=lambda r:r['mean_ms'])
    key=(best['device'],best['threads']);torch.set_num_threads(key[1])
    start=time.perf_counter();count=0;lat=[]
    while time.perf_counter()-start<60:
        tick=time.perf_counter();action=act(key,count%7)
        lat.append((time.perf_counter()-tick)*1000)
        assert action==reference[count%7]
        count+=1
    changed=[p for p,h in hashes.items() if sha256_file(ROOT/p)!=h]
    report=dict(platform=platform.platform(),cpu=platform.processor(),torch=torch.__version__,
        logical_processors=os.cpu_count(),checkpoint_sha256=sha256_file(checkpoint),
        source_hashes=hashes,source_changes=changed,reference_actions=reference,
        sweep=sweep,confirmation=confirmation,selected=best,records=records,
        stress=dict(seconds=time.perf_counter()-start,decisions=count,
            mean_ms=float(np.mean(lat)),p99_ms=float(np.percentile(lat,99)),max_ms=max(lat)),
        scope='Seven saved RGB-derived failure frames, model plus guard, batch size one; capture and detection excluded. No formal survival evaluation.')
    atomic_write_json(output,report)
    assert not changed
    print(json.dumps({k:report[k] for k in ('sweep','confirmation','selected','stress')}),flush=True)


if __name__=='__main__':main()
