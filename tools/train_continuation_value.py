"""Train image-only root-action values from actual closed-loop RGB branches."""
from __future__ import annotations
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.artifacts import atomic_torch_save, atomic_write_json, atomic_copy, prepare_new_output
from barrage_rl.deployment import configure_image_controller
from barrage_rl.evaluate_tracked_policy import load_tracked_agent, evaluate_tracked_checkpoint, checkpoint_action_delay_steps
from barrage_rl.parallel_evaluation import run_parallel_rollout
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.tracked_policy import ActionQueryPolicy, continuation_plan_context


def select_records(source_run, maximum):
    records=[]
    for folder in sorted(source_run.glob('round*/branch_records')):
        context={r['key']:r for line in (folder/'controller_decisions.jsonl').read_text().splitlines()
                 if line.strip() for r in [json.loads(line)]}
        for path in sorted(folder.glob('*.pkl')):
            with path.open('rb') as f: record=pickle.load(f)
            row=context.get(record['key'])
            if row is None or row['exploration'] or row.get('episode_has_exploration'): continue
            record['context']=row; record['source_path']=str(path.resolve())
            # Include the run/round as part of the episode group, never split roots.
            record['group']=f"{folder.parent.name}:{row['episode_id']}"
            records.append(record)
    rng=np.random.default_rng(49051)
    groups=sorted({r['group'] for r in records})
    if len(groups)<4: raise ValueError('Need at least four independent source episodes')
    rng.shuffle(groups)
    validation_groups=set(groups[:max(1,len(groups)//5)])
    selected=[]
    for validation,limit in ((False,maximum*4//5),(True,maximum-maximum*4//5)):
        pool=[r for r in records if (r['group'] in validation_groups)==validation]
        rng.shuffle(pool)
        # Round-robin episodes retains coverage even when one failed tail is large.
        counts={}
        for record in sorted(pool,key=lambda r:0 if r['reason']=='failure_tail' else 1):
            counts.setdefault(record['group'],[]).append(record)
        while len([r for r in selected if r['validation']==validation])<limit and any(counts.values()):
            for group,rows in counts.items():
                if rows and len([r for r in selected if r['validation']==validation])<limit:
                    record=rows.pop(0);record['validation']=validation;selected.append(record)
    return selected


def label_branches(agent,spec,records,workers,states_per_batch,output):
    labels=[]; durations=[]; start=time.perf_counter()
    for offset in range(0,len(records),states_per_batch):
        batch=records[offset:offset+states_per_batch]; initial=[]
        for r in batch:
            for action in range(9):
                plan=deepcopy(r['context']['prior_plan'])
                if plan is not None:
                    plan['path']=np.asarray(plan['path'],np.int64)
                    plan['expected_plane']=np.asarray(plan['expected_plane'],np.float32)
                initial.append(replace(r['state'],forced_action=action,forced_decisions=1,
                    controller_plan=plan,continuation_seconds=1.2))
        agent.reset_state()
        result=run_parallel_rollout(agent,spec,len(initial),workers,49051000+offset,
            TARGET_TASK.env_kwargs(),40.,rendered_rgb=True,causal_action_delay_steps=0,
            initial_states=initial)
        elapsed=np.asarray([t-state.env_snapshot.physics_steps/120
                            for t,state in zip(result.survival_times,initial)]).reshape(-1,9)
        if not np.isfinite(elapsed).all(): raise RuntimeError('Nonfinite branch duration')
        target=(elapsed[:,:,None]+1e-6>=np.array([.6,1.2])[None,None,:]).astype(np.float32)
        # Collision exactly on a horizon is a failed continuation.
        for i,reason in enumerate(result.termination_reasons):
            if reason=='collision':
                target[i//9,i%9]=elapsed[i//9,i%9]>np.array([.6,1.2])+1e-6
        labels.extend(target);durations.extend(elapsed)
        atomic_write_json(output/'branch_progress.json',dict(completed=len(labels),states=len(records),elapsed_seconds=time.perf_counter()-start))
        print(f'continuation_branches states={len(labels)}/{len(records)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    return np.asarray(labels),np.asarray(durations)


def supervision_summary(records, labels):
    validation=np.asarray([r['validation'] for r in records],dtype=bool)
    result={}
    for name,selected in [('train',~validation),('validation',validation)]:
        values=labels[selected]
        result[name]=dict(states=int(selected.sum()),success_labels=int(np.sum(values==1)),
            failure_labels=int(np.sum(values==0)),
            mixed_outcome_states=int(np.sum(np.any(np.ptp(values,axis=1)>0,axis=1))))
    result['informative']=all(row['mixed_outcome_states']>0 for row in result.values())
    return result


def fit_head(model,records,labels,output):
    device=next(model.parameters()).device
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.continuation_head.parameters(): p.requires_grad_(True)
    model.eval(); queries=[]
    with torch.no_grad():
        for offset in range(0,len(records),32):
            states=[r['state'] for r in records[offset:offset+32]]
            inputs=[torch.as_tensor(np.stack([getattr(s,k) for s in states]),device=device,dtype=dtype)
                    for k,dtype in [('objects',torch.float32),('mask',torch.bool),('globals_',torch.float32)]]
            queries.append(model.forward_with_arbiter_latents(*inputs)[-1].action_queries.detach())
    queries=torch.cat(queries)
    context=torch.as_tensor(continuation_plan_context([r['context']['prior_plan'] for r in records]),device=device)
    queries=torch.cat((queries,context[:,None,:].expand(-1,9,-1)),dim=-1)
    truth=torch.as_tensor(labels,device=device)
    validation=torch.as_tensor([r['validation'] for r in records],device=device,dtype=torch.bool)
    if not validation.any() or validation.all(): raise ValueError('Episode split is empty')
    optimizer=torch.optim.AdamW(model.continuation_head.parameters(),lr=3e-4,weight_decay=1e-3)
    best=float('inf'); best_state=None; stale=0; history=[]
    for epoch in range(300):
        model.continuation_head.train();optimizer.zero_grad(set_to_none=True)
        logits=model.continuation_head(queries[~validation])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,truth[~validation])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite continuation loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.continuation_head.parameters(),1.)
        optimizer.step();model.continuation_head.eval()
        with torch.no_grad():
            predicted=model.continuation_head(queries[validation])
            val=float(torch.nn.functional.binary_cross_entropy_with_logits(predicted,truth[validation]))
        history.append(dict(epoch=epoch+1,train_loss=float(loss.detach()),validation_loss=val))
        if val<best-1e-6:
            best=val; best_state=deepcopy(model.continuation_head.state_dict()); stale=0
        else: stale+=1
        if stale>=25: break
    model.continuation_head.load_state_dict(best_state)
    atomic_write_json(output/'head_training.json',dict(history=history,best_validation_loss=best,
        train_states=int((~validation).sum()),validation_states=int(validation.sum()),
        mixed_outcome_states=int(np.sum(np.ptp(labels[:,:,1],axis=1)>0)),
        continuation_horizons=[.6,1.2],label_controller='frozen image-only controller after one forced decision',
        parameters_trained=sum(p.numel() for p in model.continuation_head.parameters())))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-states',type=int,default=160)
    p.add_argument('--workers',type=int,default=9)
    p.add_argument('--states-per-batch',type=int,default=4)
    p.add_argument('--search-workers',type=int,default=1)
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    prepare_new_output(args.output)
    torch.manual_seed(49051);torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source=args.source_run/'best.pt'
    agent,spec,checkpoint=load_tracked_agent(str(source),device,analytic_shield=True,analytic_shield_gate='learned_all_unsafe')
    if spec.expected_bullet_count!=300 or checkpoint_action_delay_steps(checkpoint)!=0:
        raise ValueError('Continuation experiment requires the aligned 300-bullet zero-delay source')
    configure_image_controller(agent,'receding',search_workers=args.search_workers)
    records=select_records(args.source_run,args.max_states)
    config=json.loads((args.source_run/'config.json').read_text())
    config.update(output_dir=str(args.output),initial_checkpoint=str(source.resolve()),
        training_stage='continuation_value',paired_evaluation_source=str(args.source_run.resolve()),
        continuation_horizons=[.6,1.2],continuation_weight=1.,smoke_test=args.smoke_test)
    atomic_write_json(args.output/'config.json',config)
    atomic_write_json(args.output/'branch_manifest.json',[{k:r[k] for k in ('key','source_path','group','reason','validation')} for r in records])
    labels,durations=label_branches(agent,spec,records,args.workers,args.states_per_batch,args.output)
    np.savez(args.output/'continuation_labels.npz',labels=labels,durations=durations,
        objects=np.stack([r['state'].objects for r in records]),masks=np.stack([r['state'].mask for r in records]),
        globals=np.stack([r['state'].globals_ for r in records]),validation=np.asarray([r['validation'] for r in records]),
        route_context=continuation_plan_context([r['context']['prior_plan'] for r in records]))
    supervision=supervision_summary(records,labels)
    atomic_write_json(args.output/'supervision_summary.json',supervision)
    if not args.smoke_test and not supervision['informative']:
        raise ValueError('Continuation labels need mixed root-action outcomes in both episode splits. '
            'The aligned checkpoint and branch labels are preserved. Collect more failure/near-miss '
            'episodes or label more source states in a new output directory before fitting the head.')
    hparams=dict(checkpoint['model_hparams']);hparams.update(continuation_horizons=(.6,1.2),continuation_weight=0.)
    model=ActionQueryPolicy(spec,**hparams).to(device)
    missing,unexpected=model.load_state_dict(checkpoint['model'],strict=False)
    if unexpected or any(not k.startswith('continuation_head.') for k in missing): raise ValueError('Unexpected migration')
    fit_head(model,records,labels,args.output)
    model.continuation_weight=1.;hparams['continuation_weight']=1.
    candidate={k:checkpoint[k] for k in ('model_version','tracked_policy_spec','observation_size','inference_head')}
    candidate.update(model_version=model.model_version, model=model.state_dict(),model_hparams=hparams,config=config,round=1,
                     training_metrics={'continuation_training_states':len(records)})
    atomic_torch_save(candidate,args.output/'candidate.pt')
    summary=evaluate_tracked_checkpoint(str(args.output/'candidate.pt'),episodes=2 if args.smoke_test else 200,
        workers=args.workers,seed=int(config['evaluation_seed']),output_dir=str(args.output/'evaluation'),
        smoke_test=args.smoke_test,episode_limit_seconds=5 if args.smoke_test else 120,
        pixel_guard='receding',search_workers=args.search_workers)
    baseline=json.loads((args.source_run/'best_summary.json').read_text())
    promote=summary['success_at_limit']>baseline['success_at_limit']
    from barrage_rl.train_tracked_policy import _write_history
    from barrage_rl.plot import save_round_summary_plot
    history=[{**baseline,'round':0.,'phase':'aligned_baseline'},
             {**summary,'round':1.,'phase':'closed_loop_continuation','checkpoint_promoted':float(promote)}]
    _write_history(args.output/'round_summaries.csv',history)
    save_round_summary_plot(args.output/'round_summaries.csv',args.output/'results.png',args.output/'config.json')
    atomic_copy(args.output/'candidate.pt' if promote else source,args.output/'best.pt')
    atomic_write_json(args.output/'best_summary.json',summary if promote else baseline)
    atomic_copy(args.output/'best.pt',args.output/'latest.pt')
    atomic_write_json(args.output/'selection.json',dict(promoted=promote,selection_metric='success_at_limit',
        earlier_checkpoint_retained_on_tie=True,baseline=baseline['success_at_limit'],candidate=summary['success_at_limit'],
        baseline_checkpoint=str(source.resolve())))
    print(f'continuation_complete promoted={promote} success_at_limit={summary["success_at_limit"]}',flush=True)

if __name__=='__main__':main()
