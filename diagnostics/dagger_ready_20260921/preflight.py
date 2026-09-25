"""Bounded preparation tests: collection, gradients, checkpoint routing and 20 episodes. No optimizer updates."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,json,time,tempfile
from pathlib import Path
from dataclasses import replace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import train_targeted_dagger as entry
from barrage_rl import train_tracked_policy as training
from barrage_rl import evaluate_tracked_policy as evaluation
from barrage_rl.artifacts import atomic_write_json,contents_equal

def main():
    if (HERE/'preflight.json').exists() or (HERE/'evaluation').exists():raise FileExistsError('Preflight outputs already exist')
    torch.set_num_threads(1);config=entry.load_config();entry.validate_config(config)
    source_files=tuple(dict.fromkeys((*training._MANIFEST_SOURCE_FILES,*entry.EXTRA_SOURCES,
        'diagnostics/dagger_ready_20260921/training_config.json',
        'diagnostics/dagger_ready_20260921/start_training.ps1')))
    frozen={p:(ROOT/p).read_bytes() for p in source_files}
    # Any accidental attempt to perform a training update fails immediately.
    def forbidden(*args,**kwargs):raise RuntimeError('Optimizer updates are forbidden during preflight')
    torch.optim.AdamW.step=forbidden
    rejected=0
    for invalid in (replace(config,evaluation_episodes=50),replace(config,max_objects=300),
                    replace(config,deployment_rgb_observation=False),replace(config,smoke_test=True)):
        try:entry.validate_config(invalid)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid training config accepted')
    with tempfile.TemporaryDirectory(dir=HERE,prefix='nonempty_') as folder:
        (Path(folder)/'keep.txt').write_text('preserve')
        try:entry.validate_config(replace(config,output_dir=folder))
        except FileExistsError:rejected+=1
        else:raise AssertionError('Non-empty output accepted')
    original_load=evaluation.load_tracked_agent
    agent,spec,metadata=original_load(str(ROOT/config.initial_checkpoint),torch.device('cuda'),analytic_shield=False)
    model=agent.model;before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'model':before},HERE/'initial_model_reference.pt')
    entry.install_runtime()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    assert not optimizer.state
    probe=training._checkpoint(model,optimizer,config,spec,0,{})
    assert probe['experimental_controller']==entry.CONTROLLER
    torch.save(probe,HERE/'untrained_probe.pt')
    try:original_load(str(HERE/'untrained_probe.pt'),torch.device('cpu'))
    except ValueError as e:assert 'experimental image controller' in str(e)
    else:raise AssertionError('Default loader silently accepted experimental controller')
    restored,_,_=evaluation.load_tracked_agent(str(HERE/'untrained_probe.pt'),torch.device('cpu'))
    assert contents_equal(before,restored.model.state_dict());del restored,probe
    same=training._checkpoint_selection_key({'success_at_limit':.9,'model_mean':1},'success_at_limit',120)
    assert same==training._checkpoint_selection_key({'success_at_limit':.9,'model_mean':999},'success_at_limit',120)
    assert not same>same
    collect_config=replace(config,output_dir=str(HERE/'collection_probe'),samples_per_round=216,replay_capacity=512)
    replay=training.TrackedReplay(512,spec,len(config.safety_horizons))
    processes=[];base_pipeline=training.ParallelTrackedDaggerEnv
    class ObservedPipeline(base_pipeline):
        def __enter__(self):
            result=super().__enter__()
            processes.extend(self._processes)
            assert len(self._processes)==9 and all(p.is_alive() for p in self._processes)
            return result
    training.ParallelTrackedDaggerEnv=ObservedPipeline
    collection=training._collect_round(model,replay,collect_config,spec,torch.device('cuda'),1)
    assert replay.size==216 and len(set(map(int,replay.episode_ids[:replay.size])))==36
    assert not any(p.is_alive() for p in processes)
    assert collection['failed_episodes_discovered']==0
    assert np.all(np.isfinite(replay.regrets[:replay.size]))
    assert np.all(np.isfinite(replay.priorities[:replay.size])) and np.all(replay.priorities[:replay.size]>0)
    print('Collection passed: 36 environments, 9 concurrent workers, 216 states; workers shut down cleanly',flush=True)
    training.ParallelTrackedDaggerEnv=base_pipeline
    # Test the actual loss and backward path at the delivered batch size.
    model.train();model.zero_grad(set_to_none=True)
    data=training._batch(replay,np.arange(512)%replay.size,torch.device('cuda'))
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    loss,metrics=training._loss(model,*data,config)
    assert torch.isfinite(loss)
    loss.backward();torch.cuda.synchronize()
    grad_time=time.perf_counter()-started
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g)>0 for g in grads)
    assert contents_equal(before,model.state_dict()) and not optimizer.state
    gradient=dict(batch_size=512,loss=float(loss.detach()),backward_seconds=grad_time,
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),finite_gradients=True,weights_unchanged=True,optimizer_steps=0)
    model.zero_grad(set_to_none=True);model.eval();del data,loss,grads,optimizer,model,agent,replay
    torch.cuda.empty_cache()
    atomic_write_json(HERE/'preflight_partial.json',dict(config_rejections=rejected,collection=collection,
        concurrent_workers=9,gradient=gradient,checkpoint_controller_guard=True,tie_keeps_earlier=True,optimizer_steps=0))
    seeds=json.loads((ROOT/'diagnostics/planner_readiness_20260921/design.json').read_text())['paired_seeds']
    initial=list(config.collection_seed_list)+[config.collection_seed+training._COLLECTION_ROUND_SEED_STRIDE+i for i in range(9,36)]
    assert len(set(seeds+initial))<=50
    print('Starting 20 complete evaluation episodes with the delivered parallel controller; no training loop has run',flush=True)
    summary=evaluation.evaluate_tracked_checkpoint(str(HERE/'untrained_probe.pt'),episodes=20,episode_seeds=seeds,
        workers=9,evaluation_batch_size=20,output_dir=str(HERE/'evaluation'),device_name='cuda',smoke_test=True,
        episode_limit_seconds=120,bullet_count=300,targeted_bullet_probability=.10,rendered_rgb=True,
        causal_action_delay_steps=0,analytic_shield=False,pixel_guard='receding',search_workers=9)
    assert summary['success_at_limit']==1.0
    assert all((ROOT/name).read_bytes()==value for name,value in frozen.items())
    current=torch.load(ROOT/config.initial_checkpoint,map_location='cpu',weights_only=False)
    assert contents_equal(before,current['model'])
    for name,value in frozen.items():
        target=HERE/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(value)
    result=dict(passed=True,optimizer_steps=0,training_started=False,training_output_created=(ROOT/config.output_dir).exists(),
        complete_evaluation_episodes=20,partial_collection_environments=36,total_distinct_seeds=len(set(seeds+initial)),
        success_at_limit=summary['success_at_limit'],config_rejections=rejected,collection=collection,gradient=gradient,
        concurrent_workers=9,workers_cleanly_stopped=True,checkpoint_controller_guard=True,tie_keeps_earlier=True,
        source_files=list(source_files),source_contents_unchanged=True,checkpoint_model_unchanged=True,
        formal_evaluation=False,evidence='Prepared parallel controller, existing collection/loss/evaluation implementations')
    atomic_write_json(HERE/'preflight.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
