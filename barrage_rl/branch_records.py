"""Training-only snapshots and image-controller decision provenance."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import pickle
import numpy as np


SNAPSHOT_STRIDE = 9


def record_key(index, serial, step):
    return f"env{int(index)}_episode{int(serial)}_step{int(step)}"


class BranchRecorder:
    """Keep 2.1 seconds of sparse history and bounded near-miss examples."""
    def __init__(self, output, index):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.index = index
        self.tail = deque(maxlen=8)
        self.near_count = 0
        self.serial = -1
        self.last_near_step = -90
        self.near_buckets=set()
        self.near_candidates={}

    def observe(self, env, extractor, rendered, features, serial, safety):
        if serial != self.serial:
            self.flush()
            self.tail.clear(); self.near_count=0; self.last_near_step=-90
            self.near_buckets.clear()
            self.near_candidates.clear()
            self.serial=serial
        step=int(env.episode_steps)
        if step % SNAPSHOT_STRIDE: return
        from .parallel_evaluation import ParallelRolloutInitialState
        state=ParallelRolloutInitialState(env.capture_state(),deepcopy(extractor),
            deepcopy(vars(rendered.semanticizer)), *(np.array(x,copy=True) for x in features),
            forced_action=0,forced_decisions=1)
        key=record_key(self.index,serial,step)
        item=dict(key=key,env_index=self.index,episode_serial=int(serial),episode_step=step,
                  state=state,teacher_fixed_action_safety=np.array(safety,copy=True))
        self.tail.append(item)
        if step>=90 and (step-90)%900==0:
            self.save(item,'uniform_control')
        bucket=int(env.physics_steps//3600)
        if step>=30 and (np.sum(safety[0])>=5 or np.sum(safety[1])>=7):
            distances=features[0][features[1],4]
            priority=(int(np.sum(safety[0])),int(np.sum(safety[1])),
                      -float(distances.min()) if len(distances) else 0.)
            previous=self.near_candidates.get(bucket)
            if previous is None or priority>previous[0]:
                self.near_candidates[bucket]=(priority,item)

    def save(self,item,reason):
        path=self.output/(item['key']+'.pkl')
        # Preserve the original snapshot, but promote its sampling category if
        # the episode later fails. A pre-saved uniform/near-miss record must not
        # disappear from the failure-tail pool. Later flushes cannot demote it.
        if path.exists():
            priority={'uniform_control':0,'near_miss_candidate':1,'failure_tail':2}
            with path.open('rb') as f:
                saved=pickle.load(f)
            if priority[reason]<=priority[saved['reason']]: return
            from .artifacts import _replace_with_retry
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:
                pickle.dump({**saved,'reason':reason},f,protocol=pickle.HIGHEST_PROTOCOL)
            _replace_with_retry(temporary,path)
            return
        with path.open('xb') as f:
            pickle.dump({**item,'reason':reason},f,protocol=pickle.HIGHEST_PROTOCOL)

    def failed(self):
        for item in self.tail: self.save(item,'failure_tail')

    def flush(self):
        for _,item in self.near_candidates.values(): self.save(item,'near_miss_candidate')


def plain_plan(plan):
    if plan is None: return None
    return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in plan.items()}


class DecisionRecorder:
    def __init__(self,output):
        self.path=Path(output)/'controller_decisions.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('x',encoding='utf-8')
        self.explored_episodes=set()

    def before(self,agent,pipeline,episode_ids):
        guard=getattr(agent,'_receding_pixel_guard',None)
        indices=np.flatnonzero(pipeline.episode_steps % SNAPSHOT_STRIDE == 0)
        self.traces={}
        wanted={int(episode_ids[i]) for i in indices}
        def observe(key,event):
            if key in wanted:
                self.traces[key]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in event.items()}
        if guard is not None: guard._decision_observer=observe
        return [(int(i),plain_plan(deepcopy(guard._plans.get(int(episode_ids[i])))) if guard else None)
                for i in indices]

    def after(self,agent,pipeline,episode_ids,before,actions,explored,diagnostics):
        guard=getattr(agent,'_receding_pixel_guard',None)
        self.explored_episodes.update(int(e) for e,x in zip(episode_ids,explored) if x)
        for i,prior in before:
            row=dict(key=record_key(i,pipeline.episode_serial[i],pipeline.episode_steps[i]),
                episode_id=int(episode_ids[i]),behavior_action=int(actions[i]),exploration=bool(explored[i]),
                episode_has_exploration=int(episode_ids[i]) in self.explored_episodes,
                prior_plan=prior,chosen_plan=plain_plan(guard._plans.get(int(episode_ids[i]))) if guard else None,
                raw_action=int(diagnostics['raw_policy_actions'][i]),
                learned_action=int(diagnostics['learned_filter_actions'][i]),
                all_actions_unsafe=bool(diagnostics['all_actions_unsafe'][i]))
            row['arbitration']=self.traces.get(int(episode_ids[i]))
            self.file.write(json.dumps(row)+'\n')
        self.file.flush()

    def close(self): self.file.close()
