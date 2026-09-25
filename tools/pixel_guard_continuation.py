"""Image-only recovery memory and coherent uncertainty arbitration."""
from dataclasses import replace
import time
import numpy as np
import torch
from barrage_rl.runtime_core import ACTION_VECTORS


class ContinuationMixin:
    def _trace_decision(self, key, reason, incumbent, action, **details):
        observer=getattr(self,'_decision_observer',None)
        if observer is not None:
            observer(key,dict(reason=reason,incumbent=int(incumbent),action=int(action),**details))

    def __init__(self, config):
        super().__init__(config)
        self._plans = {}
        self._context_indices = None
        self._context_steps = 1
        self.counters.update(plan_reuses=0, plan_invalidations=0, interval_conflicts=0, model_releases=0,horizon_searches=0)
        from tools.pixel_receding_kernel import beam_paths, assess_paths
        args=(np.array([410,410],np.float32),self.half_size,
              np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),np.ones(1,np.float32))
        beam_paths(*args,self.table,self.integral,ACTION_VECTORS,1,1,48.,1.)
        assess_paths(*args,self.table,ACTION_VECTORS,np.zeros((1,1),np.int64),np.ones(1,np.int64))
        self._assess(*args,np.zeros((1,1),np.int64),np.ones(1,np.int64))

    def _assess(self,plane,half_size,bullets,velocity,error,paths,lengths):
        from tools.pixel_receding_kernel import assess_paths,assess_path_intervals
        scenarios=assess_paths(plane,half_size,bullets,velocity,error,self.table,ACTION_VECTORS,paths,lengths)
        intervals=assess_path_intervals(plane,half_size,bullets,velocity,error,self.table,
                                      self.integral,ACTION_VECTORS,paths,lengths)
        return np.column_stack((scenarios,intervals))

    def reset(self, episode_indices=None):
        if episode_indices is None:
            self._plans.clear()
        else:
            for i in np.asarray(episode_indices).reshape(-1): self._plans.pop(int(i),None)

    def manifest(self):
        return {**super().manifest(),
                'variant':'receding_continuation',
                'uncertainty_scenarios':9,
                'uncertainty_semantics':'coherent deterministic offsets; scenario counts are not probabilities',
                'safety_certificate':'full independent per-bullet pixel-offset rectangles over the complete horizon',
                'plan_memory':'image-derived suffix, revalidated each decision and reset per episode'}

    def _geometry(self,o,m,g):
        cfg=self.recovery_config
        observed=g[:2]*820
        valid=m & (o[:,9]==0) & (o[:,8]>=.15)
        valid &= np.linalg.norm(o[:,:2]*820,axis=1)<24+480*cfg.search_depth/30
        selected=o[valid]
        error=np.where(selected[:,15]>.5,
            np.maximum(2/np.maximum(selected[:,10],2/30),cfg.velocity_error_floor),240)
        return (observed-self.centroid_bias,observed+selected[:,:2]*820-self.bullet_bias,
                (selected[:,2:4]+g[None,6:8])*240,error)

    def apply(self, selection, objects, masks, globals_):
        if not self.recovery_config.preserve_plans and not self.recovery_config.interval_consensus:
            return super().apply(selection,objects,masks,globals_)
        from tools.pixel_receding_kernel import beam_paths,assess_paths
        started=time.perf_counter()
        nominal,possible=self.hazards(objects,masks,globals_)
        sn,sp=self.commit_guard.hazards(objects,masks,globals_)
        actions=selection.actions.detach().cpu().numpy().copy()
        risk=selection.immediate_risk.detach().cpu().numpy()
        all_unsafe=selection.all_unsafe.detach().cpu().numpy()
        cfg=self.recovery_config
        ids=np.arange(len(actions)) if self._context_indices is None else np.asarray(self._context_indices).reshape(-1)
        if len(ids)!=len(actions): raise ValueError('episode_indices must match the action batch')
        for row,incumbent in enumerate(actions):
            key=int(ids[row])
            o,m,g=objects[row],masks[row],globals_[row]
            plane,bullets,velocity,error=self._geometry(o,m,g)
            prior=self._plans.get(key)
            if prior is not None and (self._context_steps!=1 or np.max(np.abs(plane-prior['expected_plane']))>1.5):
                prior=None
                self._plans.pop(key,None)
                self.counters['plan_invalidations']+=1
            young=m & (o[:,15]<.5) & (o[:,9]==0) & (o[:,8]>=.15)
            young &= np.linalg.norm(o[:,:2]*820,axis=1)<40
            endpoint=np.clip(plane+ACTION_VECTORS[incumbent]*32,self.half_size,820-self.half_size)
            wall=np.minimum(endpoint-self.half_size,820-self.half_size-endpoint).min()
            geometry_gate=possible[row,incumbent] or np.count_nonzero(~possible[row])<=cfg.trigger_safe_actions
            wall_gate=wall<cfg.wall_reserve_pixels
            self.counters['decisions']+=1
            trigger=all_unsafe[row] or geometry_gate or wall_gate or young.any()
            if cfg.interval_consensus and not sp[row,incumbent] and not wall_gate:
                direct=np.full((1,cfg.search_depth),incumbent,np.int64)
                certificate=self._assess(plane,self.half_size,bullets,velocity,error,direct,
                                         np.array([cfg.search_depth],np.int64))[0]
                if certificate[5]==1 and certificate[4]>=cfg.wall_reserve_pixels:
                    self._plans.pop(key,None)
                    self.counters['model_releases']+=int(prior is not None)
                    self._trace_decision(key,'model_full_horizon',incumbent,incumbent,metrics=certificate)
                    continue
                self.counters['horizon_searches']+=int(not trigger and prior is None)
                trigger=True
            if not trigger and prior is None:
                self._trace_decision(key,'no_recovery_trigger',incumbent,incumbent)
                continue
            retained_paths=None
            retained_metrics=None
            if prior is not None and cfg.preserve_plans:
                suffix=prior['path'][:cfg.search_depth-1]
                retained_paths=np.repeat(np.arange(9)[:,None],cfg.search_depth,axis=1)
                retained_paths[:,:len(suffix)]=suffix[None]
                retained_metrics=self._assess(plane,self.half_size,bullets,velocity,error,
                    retained_paths,np.full(9,cfg.search_depth,np.int64))
                viable=np.flatnonzero((retained_metrics[:,5]==1)&(retained_metrics[:,4]>=cfg.wall_reserve_pixels))
                # Extend the suffix before reuse. A dwindling safe prefix gives
                # no evidence that there will be an exit at its endpoint.
                if len(viable) and not wall_gate and not sp[row,int(suffix[0])]:
                    extension=int(suffix[-1]) if int(suffix[-1]) in viable else int(viable[0])
                    action=int(suffix[0])
                    actions[row]=action
                    remaining=prior.get('remaining',len(prior['path']))-1
                    if remaining>0:
                        expected=plane.copy()
                        for _ in range(4): expected=np.clip(expected+ACTION_VECTORS[action]*2,self.half_size,820-self.half_size)
                        self._plans[key]=dict(path=retained_paths[extension,1:].copy(),expected_plane=expected,remaining=remaining)
                    else: self._plans.pop(key,None)
                    self.counters['plan_reuses']+=1
                    self.counters['overrides']+=int(action!=incumbent)
                    self._trace_decision(key,'retained_route_certified',incumbent,action,metrics=retained_metrics[extension])
                    continue
            build=beam_paths if cfg.compiled_beam else beam_paths.py_func
            best,paths=build(plane,self.half_size,bullets,velocity,error,self.table,
                self.integral,ACTION_VECTORS,cfg.search_depth,cfg.beam_width,cfg.wall_reserve_pixels,cfg.wall_cost_weight)
            # Beam pruning must never remove a directly executable direction.
            paths=np.concatenate((paths,np.repeat(np.arange(9)[:,None],cfg.search_depth,axis=1)))
            lengths=np.full(18,cfg.search_depth,np.int64)
            prior_index=18 if retained_paths is not None else -1
            if retained_paths is not None:
                paths=np.concatenate((paths,retained_paths),axis=0)
                lengths=np.full(27,cfg.search_depth,np.int64)
            metrics=self._assess(plane,self.half_size,bullets,velocity,error,paths,lengths)
            roots=paths[:,0]
            eligible=~sp[row,roots]
            if not eligible.any():
                self.counters['interval_conflicts']+=1
                eligible=np.ones(len(paths),bool)
            if cfg.interval_consensus:
                first_positions=np.clip(plane+ACTION_VECTORS[roots]*8,self.half_size,820-self.half_size)
                first_walls=np.minimum(first_positions-self.half_size,820-self.half_size-first_positions).min(axis=1)
                ranking=sorted(np.flatnonzero(eligible),key=lambda i:(
                    -metrics[i,0],-metrics[i,5],-metrics[i,2],-metrics[i,1],-metrics[i,3],
                    -min(metrics[i,4],cfg.wall_reserve_pixels),
                    -float(first_walls[i]) if wall_gate else 0.,
                    int(i!=9+incumbent),
                    int(prior_index<0 or i<prior_index),float(risk[row,roots[i]]),int(roots[i])))
                chosen=int(ranking[0])
            else:
                chosen=self._choose_root(best.copy(),risk[row],incumbent,sp[row],sn[row])
                if prior_index>=0 and eligible[prior_index] and metrics[prior_index,3]==1:
                    chosen=prior_index
            action=int(roots[chosen])
            actions[row]=action
            path=paths[chosen,:lengths[chosen]].copy()
            reused=prior_index>=0 and chosen>=prior_index
            remaining=prior.get('remaining',len(prior['path']))-1 if reused else len(path)-1
            if cfg.preserve_plans and metrics[chosen,3]==1 and len(path)>1 and remaining>0:
                expected=plane.copy()
                for _ in range(4): expected=np.clip(expected+ACTION_VECTORS[action]*2,self.half_size,820-self.half_size)
                self._plans[key]=dict(path=path[1:],expected_plane=expected,remaining=remaining)
            else: self._plans.pop(key,None)
            self.counters['plan_reuses']+=int(reused)
            self.counters['searches']+=1
            self.counters['geometric_searches']+=int(geometry_gate)
            self.counters['wall_searches']+=int(wall_gate)
            self.counters['overrides']+=int(action!=incumbent)
            self._trace_decision(key,'candidate_ranking',incumbent,action,
                chosen=chosen,roots=roots,metrics=metrics,eligible=eligible,
                short_possible=sp[row],short_nominal=sn[row],reused=bool(reused))
        counters=selection.counter_values.clone()
        revised=torch.as_tensor(actions,device=selection.actions.device)
        counters[4]=(revised!=selection.raw_actions).sum()
        self.elapsed_seconds+=time.perf_counter()-started
        return replace(selection,actions=revised,counter_values=counters)
