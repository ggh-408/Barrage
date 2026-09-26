"""Image-only recovery memory and coherent uncertainty arbitration."""
from dataclasses import asdict, dataclass, replace
import os
from .window_pixel_geometry import WindowPixelGeometry
from .action_selector import WindowActionSelection
import time
import numpy as np
import torch
from barrage_rl.runtime_core import ACTION_VECTORS


def warmup_parallel_controller(guard):
    """Compile the installed kernels before any live decision, without policy state."""
    from barrage_rl.window_planner_kernel import beam_paths
    args = (np.array([410, 410], np.float32), guard.half_size,
            np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32),
            np.ones(1, np.float32))
    cfg = guard.recovery_config
    if cfg.compiled_beam:
        beam_paths(*args, guard.table, guard.integral, ACTION_VECTORS,
                   1, 1, float(cfg.wall_reserve_pixels), float(cfg.wall_cost_weight))
    guard._assess(*args, np.zeros((1, 1), np.int64), np.ones(1, np.int64))
    guard._certificate(*args, np.zeros((1, 1), np.int64), np.ones(1, np.int64))
    guard._assess(*args, np.zeros((1, 1), np.int64), np.ones(1, np.int64), ranking_only=True)


@dataclass(frozen=True)
class WindowPlannerConfig:
    search_depth: int = 15
    beam_width: int = 8
    wall_reserve_pixels: float = 48.0
    wall_cost_weight: float = 1.0
    velocity_error_floor: float = 2.0
    compiled_beam: bool = True
    search_workers: int = 9
    preserve_plans: bool = True
    interval_consensus: bool = True


class WindowPixelGuard(WindowPixelGeometry):
    def _trace_decision(self, key, reason, incumbent, action, **details):
        observer=getattr(self,'_decision_observer',None)
        if observer is not None:
            observer(key,dict(reason=reason,incumbent=int(incumbent),action=int(action),**details))

    def __init__(self, config):
        if min(config.search_depth, config.beam_width, config.search_workers) < 1:
            raise ValueError('search depth, beam width, and workers must be positive')
        if min(config.wall_reserve_pixels, config.wall_cost_weight, config.velocity_error_floor) < 0:
            raise ValueError('recovery weights must be non-negative')
        if not config.interval_consensus or not config.preserve_plans:
            raise ValueError('The pinned window requires consensus and persistent plans')
        super().__init__()
        self.recovery_config = config
        from . import window_planner_kernel  # Resolve the bundled Numba fallback first.
        from numba import set_num_threads, config as numba_config
        set_num_threads(min(config.search_workers, os.cpu_count() or 1, numba_config.NUMBA_NUM_THREADS))
        self.counters = dict(decisions=0, searches=0, geometric_searches=0, wall_searches=0,
            plan_reuses=0, plan_invalidations=0, model_releases=0, overrides=0)
        self.elapsed_seconds = 0.0
        self._plans = {}
        self._context_indices = None
        self._context_steps = 1
        warmup_parallel_controller(self)

    def _certificate(self,plane,half_size,bullets,velocity,error,paths,lengths):
        if getattr(self,'_decision_observer',None) is not None:
            return self._assess(plane,half_size,bullets,velocity,error,paths,lengths)
        from barrage_rl.window_planner_kernel import assess_path_certificates
        result=np.zeros((len(paths),7),np.float64)
        result[:,4:6]=assess_path_certificates(plane,half_size,bullets,velocity,error,
            self.table,self.integral,ACTION_VECTORS,paths,lengths)
        return result

    def _assess(self,plane,half_size,bullets,velocity,error,paths,lengths,certificate=None,ranking_only=False):
        from barrage_rl.window_planner_kernel import assess_paths,assess_path_interval_exposure
        if certificate is not None and getattr(self,'_decision_observer',None) is not None:
            return certificate
        full_metrics=not ranking_only or getattr(self,"_decision_observer",None) is not None
        saved = (np.empty((0,2),np.float64) if certificate is None
                 else np.ascontiguousarray(certificate[:,4:6]))
        scenarios=assess_paths(plane,half_size,bullets,velocity,error,self.table,ACTION_VECTORS,paths,lengths,saved,full_metrics)
        combined=assess_path_interval_exposure(plane,half_size,bullets,velocity,error,
            self.table,self.integral,ACTION_VECTORS,paths,lengths,saved,full_metrics)
        # The combined pass yields exposure and the safe prefix together.
        return np.column_stack((scenarios,combined))

    def reset(self, episode_indices=None):
        if episode_indices is None:
            self._plans.clear()
        else:
            for i in np.asarray(episode_indices).reshape(-1): self._plans.pop(int(i),None)

    def manifest(self):
        return {'recovery_config':asdict(self.recovery_config),
                'counters':dict(self.counters), 'guard_seconds':self.elapsed_seconds,
                'committed_physics_steps':4, 'ranking_variant':'commit_safe_ranking',
                'ranking_safety_basis':'nominal_pixel',
                'dynamic_input':'current image-derived features only',
                'variant':'receding_continuation',
                'trigger_statistics_basis':'short_commitment_and_certificate',
                'uncertainty_scenarios':9,
                'uncertainty_semantics':'coherent deterministic offsets; scenario counts are not probabilities',
                'safety_certificate':'conditional sensitivity test for currently observed tracks; excludes future births and association errors',
                'plan_memory':'image-derived suffix, revalidated each decision and reset per episode'}

    def _geometry(self,o,m,g):
        cfg=self.recovery_config
        observed=g[:2]*820
        valid=m & (o[:,9]==0) & (o[:,8]>=.15)
        valid &= np.linalg.norm(o[:,:2]*820,axis=1)<24+480*cfg.search_depth/30
        selected=o[valid]
        error=np.where(selected[:,15]>.5,
            np.maximum(2/np.maximum(np.minimum(selected[:,10],7/30),2/30),cfg.velocity_error_floor),240)
        return (observed-self.centroid_bias,observed+selected[:,:2]*820-self.bullet_bias,
                (selected[:,2:4]+g[None,6:8])*240,error)

    def apply(self, selection, objects, masks, globals_):
        from barrage_rl.window_planner_kernel import beam_paths
        started=time.perf_counter()
        actions=selection.actions.detach().cpu().numpy().copy()
        observer=getattr(self,'_decision_observer',None)
        if observer is not None:
            sn,sp=self.hazards(objects,masks,globals_)
            short_possible=sp[np.arange(len(actions)),actions]
        else:
            _,sp_selected=self.hazards(objects,masks,globals_,actions=actions,include_nominal=False)
            short_possible=sp_selected[:,0]
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
            endpoint=np.clip(plane+ACTION_VECTORS[incumbent]*32,self.half_size,820-self.half_size)
            wall=np.minimum(endpoint-self.half_size,820-self.half_size-endpoint).min()
            geometry_gate=bool(short_possible[row])
            wall_gate=wall<cfg.wall_reserve_pixels
            self.counters['decisions']+=1
            if not short_possible[row] and not wall_gate:
                direct=np.full((1,cfg.search_depth),incumbent,np.int64)
                certificate=self._certificate(plane,self.half_size,bullets,velocity,error,direct,
                                         np.array([cfg.search_depth],np.int64))[0]
                if certificate[5]==1 and certificate[4]>=cfg.wall_reserve_pixels:
                    self._plans.pop(key,None)
                    self.counters['model_releases']+=int(prior is not None)
                    self._trace_decision(key,'model_full_horizon',incumbent,incumbent,metrics=certificate)
                    continue
            retained_paths=None
            retained_metrics=None
            if prior is not None and cfg.preserve_plans:
                suffix=prior['path'][:cfg.search_depth-1]
                retained_paths=np.repeat(np.arange(9,dtype=np.int64)[:,None],cfg.search_depth,axis=1)
                retained_paths[:,:len(suffix)]=suffix[None]
                retained_metrics=self._certificate(plane,self.half_size,bullets,velocity,error,
                    retained_paths,np.full(9,cfg.search_depth,np.int64))
                viable=np.flatnonzero((retained_metrics[:,5]==1)&(retained_metrics[:,4]>=cfg.wall_reserve_pixels))
                # Extend the suffix before reuse. A dwindling safe prefix gives
                # no evidence that there will be an exit at its endpoint.
                if len(viable):
                    extension=int(suffix[-1]) if int(suffix[-1]) in viable else int(viable[0])
                    action=int(suffix[0])
                    actions[row]=action
                    remaining=cfg.search_depth-1
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
            _,paths=build(plane,self.half_size,bullets,velocity,error,self.table,
                self.integral,ACTION_VECTORS,cfg.search_depth,cfg.beam_width,cfg.wall_reserve_pixels,cfg.wall_cost_weight)
            # Beam pruning must never remove a directly executable direction.
            paths=np.concatenate((paths,np.repeat(np.arange(9,dtype=np.int64)[:,None],cfg.search_depth,axis=1)))
            lengths=np.full(18,cfg.search_depth,np.int64)
            prior_index=18 if retained_paths is not None else -1
            if retained_paths is not None:
                paths=np.concatenate((paths,retained_paths),axis=0)
                lengths=np.full(27,cfg.search_depth,np.int64)
            if retained_metrics is None:
                metrics=self._assess(plane,self.half_size,bullets,velocity,error,paths,lengths,ranking_only=True)
            else:
                # Reuse the certificate columns computed for these same paths.
                retained_metrics=self._assess(plane,self.half_size,bullets,velocity,error,
                    retained_paths,lengths[18:],certificate=retained_metrics,ranking_only=True)
                metrics=np.concatenate((self._assess(plane,self.half_size,bullets,velocity,error,
                    paths[:18],lengths[:18],ranking_only=True),retained_metrics))
            roots=paths[:,0]
            first_walls=None
            if wall_gate:
                first_positions=np.clip(plane+ACTION_VECTORS[roots]*8,self.half_size,820-self.half_size)
                first_walls=np.minimum(first_positions-self.half_size,820-self.half_size-first_positions).min(axis=1)
            from tools.commit_safe_ranking import rank_routes
            ranking,eligible=rank_routes(metrics,lengths,roots,first_walls,
                cfg.wall_reserve_pixels,wall_gate,incumbent,prior_index)
            chosen=int(ranking[0])
            action=int(roots[chosen])
            actions[row]=action
            path=paths[chosen,:lengths[chosen]].copy()
            reused=prior_index>=0 and chosen>=prior_index
            remaining=len(path)-1
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
            if observer is not None:
                self._trace_decision(key,'candidate_ranking',incumbent,action,
                    chosen=chosen,roots=roots,metrics=metrics,eligible=eligible,
                    short_possible=sp[row],short_nominal=sn[row],reused=bool(reused))
        revised=torch.as_tensor(actions,device=selection.actions.device)
        self.elapsed_seconds+=time.perf_counter()-started
        if isinstance(selection, WindowActionSelection):
            return replace(selection, actions=revised)
        counters=selection.counter_values.clone()
        counters[4]=(revised!=selection.raw_actions).sum()
        return replace(selection,actions=revised,counter_values=counters)

