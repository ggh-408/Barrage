"""Image-derived short commitment protection and receding-horizon recovery."""
from dataclasses import asdict, dataclass, replace
import os
import time
import numpy as np
import torch

from barrage_rl.runtime_core import ACTION_VECTORS
from tools.pixel_guard_candidate import PixelGuardConfig
from tools.pixel_guard_refined import RefinedPixelGuard
from tools.pixel_search_kernel import search_step


@dataclass(frozen=True)
class RecedingGuardConfig:
    search_depth: int = 18
    beam_width: int = 8
    trigger_safe_actions: int = 2
    wall_reserve_pixels: float = 48.0
    wall_cost_weight: float = 1.0
    velocity_error_floor: float = 2.0
    compiled_beam: bool = True
    search_workers: int = 9
    preserve_plans: bool = True
    interval_consensus: bool = True


class _StatelessPixelGuard(RefinedPixelGuard):
    def __init__(self, config=RecedingGuardConfig()):
        if config.search_depth < 1 or config.beam_width < 1 or config.search_workers < 1:
            raise ValueError('search depth, beam width, and workers must be positive')
        if not 0 <= config.trigger_safe_actions <= 9:
            raise ValueError('trigger_safe_actions must be in [0, 9]')
        if min(config.wall_reserve_pixels, config.wall_cost_weight, config.velocity_error_floor) < 0:
            raise ValueError('recovery weights must be non-negative')
        super().__init__(PixelGuardConfig(physics_steps=16, compiled_search=True))
        self.recovery_config = config
        self.commit_guard = RefinedPixelGuard(PixelGuardConfig(physics_steps=4))
        self.counters.update(searches=0, geometric_searches=0, wall_searches=0)
        if config.compiled_beam:
            from tools.pixel_receding_kernel import beam_costs, set_num_threads
            from numba import config as numba_config
            set_num_threads(min(config.search_workers,os.cpu_count() or 1,numba_config.NUMBA_NUM_THREADS))
            beam_costs(np.array([410,410],np.float32),self.half_size,
                np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),np.ones(1,np.float32),
                self.table,self.integral,ACTION_VECTORS,1,1,48.,1.)

    def search(self, objects, masks, globals_, risk, incumbent, committed_possible, committed_nominal):
        cfg = self.recovery_config
        observed = globals_[:2]*820
        plane = observed-self.centroid_bias
        valid = masks & (objects[:,9] == 0) & (objects[:,8] >= .15)
        valid &= np.linalg.norm(objects[:,:2]*820, axis=1) < 24+480*cfg.search_depth/30
        o = objects[valid]
        bullets = observed+o[:,:2]*820-self.bullet_bias
        velocity = (o[:,2:4]+globals_[6:8])*240
        error = np.where(o[:,15]>.5, np.maximum(2/np.maximum(o[:,10],2/30),cfg.velocity_error_floor),240)
        if cfg.compiled_beam:
            from tools.pixel_receding_kernel import beam_costs
            best=beam_costs(plane,self.half_size,bullets,velocity,error,self.table,
                self.integral,ACTION_VECTORS,cfg.search_depth,cfg.beam_width,
                cfg.wall_reserve_pixels,cfg.wall_cost_weight)
            return self._choose_root(best,risk,incumbent,committed_possible,committed_nominal)
        positions = plane[None].copy()
        roots = np.array([-1])
        costs = np.zeros(1)
        alive = np.ones(1, bool)
        wall_cost = np.zeros(1)
        for level in range(cfg.search_depth):
            positions = np.repeat(positions,9,axis=0)
            actions = np.tile(np.arange(9),len(roots))
            roots = actions.copy() if level == 0 else np.repeat(roots,9)
            costs = np.repeat(costs,9)
            alive = np.repeat(alive,9)
            for substep in range(4):
                t = (level*4+substep+1)/120
                b = bullets+velocity*t
                positions,hit,mass = search_step(positions,ACTION_VECTORS[actions]*2,
                    self.half_size,b,.5+error*t,self.table,self.integral)
                alive &= ~hit
                costs += (~alive)*10000+mass.sum(axis=1)
            wall = np.minimum(positions-self.half_size,820-self.half_size-positions).min(axis=1)
            wall_cost = cfg.wall_cost_weight*np.square(np.maximum(cfg.wall_reserve_pixels-wall,0)/max(cfg.wall_reserve_pixels,1))
            if len(bullets):
                distance = np.linalg.norm(positions[:,None]-b[None],axis=-1).min(axis=1)
                proximity = .02/np.maximum(distance,1)
            else:
                proximity = np.zeros(len(positions))
            ranking = costs+wall_cost+proximity
            keep = []
            for root in range(9):
                indices = np.flatnonzero(roots == root)
                indices = indices[np.argsort(ranking[indices],kind='stable')]
                _,first = np.unique(np.round(positions[indices],1),axis=0,return_index=True)
                keep.extend(indices[np.sort(first)[:cfg.beam_width]])
            keep = np.asarray(keep)
            positions,roots,costs,alive,wall_cost = positions[keep],roots[keep],costs[keep],alive[keep],wall_cost[keep]
        best = np.array([np.min((costs+wall_cost)[roots == root]) for root in range(9)])
        return self._choose_root(best,risk,incumbent,committed_possible,committed_nominal)

    @staticmethod
    def _choose_root(best,risk,incumbent,committed_possible,committed_nominal):
        # A longer fixed-action collision must not veto a valid 33 ms prefix.
        eligible = ~committed_possible
        if not eligible.any():
            eligible = ~committed_nominal
        if eligible.any():
            best[~eligible] = np.inf
        return int(np.argmin(best+risk*1e-3+(np.arange(9)!=incumbent)*1e-6))

    def apply(self, selection, objects, masks, globals_):
        started = time.perf_counter()
        nominal, possible = self.hazards(objects,masks,globals_)
        short_nominal, short_possible = self.commit_guard.hazards(objects,masks,globals_)
        actions = selection.actions.detach().cpu().numpy().copy()
        risk = selection.immediate_risk.detach().cpu().numpy()
        all_unsafe = selection.all_unsafe.detach().cpu().numpy()
        cfg = self.recovery_config
        for row, incumbent in enumerate(actions):
            o = objects[row]
            young = masks[row] & (o[:,15]<.5) & (o[:,9]==0) & (o[:,8]>=.15)
            young &= np.linalg.norm(o[:,:2]*820,axis=1)<40
            plane = globals_[row,:2]*820-self.centroid_bias
            endpoint = np.clip(plane+ACTION_VECTORS[incumbent]*32,self.half_size,820-self.half_size)
            wall = np.minimum(endpoint-self.half_size,820-self.half_size-endpoint).min()
            geometry_gate = possible[row,incumbent] or np.count_nonzero(~possible[row]) <= cfg.trigger_safe_actions
            wall_gate = wall < cfg.wall_reserve_pixels
            self.counters['decisions'] += 1
            if all_unsafe[row] or geometry_gate or wall_gate or young.any():
                actions[row] = self.search(o,masks[row],globals_[row],risk[row],incumbent,short_possible[row],short_nominal[row])
                self.counters['searches'] += 1
                self.counters['geometric_searches'] += int(geometry_gate)
                self.counters['wall_searches'] += int(wall_gate)
                self.counters['overrides'] += int(actions[row] != incumbent)
        revised = torch.as_tensor(actions,device=selection.actions.device)
        counters = selection.counter_values.clone()
        counters[4] = (revised != selection.raw_actions).sum()
        self.elapsed_seconds += time.perf_counter()-started
        return replace(selection,actions=revised,counter_values=counters)

    def manifest(self):
        return {**super().manifest(), 'variant':'receding',
                'recovery_config':asdict(self.recovery_config),
                'committed_physics_steps':4}


from tools.pixel_guard_continuation import ContinuationMixin


class RecedingPixelGuard(ContinuationMixin, _StatelessPixelGuard):
    def __init__(self, config=RecedingGuardConfig()):
        super().__init__(config)


def install_receding_guard(agent, config=RecedingGuardConfig()):
    existing = getattr(agent, '_receding_pixel_guard', None)
    if existing is not None:
        if existing.recovery_config != config:
            raise ValueError('A different receding guard is already installed')
        return existing
    guard = RecedingPixelGuard(config)
    guard.safety_threshold = agent.safety_threshold
    original = agent._select_actions
    def select(objects,masks,globals_,*,deterministic):
        model=getattr(agent,'model',None)
        previous=getattr(model,'_image_continuation_context',None)
        if getattr(model,'continuation_head',None) is not None:
            from barrage_rl.tracked_policy import continuation_plan_context
            ids=np.arange(len(objects)) if guard._context_indices is None else guard._context_indices
            context=continuation_plan_context([guard._plans.get(int(i)) for i in ids])
            model._image_continuation_context=torch.as_tensor(context,device=objects.device,dtype=objects.dtype)
        try:
            selection = original(objects,masks,globals_,deterministic=deterministic)
        finally:
            if getattr(model,'continuation_head',None) is not None:
                model._image_continuation_context=previous
        return guard.apply(selection,objects.detach().cpu().numpy(),masks.detach().cpu().numpy(),globals_.detach().cpu().numpy())
    agent._select_actions = select
    agent._receding_pixel_guard = guard
    from functools import wraps
    def with_context(original, episode_argument):
        @wraps(original)
        def act(*args, **kwargs):
            previous=guard._context_indices,guard._context_steps
            guard._context_indices=kwargs.get('episode_indices',args[episode_argument] if len(args)>episode_argument else None)
            guard._context_steps=kwargs.get('decision_steps',1)
            try:
                return original(*args,**kwargs)
            finally:
                guard._context_indices,guard._context_steps=previous
        return act
    for name,episode_argument in (('act_features',4),('act_features_with_diagnostics',3)):
        if hasattr(agent,name): setattr(agent,name,with_context(getattr(agent,name),episode_argument))
    if hasattr(agent,'reset_state'):
        original_reset=agent.reset_state
        def reset_state(episode_indices=None):
            guard.reset(episode_indices)
            return original_reset(episode_indices)
        agent.reset_state=reset_state
    return guard
