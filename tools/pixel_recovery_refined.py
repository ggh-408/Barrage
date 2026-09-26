"""Bounded image-only receding-horizon search over exact sprite footprints."""
import numpy as np
from barrage_rl.runtime_core import ACTION_VECTORS


def recovery_action(guard, objects, masks, globals_, risk, depth=9, beam_width=8, compiled=False):
    observed=globals_[:2]*820
    plane=observed-guard.centroid_bias
    valid=masks & (objects[:,9]==0) & (objects[:,8]>=.15)
    valid &= np.linalg.norm(objects[:,:2]*820,axis=1)<24+480*depth/30
    if not valid.any(): return int(np.argmin(risk))
    o=objects[valid]
    bullets=observed+o[:,:2]*820-guard.bullet_bias
    velocity=(o[:,2:4]+globals_[6:8])*240
    velocity_error=np.where(o[:,15]>.5,2/np.maximum(o[:,10],2/30),240)
    positions=plane[None].copy()
    roots=np.array([-1]); costs=np.zeros(1); alive=np.ones(1,bool)
    for level in range(depth):
        positions=np.repeat(positions,9,axis=0)
        actions=np.tile(np.arange(9),len(roots))
        roots=actions.copy() if level==0 else np.repeat(roots,9)
        costs=np.repeat(costs,9); alive=np.repeat(alive,9)
        for substep in range(4):
            t=(level*4+substep+1)/120
            if compiled:
                from tools.pixel_search_kernel import search_step
                b=bullets+velocity*t
                positions,hit,mass=search_step(positions,ACTION_VECTORS[actions]*2,
                    guard.half_size,b,.5+velocity_error*t,guard.table,guard.integral)
                alive &= ~hit
                costs += (~alive)*10000 + mass.sum(axis=1)
                continue
            positions=np.clip(positions+ACTION_VECTORS[actions]*2,
                              guard.half_size,820-guard.half_size)
            b=bullets+velocity*t
            offsets=guard.rounded(b)[None]-guard.rounded(positions)[:,None]
            hit=(guard.rectangle_hits(offsets,offsets)>0).any(axis=1)
            alive &= ~hit
            pe=.5; be=.5+velocity_error[:,None]*t
            lo=guard.rounded(b-be)[None]-guard.rounded(positions+pe-1e-5)[:,None]
            hi=guard.rounded(b+be-1e-5)[None]-guard.rounded(positions-pe)[:,None]
            # Fraction of overlapping integer offsets is a ranking surrogate,
            # not a calibrated collision probability or a universal veto.
            mass=guard.rectangle_hits(lo,hi)/np.prod(hi-lo+1,axis=-1)
            costs += (~alive)*10000 + mass.sum(axis=1)
        distance=np.linalg.norm(positions[:,None]-b[None],axis=-1).min(axis=1)
        ranking=costs+0.02/np.maximum(distance,1)
        keep=[]
        for root in range(9):
            indices=np.flatnonzero(roots==root)
            indices=indices[np.argsort(ranking[indices],kind='stable')]
            # Equivalent locations need one representative; preserve routes.
            _,first=np.unique(np.round(positions[indices],1),axis=0,return_index=True)
            keep.extend(indices[np.sort(first)[:beam_width]])
        keep=np.asarray(keep)
        positions=positions[keep]; roots=roots[keep]
        costs=costs[keep]; alive=alive[keep]
    best=np.full(9,np.inf)
    for root in range(9):
        best[root]=np.min(costs[roots==root])
    return int(np.argmin(best+np.asarray(risk)*1e-3))
