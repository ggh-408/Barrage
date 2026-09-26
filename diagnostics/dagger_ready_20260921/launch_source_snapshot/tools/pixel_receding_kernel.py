"""Compiled beam bookkeeping with the shared pixel-overlap kernel."""
import numpy as np
from tools.pixel_search_kernel import njit, search_step
from numba import prange, set_num_threads, get_num_threads


@njit(cache=False)
def beam_costs_serial(plane,half_size,bullets,velocity,error,table,integral,vectors,
               depth,beam_width,wall_reserve,wall_weight,forced_root=-1):
    positions=plane.reshape(1,2).copy()
    roots=np.array([-1],np.int64)
    costs=np.zeros(1,np.float64)
    alive=np.ones(1,np.bool_)
    wall_cost=np.zeros(1,np.float32)
    for level in range(depth):
        n=1 if level==0 and forced_root>=0 else len(roots)*9
        next_positions=np.empty((n,2),np.float32)
        movement=np.empty((n,2),np.float32)
        next_roots=np.empty(n,np.int64)
        next_costs=np.empty(n,np.float64)
        next_alive=np.empty(n,np.bool_)
        for i in range(n):
            parent=i//9
            action=np.int64(forced_root) if level==0 and forced_root>=0 else np.int64(i%9)
            next_positions[i]=positions[parent]
            movement[i]=vectors[action]*np.float32(2)
            next_roots[i]=action if level==0 else roots[parent]
            next_costs[i]=costs[parent]
            next_alive[i]=alive[parent]
        for substep in range(4):
            t=np.float32((level*4+substep+1)/120)
            b=bullets+velocity*t
            next_positions,hit,mass=search_step(next_positions,movement,half_size,
                b,np.float32(.5)+error*t,table,integral,True)
            for i in range(n):
                next_alive[i]=next_alive[i] and not hit[i]
                next_costs[i]+=(0 if next_alive[i] else 10000)+np.sum(mass[i])
        ranking=next_costs.copy()
        next_wall_cost=np.zeros(n,np.float32)
        for i in range(n):
            p=next_positions[i]
            wall=min(p[0]-half_size[0],p[1]-half_size[1],
                     np.float32(820)-half_size[0]-p[0],np.float32(820)-half_size[1]-p[1])
            fraction=np.float32(max(np.float32(wall_reserve)-wall,np.float32(0))/np.float32(max(wall_reserve,1)))
            next_wall_cost[i]=np.float32(wall_weight)*np.float32(fraction*fraction)
            distance=np.float32(np.inf)
            for j in range(len(bullets)):
                dx=p[0]-b[j,0]
                dy=p[1]-b[j,1]
                distance=min(distance,np.sqrt(np.float32(dx*dx+dy*dy)))
            proximity=np.float32(.02)/max(distance,np.float32(1)) if len(bullets) else np.float32(0)
            ranking[i]+=next_wall_cost[i]
            ranking[i]+=np.float64(proximity)
        rounded_positions=np.round(next_positions,1)
        keep=np.empty(9*beam_width,np.int64)
        kept=0
        for root in range(9):
            indices=np.flatnonzero(next_roots==root)
            ordered=indices[np.argsort(ranking[indices],kind='mergesort')]
            root_start=kept
            for i in ordered:
                duplicate=False
                for k in range(root_start,kept):
                    other=keep[k]
                    if rounded_positions[i,0]==rounded_positions[other,0] and rounded_positions[i,1]==rounded_positions[other,1]:
                        duplicate=True
                        break
                if not duplicate:
                    keep[kept]=i
                    kept+=1
                    if kept-root_start>=beam_width:
                        break
        selected=keep[:kept]
        positions=next_positions[selected]
        roots=next_roots[selected]
        costs=next_costs[selected]
        alive=next_alive[selected]
        wall_cost=next_wall_cost[selected]
    best=np.full(9,np.inf)
    for i in range(len(roots)):
        best[roots[i]]=min(best[roots[i]],costs[i]+wall_cost[i])
    return best


@njit(cache=False, parallel=True)
def beam_costs(plane,half_size,bullets,velocity,error,table,integral,vectors,
               depth,beam_width,wall_reserve,wall_weight):
    # Each root owns its beam. No branch competes with another root for capacity.
    best=np.empty(9,np.float64)
    for root in prange(9):
        costs=beam_costs_serial(plane,half_size,bullets,velocity,error,table,integral,
            vectors,depth,beam_width,wall_reserve,wall_weight,np.int64(root))
        best[root]=costs[root]
    return best

@njit(cache=False)
def beam_paths_serial(plane,half_size,bullets,velocity,error,table,integral,vectors,
               depth,beam_width,wall_reserve,wall_weight,forced_root=-1):
    positions=plane.reshape(1,2).copy()
    roots=np.array([-1],np.int64)
    costs=np.zeros(1,np.float64)
    alive=np.ones(1,np.bool_)
    wall_cost=np.zeros(1,np.float32)
    paths=np.full((1,depth),-1,np.int64)
    for level in range(depth):
        n=1 if level==0 and forced_root>=0 else len(roots)*9
        next_positions=np.empty((n,2),np.float32)
        movement=np.empty((n,2),np.float32)
        next_roots=np.empty(n,np.int64)
        next_costs=np.empty(n,np.float64)
        next_alive=np.empty(n,np.bool_)
        next_paths=np.full((n,depth),-1,np.int64)
        for i in range(n):
            parent=i//9
            action=np.int64(forced_root) if level==0 and forced_root>=0 else np.int64(i%9)
            next_positions[i]=positions[parent]
            movement[i]=vectors[action]*np.float32(2)
            next_roots[i]=action if level==0 else roots[parent]
            next_costs[i]=costs[parent]
            next_alive[i]=alive[parent]
            next_paths[i,:level]=paths[parent,:level]
            next_paths[i,level]=action
        for substep in range(4):
            t=np.float32((level*4+substep+1)/120)
            b=bullets+velocity*t
            next_positions,hit,mass=search_step(next_positions,movement,half_size,
                b,np.float32(.5)+error*t,table,integral,True)
            for i in range(n):
                next_alive[i]=next_alive[i] and not hit[i]
                next_costs[i]+=(0 if next_alive[i] else 10000)+np.sum(mass[i])
        ranking=next_costs.copy()
        next_wall_cost=np.zeros(n,np.float32)
        for i in range(n):
            p=next_positions[i]
            wall=min(p[0]-half_size[0],p[1]-half_size[1],
                     np.float32(820)-half_size[0]-p[0],np.float32(820)-half_size[1]-p[1])
            fraction=np.float32(max(np.float32(wall_reserve)-wall,np.float32(0))/np.float32(max(wall_reserve,1)))
            next_wall_cost[i]=np.float32(wall_weight)*np.float32(fraction*fraction)
            distance=np.float32(np.inf)
            for j in range(len(bullets)):
                dx=p[0]-b[j,0]
                dy=p[1]-b[j,1]
                distance=min(distance,np.sqrt(np.float32(dx*dx+dy*dy)))
            proximity=np.float32(.02)/max(distance,np.float32(1)) if len(bullets) else np.float32(0)
            ranking[i]+=next_wall_cost[i]
            ranking[i]+=np.float64(proximity)
        rounded_positions=np.round(next_positions,1)
        keep=np.empty(9*beam_width,np.int64)
        kept=0
        for root in range(9):
            indices=np.flatnonzero(next_roots==root)
            ordered=indices[np.argsort(ranking[indices],kind='mergesort')]
            root_start=kept
            for i in ordered:
                duplicate=False
                for k in range(root_start,kept):
                    other=keep[k]
                    if rounded_positions[i,0]==rounded_positions[other,0] and rounded_positions[i,1]==rounded_positions[other,1]:
                        duplicate=True
                        break
                if not duplicate:
                    keep[kept]=i
                    kept+=1
                    if kept-root_start>=beam_width:
                        break
        selected=keep[:kept]
        positions=next_positions[selected]
        roots=next_roots[selected]
        costs=next_costs[selected]
        alive=next_alive[selected]
        wall_cost=next_wall_cost[selected]
        paths=next_paths[selected]
    best=np.full(9,np.inf)
    best_paths=np.full((9,depth),-1,np.int64)
    for i in range(len(roots)):
        if costs[i]+wall_cost[i]<best[roots[i]]:
            best_paths[roots[i]]=paths[i]
        best[roots[i]]=min(best[roots[i]],costs[i]+wall_cost[i])
    return best,best_paths



@njit(cache=False, parallel=True)
def beam_paths(plane,half_size,bullets,velocity,error,table,integral,vectors,
               depth,beam_width,wall_reserve,wall_weight):
    best=np.empty(9,np.float64)
    paths=np.empty((9,depth),np.int64)
    for root in prange(9):
        costs,plans=beam_paths_serial(plane,half_size,bullets,velocity,error,table,integral,
            vectors,depth,beam_width,wall_reserve,wall_weight,np.int64(root))
        best[root]=costs[root]
        paths[root]=plans[root]
    return best,paths

@njit(cache=False)
def assess_paths(plane,half_size,bullets,velocity,error,table,vectors,paths,lengths):
    # Deterministic coherent perturbations, not probabilities. Each scenario
    # carries the same initial/velocity offset throughout its entire path.
    metrics=np.zeros((len(paths),5),np.float64)
    for i in range(len(paths)):
        terminal=plane.copy()
        for step in range(lengths[i]*4):
            a=paths[i,step//4]
            for axis in range(2):
                terminal[axis]=min(max(np.float32(terminal[axis]+vectors[a,axis]*np.float32(2)),
                                      half_size[axis]),np.float32(820)-half_size[axis])
        metrics[i,4]=min(terminal[0]-half_size[0],terminal[1]-half_size[1],
                         np.float32(820)-half_size[0]-terminal[0],
                         np.float32(820)-half_size[1]-terminal[1])
        for scenario in range(9):
            dx=np.float32(scenario%3-1)
            dy=np.float32(scenario//3-1)
            p=plane.copy()
            survived=0
            failed=False
            for step in range(lengths[i]*4):
                a=paths[i,step//4]
                for axis in range(2):
                    p[axis]=min(max(np.float32(p[axis]+vectors[a,axis]*np.float32(2)),
                                   half_size[axis]),np.float32(820)-half_size[axis])
                t=np.float32((step+1)/120)
                px=int(np.floor(np.float32(p[0]+np.float32(.5))))
                py=int(np.floor(np.float32(p[1]+np.float32(.5))))
                for j in range(len(bullets)):
                    e=np.float32(1)+error[j]*t
                    bx=bullets[j,0]+velocity[j,0]*t+dx*e
                    by=bullets[j,1]+velocity[j,1]*t+dy*e
                    if abs(bx-p[0])>34 or abs(by-p[1])>34: continue
                    cx=int(np.floor(np.float32(bx+np.float32(.5))))-px
                    cy=int(np.floor(np.float32(by+np.float32(.5))))-py
                    if -32<=cx<=32 and -32<=cy<=32 and table[cy+32,cx+32]>0:
                        failed=True
                        break
                if failed: break
                survived+=1
            metrics[i,0]+=int(survived>=4)
            metrics[i,1]+=survived/max(lengths[i]*4,1)
            metrics[i,2]+=int(not failed)
            if scenario==4: metrics[i,3]=survived/max(lengths[i]*4,1)
    return metrics


@njit(cache=False)
def assess_path_intervals(plane,half_size,bullets,velocity,error,table,integral,vectors,paths,lengths):
    """Guaranteed safe prefix over full per-bullet pixel-offset rectangles."""
    result=np.zeros(len(paths),np.float64)
    for i in range(len(paths)):
        p=plane.reshape(1,2).copy()
        survived=0
        for step in range(lengths[i]*4):
            t=np.float32((step+1)/120)
            movement=(vectors[paths[i,step//4]]*np.float32(2)).reshape(1,2)
            p,hit,mass=search_step(p,movement,half_size,bullets+velocity*t,
                np.float32(1)+error*t,table,integral,True)
            if np.any(mass>0):break
            survived+=1
        result[i]=survived/max(lengths[i]*4,1)
    return result
