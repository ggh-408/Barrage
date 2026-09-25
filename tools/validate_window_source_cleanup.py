"""Paired pre/post source cleanup window trajectory; image-only controller inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--decisions',type=int,default=3600)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    import pygame,torch
    torch.set_num_threads(10)
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.window_runtime import CONTROLLER,configure_window_controller
    from barrage_rl.runtime_core import ACTION_VECTORS
    from tools.validate_window_cleanup import load_module
    from tools.benchmark_tracker_exact import exact,state
    from barrage_rl import foreground_kernel
    from numba import get_num_threads
    pygame.display.init();pygame.display.set_mode((1,1))
    before=ROOT/'diagnostics/window_source_cleanup_20260925/before'
    old_live=load_module('barrage_rl._source_cleanup_before_live',before/'barrage_rl/live_screen.py')
    modules=[];controllers=[];observations=[{},{}]
    for j,workers in enumerate((args.workers,args.workers)):
        module=load_module(f'cleanup_check_game_{j}',(before if j==0 else ROOT)/'Barrage.py');module.PROJECT_ROOT=ROOT;modules.append(module)
        g=module.Barrage
        g.SCREEN_WIDTH=g.SCREEN_HEIGHT=820;g.QUANTITY=300;g.BULLET_SIZE=5
        g.PLANE_SPEED=g.BULLET_SPEED=240.;g.TARGETED_BULLET_PROBABILITY=.10
        g.COLLISION=False;g.INVINCIBLE=False;g.KEY=True;g.MUSIC=False;g.AI_PIPELINE=None
        g.RNG=np.random.default_rng(20260925);g.window=pygame.Surface((820,820));module.Plane.SKIN=0
        controller_class=old_live.LiveVisualController if j==0 else LiveVisualController
        c=controller_class(str(ROOT/'best.pt'),experimental_controller=CONTROLLER,rgb_workers=workers,record_timings=True)
        configure_window_controller(c.agent,search_workers=9)
        detect=c.semanticizer.detect
        def record_detect(rgb,_fn=detect,_j=j,**kw):
            result=_fn(rgb,**kw)
            observations[_j]['rgb']=rgb.copy()
            observations[_j]['detections']=vars(result)
            return result
        c.semanticizer.detect=record_detect
        forward=c.agent.model.forward_policy
        def record_logits(*a,_fn=forward,_j=j,**kw):
            result=_fn(*a,**kw);observations[_j]['logits']=result.detach().numpy().copy();return result
        c.agent.model.forward_policy=record_logits
        g.AI_CONTROLLER=c;controllers.append(c);g.reset_game()
    timings=[[],[]];actions=[];started=time.perf_counter();cpu=time.process_time()
    for step in range(args.decisions+1):
        if step:
            for j in ((0,1) if step%2 else (1,0)):
                g=modules[j].Barrage
                for _ in range(4):g.advance_physics(ACTION_VECTORS[g.AI_ACTION],1/120)
                g.render_world();before_threads=get_num_threads();t=time.perf_counter()
                g.AI_ACTION=controllers[j].observe_due_surface(g.window)
                timings[j].append((time.perf_counter()-t)*1000)
                assert get_num_threads()==before_threads
        exact(observations[0],observations[1])
        exact(state(controllers[0].tracked_extractor),state(controllers[1].tracked_extractor))
        exact(controllers[0].tracked_extractor._features(),controllers[1].tracked_extractor._features())
        ga,gb=[c.agent._receding_pixel_guard for c in controllers]
        exact(ga._plans,gb._plans);exact(ga.counters,gb.counters)
        a,b=[m.Barrage for m in modules]
        assert a.AI_ACTION==b.AI_ACTION
        exact(a.RNG.bit_generator.state,b.RNG.bit_generator.state)
        exact(np.asarray(a.PLANE.position),np.asarray(b.PLANE.position))
        exact(modules[0].Bullet.LIST.state,modules[1].Bullet.LIST.state)
        exact(modules[0].Bullet.LIST.targeted,modules[1].Bullet.LIST.targeted)
        actions.append([int(a.AI_ACTION),int(b.AI_ACTION)])
        if step and step%600==0:print(f'{step}/{args.decisions}: pixels, detections, state, features, logits, plans and actions exact',flush=True)
    wall=time.perf_counter()-started;endcpu=time.process_time()
    result=dict(decisions=args.decisions,initial_decisions=1,physics_steps=args.decisions*4,seed=20260925,
        workers=args.workers,baseline_source=str(before),all_intermediate_data_bitwise_equal=True,all_actions_equal=True,
        planner_thread_mask_restored=True,wall_seconds=wall,
        process_cpu_seconds=endcpu-cpu,
        scope='Paired complete 120-second simulated trajectory with damage disabled, offscreen correctness test; no survival-rate estimate.',
        tracking_mean_ms=[float(np.mean(c._stage_tracking_ms[1:])) for c in controllers],
        timings=[dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95))) for v in timings])
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    args.output.with_suffix('.actions.json').write_text(json.dumps(actions),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True);pygame.quit()

if __name__=='__main__':main()
