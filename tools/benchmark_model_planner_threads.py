"""Screen model and planner worker counts against exact captured image features."""
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.benchmark_model_norm import model_for
from tools.benchmark_tracker_exact import exact


def main():
    torch.set_num_threads(10)
    source=ROOT/'diagnostics/model_blocks_20260925_122341_223983/model_samples.pkl'
    with source.open('rb') as f:samples=pickle.load(f)
    arrays=[s['inputs'] for s in samples]
    inputs=[tuple(torch.from_numpy(a) for a in row) for row in arrays]
    model=model_for(ROOT/'best.pt')
    def outputs(row):
        r=model.forward_with_geometry(*row)
        return tuple(t.numpy().copy() for t in (*r[:3],r[3].clearance_by_object,r[3].normalized_minimum_clearance))
    results={}
    with torch.inference_mode():
        references=[outputs(row) for row in inputs]
        for threads in (10,1,2,4):
            torch.set_num_threads(threads)
            matched=True
            for row,ref in zip(inputs,references):
                try:exact(outputs(row),ref)
                except AssertionError:matched=False
            times=[]
            for _ in range(4):
                for row in inputs:
                    start=time.perf_counter();model.forward_with_geometry(*row);times.append((time.perf_counter()-start)*1000)
            results[f'model_threads_{threads}']=dict(exact=matched,mean_ms=float(np.mean(times)))
    print(json.dumps(results),flush=True)
    torch.set_num_threads(10)
    from tools.train_targeted_dagger import install_runtime,CONTROLLER
    install_runtime()
    from barrage_rl.deployment import configure_image_controller
    from barrage_rl.live_screen import LiveVisualController
    import numba
    controller=LiveVisualController(str(ROOT/'best.pt'),experimental_controller=CONTROLLER)
    configure_image_controller(controller.agent,'receding',search_workers=9)
    guard=controller.agent._receding_pixel_guard
    original=guard.apply;times=[]
    def measured(*args,**kw):
        start=time.perf_counter()
        try:return original(*args,**kw)
        finally:times.append((time.perf_counter()-start)*1000)
    guard.apply=measured
    expected=None
    for threads in (9,1,2,4):
        numba.set_num_threads(threads)
        controller.agent.reset_state();times.clear();actions=[]
        for row in arrays:
            actions.append(controller.agent.act_features(*row,deterministic=True).copy())
        if expected is None:expected=actions
        exact(actions,expected)
        results[f'planner_threads_{threads}']=dict(actions_exact=True,mean_ms=float(np.mean(times)))
    (ROOT/'diagnostics/five_stage_20260925/thread_screening.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
