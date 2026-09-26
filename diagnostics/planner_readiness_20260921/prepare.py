"""Freeze one isolated planner candidate and a bounded verification plan."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
old=ROOT/'diagnostics/planner_consistency_20260921/candidate_controller.py'
source=old.read_text(encoding='utf-8')
assert 'from tools.pixel_receding_kernel import beam_paths,assess_paths' in source
source=source.replace('from tools.pixel_receding_kernel import beam_paths,assess_paths',
    'from readiness_kernel import beam_paths,assess_paths')
source=source.replace('from tools.pixel_receding_kernel import assess_paths,assess_path_intervals',
    'from readiness_kernel import assess_paths,assess_path_intervals,assess_path_exposure')
source=source.replace('return np.column_stack((scenarios,intervals))',
    'exposure=assess_path_exposure(plane,half_size,bullets,velocity,error,self.table,\n            self.integral,ACTION_VECTORS,paths,lengths)\n        return np.column_stack((scenarios,intervals,exposure))')
# The estimator refits a rolling eight-frame window; old tracks do not get
# additional independent observations in that estimate. This remains a
# sensitivity envelope, not a statistical confidence interval.
source=source.replace('2/np.maximum(selected[:,10],2/30)',
    '2/np.maximum(np.minimum(selected[:,10],7/30),2/30)')
# A renewed full-horizon suffix is valid independent of the proposed action's
# wall test. Its own terminal wall reserve is already checked above.
source=source.replace('if len(viable) and not wall_gate:', 'if len(viable):')
source=source.replace("remaining=prior.get('remaining',len(prior['path']))-1", 'remaining=cfg.search_depth-1')
source=source.replace("remaining=cfg.search_depth-1 if reused else len(path)-1", 'remaining=len(path)-1')
oldrank='-metrics[i,5],-metrics[i,2],-metrics[i,1],-metrics[i,3],'
assert oldrank in source
source=source.replace(oldrank,'metrics[i,6],-metrics[i,5],-metrics[i,2],-metrics[i,1],-metrics[i,3],')
source=source.replace("'safety_certificate':'full independent per-bullet pixel-offset rectangles over the complete horizon'",
    "'safety_certificate':'conditional sensitivity test for currently observed tracks; excludes future births and association errors'")
with (HERE/'candidate_controller.py').open('x',encoding='utf-8') as f:f.write(source)
kernel=(ROOT/'tools/pixel_receding_kernel.py').read_text(encoding='utf-8')
# Search and final assessment must use the same initial error width.
kernel=kernel.replace('np.float32(.5)+error*t','np.float32(1)+error*t')
kernel+='''

@njit(cache=False)
def assess_path_exposure(plane,half_size,bullets,velocity,error,table,integral,vectors,paths,lengths):
    """Integrated rectangle overlap sensitivity, never a collision probability."""
    result=np.zeros(len(paths),np.float64)
    for i in range(len(paths)):
        p=plane.reshape(1,2).copy()
        for step in range(lengths[i]*4):
            t=np.float32((step+1)/120)
            movement=(vectors[paths[i,step//4]]*np.float32(2)).reshape(1,2)
            p,hit,mass=search_step(p,movement,half_size,bullets+velocity*t,
                np.float32(1)+error*t,table,integral,True)
            result[i]+=np.sum(mass)
    return result
'''
with (HERE/'readiness_kernel.py').open('x',encoding='utf-8') as f:f.write(kernel)
old_design=json.loads((ROOT/'diagnostics/planner_small_regression_20260921/design.json').read_text())
design=dict(maximum_complete_episode_executions=50,planned_complete_episode_executions=40,
    paired_seeds=old_design['paired_seeds'],known_failure_seeds=old_design['known_failure_seeds'],
    production_modified=False,training_started=False,formal_evaluation=False,
    scope=['rolling-window uncertainty consistency','full-route sensitivity ranking','renewed history lifetime and own-route eligibility'],
    acceptance=['mechanism evidence and limitation for each issue','image-only candidate without new online component',
      'matched full episodes report rescues and regressions','source and weight content unchanged',
      'no production promotion without fixed-200 acceptance'],
    source_files=json.loads((ROOT/'diagnostics/planner_consistency_20260921/design.json').read_text())['source_files'])
with (HERE/'design.json').open('x',encoding='utf-8') as f:json.dump(design,f,indent=2)
print('Frozen candidate and 40-episode plan created; no evaluation started.')
