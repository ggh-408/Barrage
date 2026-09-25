"""Privileged labels audit interval coverage; labels never enter policy input."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pixel_guard_candidate import PixelGuard, PixelGuardConfig


def main():
    out=ROOT/'diagnostics/pixel_guard_validation/interval_coverage.json'
    if out.exists():raise FileExistsError(out)
    source=ROOT/'diagnostics/success_plateau_audit_20260905'
    data=json.loads((source/'current_failure_reproduction.json').read_text())
    z=np.load(source/'current_terminal_features.npz')
    guard=PixelGuard(PixelGuardConfig())
    rows=[]
    for i,f in enumerate(data['failures']):
        o=z['objects'][i]; g=z['globals'][i]
        observed=g[:2]*820
        plane_estimate=observed-guard.centroid_bias
        plane_truth=np.asarray(f['backtrace'][-1]['plane_center'])
        plane_error=plane_estimate-plane_truth
        dt=f['physics_substeps_in_terminal_action']/120
        bullet=f['collision_bullets'][0]
        bullet_truth=np.asarray(bullet['position'])-np.asarray(bullet['velocity'])*dt
        estimates=observed[None]+o[:,:2]*820-guard.bullet_bias
        distances=np.linalg.norm(estimates-bullet_truth,axis=1)
        distances[~z['masks'][i]]=np.inf
        j=int(np.argmin(distances))
        velocity=(o[j,2:4]+g[6:8])*240
        error=estimates[j]-bullet_truth
        terminal_error=estimates[j]+velocity*dt-np.asarray(bullet['position'])
        bound=.5+2/max(float(o[j,10]),2/30)*dt
        rows.append(dict(seed=f['seed'],nearest_track_distance=float(distances[j]),
            plane_center_error=plane_error.tolist(),bullet_center_error=error.tolist(),
            terminal_bullet_error=terminal_error.tolist(),terminal_bullet_bound=bound,
            plane_position_covered=bool(np.all(np.abs(plane_error)<=.5001)),
            bullet_position_covered=bool(np.all(np.abs(error)<=.5001)),
            terminal_bullet_covered=bool(np.all(np.abs(terminal_error)<=bound+1e-4))))
    report={'kind':'selected_baseline_failure_interval_coverage',
            'privileged_labels_are_policy_input':False,'rows':rows,
            'limitation':'nearest track matching; seven selected terminal samples cannot certify a bound'}
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
