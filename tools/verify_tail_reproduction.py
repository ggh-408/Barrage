"""Check whether small training reproduction differences change decisions."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys
import json
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from barrage_rl.artifacts import atomic_write_json
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.deployment import configure_image_controller


def main():
    output=ROOT/'diagnostics/imitation_tail_audit_20260920/reproduction_decisions.json'
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    paths=[ROOT/'runs/visual_set_v49/best.pt',output.parent/'legacy_reproduction.pt']
    agents=[load_tracked_agent(str(p),torch.device('cuda'),analytic_shield=True,
                analytic_shield_gate='learned_all_unsafe')[0] for p in paths]
    started=time.perf_counter()
    with np.load(ROOT/'runs/visual_set_v49/replay_latest.npz') as data:
        features=[data[k] for k in ('objects','masks','globals')]
    mismatches={'raw_policy_actions':0,'learned_filter_actions':0,'final_actions':0}
    maximum_risk_difference=0.
    for start in range(0,len(features[0]),512):
        batch=[x[start:start+512].astype(bool if i==1 else np.float32) for i,x in enumerate(features)]
        rows=[agent.act_features_with_diagnostics(*batch) for agent in agents]
        mismatches['final_actions']+=int(np.sum(rows[0][0]!=rows[1][0]))
        for key in ('raw_policy_actions','learned_filter_actions'):
            mismatches[key]+=int(np.sum(rows[0][1][key]!=rows[1][1][key]))
        maximum_risk_difference=max(maximum_risk_difference,
            float(np.max(np.abs(rows[0][1]['selected_immediate_risk']-rows[1][1]['selected_immediate_risk']))))
    count=len(features[0]);del features
    for agent in agents: configure_image_controller(agent,'receding',search_workers=1)
    with np.load(ROOT/'runs/visual_set_v50/continuation_labels.npz') as data:
        features=[data[k] for k in ('objects','masks','globals')]
    guard_mismatches=0
    for start in range(0,len(features[0]),10):
        # Independent recorded roots have no shared route history.
        for agent in agents: agent.reset_state()
        batch=[x[start:start+10] for x in features]
        decisions=[agent.act_features(*batch) for agent in agents]
        guard_mismatches+=int(np.sum(decisions[0]!=decisions[1]))
    report=dict(checkpoints=[str(p) for p in paths],replay_states=count,
        mismatches=mismatches,maximum_selected_risk_difference=maximum_risk_difference,
        independent_guard_states=len(features[0]),guard_mismatches=guard_mismatches,
        all_tested_decisions_equal=not any(mismatches.values()) and guard_mismatches==0,
        elapsed_seconds=time.perf_counter()-started,
        scope='Saved replay and independent image snapshots; not a new episode success estimate')
    atomic_write_json(output,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
