"""Hierarchical CPU wall-time profiling of the real window image policy only."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import argparse,csv,json,os,runpy,sys,time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class Profiler:
    def __init__(self):
        self.active=False;self.stack=[];self.current={};self.rows=[]
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        def measured(*a,**kw):
            if not self.active:return original(*a,**kw)
            frame=[time.perf_counter_ns(),0];self.stack.append(frame)
            try:return original(*a,**kw)
            finally:
                elapsed=time.perf_counter_ns()-frame[0];self.stack.pop()
                if self.stack:self.stack[-1][1]+=elapsed
                v=self.current.setdefault(label,[0,0,0]);v[0]+=elapsed/1e6;v[1]+=(elapsed-frame[1])/1e6;v[2]+=1
        setattr(obj,name,measured)
        return original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--bullets',type=int,default=300)
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--seed',type=int,default=2131058160)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.seconds<=0 or args.bullets<300 or args.threads<1:
        parser.error('seconds/threads must be positive and bullets must be at least 300')
    os.environ['SDL_VIDEODRIVER']='windows'
    from tools.measure_visible_latency import verify_visible_desktop
    desktop_verification=verify_visible_desktop()
    out=args.output or ROOT/'diagnostics'/f'window_policy_breakdown_{args.bullets}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
    import numpy as np
    import torch
    import barrage_rl.live_screen as live
    import barrage_rl.tracked_policy as tracked
    from barrage_rl.image_oracle import PersistentImageTracker
    from tools.pixel_guard_receding import install_receding_guard
    import tools.pixel_recovery_planner as search
    import tools.pixel_recovery_refined as young
    import tools.pixel_search_kernel as kernel
    from barrage_rl.artifacts import sha256_file
    profiler=Profiler();guards=[];agents=[];controllers=[];last_detection=[];last_features=[]
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    source_paths=[ROOT/'Barrage.py',*sorted((ROOT/'barrage_rl').glob('*.py')),*sorted((ROOT/'tools').glob('pixel_*.py')),Path(__file__)]
    source_hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in source_paths}
    checkpoint_hash=sha256_file(checkpoint)
    (out/'manifest.json').write_text(json.dumps({'bullet_count':args.bullets,'threads':args.threads,'seed':args.seed,'seconds':args.seconds,'targeted_bullet_probability':0.10,'checkpoint_sha256':checkpoint_hash,'source_hashes':source_hashes,'desktop_verification':desktop_verification},indent=2))
    original_init=live.LiveVisualController.__init__
    original_observe=live.LiveVisualController.observe_due_surface
    for name in ['initialize_detections','update_detections','_update_measurements','_fit_velocity','_trim_tracks','_retention_threat']:
        profiler.wrap(PersistentImageTracker,name,'tracker.'+name)
    for module,name,label in [(live,'snapshot_surface_rgb','capture.rgb_copy'),(search,'recovery_action','recovery.established'),(young,'recovery_action','recovery.young'),(kernel,'search_step','recovery.compiled_kernel'),(tracked,'build_image_geometry_belief','geometry.belief'),(tracked,'constant_action_clearance_by_object','geometry.clearance')]:
        profiler.wrap(module,name,label)
    def initialize(self,*a,**kw):
        torch.set_num_threads(args.threads)
        original_init(self,str(checkpoint),**kw)
        assert self.action_delay_steps==0
        guards.append(install_receding_guard(self.agent));agents.append(self.agent);controllers.append(self)
        for name in ['detect','_foreground_mask','_classify_foreground','_predictive_centers','_candidate_centers','_candidate_hypotheses','_validated_candidates','_select_cover','_erase_sprite_coverage']:
            profiler.wrap(self.semanticizer,name,'detector.'+name)
        measured_detect=self.semanticizer.detect
        def capture_detect(*x,**y):
            result=measured_detect(*x,**y);last_detection[:]=[result];return result
        self.semanticizer.detect=capture_detect
        profiler.wrap(self,'_prediction_hints','detector.prediction_hints')
        # Class wrapper survives controller.reset replacing its extractor instance.
        measured_act=profiler.wrap(self.agent,'act_features','agent.total')
        timed_act=self.agent.act_features
        def capture_features(objects,masks,globals_,*x,**y):
            last_features[:]=[objects,masks,globals_]
            return timed_act(objects,masks,globals_,*x,**y)
        self.agent.act_features=capture_features
        profiler.wrap(self.agent._action_selector,'select','selector.total')
        for name in ['apply','hazards','rectangle_hits']:
            profiler.wrap(guards[-1],name,'guard.'+name)
        for name in ['search','_geometry','_assess']:
            if hasattr(guards[-1],name):profiler.wrap(guards[-1],name,'guard.'+name)
        import tools.pixel_receding_kernel as receding_kernel
        for name in ['beam_costs','beam_paths','assess_paths','assess_path_intervals']:
            profiler.wrap(receding_kernel,name,'recovery.'+name)
        profiler.wrap(self.agent.model,'forward_with_geometry','model.total')
        for name in ['prepare_action_geometry','action_geometry_features_from_shared']:
            profiler.wrap(self.agent.model,name,'model.'+name)
        profiler.wrap(self.agent.model._window_action_query_cache,'get','model.cached_action_query')
        for name,module in self.agent.model.named_modules():
            if name:profiler.wrap(module,'forward','nn.'+name)
    for name in ['reset_detections','step_detections','_features']:
        if hasattr(tracked.TrackedFeatureExtractor,name):profiler.wrap(tracked.TrackedFeatureExtractor,name,'features.'+name)
    def observe(self,surface):
        profiler.current={};profiler.active=True;started=time.perf_counter()
        try:result=original_observe(self,surface)
        finally:profiler.active=False
        total=(time.perf_counter()-started)*1000
        profiler.rows.append({'time':started,'total_ms':total,'stages':profiler.current})
        return result
    sys.argv=[str(ROOT/'Barrage.py'),'--ai','--bullets',str(args.bullets),'--targeted-probability','0.10','--seed',str(args.seed),'--no-music','--latency-test-seconds',str(args.seconds),'--latency-output',str(out/'window.json')]
    with patch.object(live.LiveVisualController,'__init__',initialize),patch.object(live.LiveVisualController,'observe_due_surface',observe):
        runpy.run_path(str(ROOT/'Barrage.py'),run_name='__main__')
    n=len(profiler.rows);labels=sorted(set(k for r in profiler.rows for k in r['stages']))
    if not n or not last_features:raise RuntimeError('No visible-window decisions recorded')
    window=json.loads((out/'window.json').read_text())
    if window['bullets']!=args.bullets:raise RuntimeError('Window bullet count mismatch')
    def stats(a):
        return {'mean_ms':float(np.mean(a)),'p95_ms':float(np.percentile(a,95)),'p99_ms':float(np.percentile(a,99)),'max_ms':float(np.max(a))}
    summary={}
    with (out/'stages.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['decision','elapsed_seconds','stage','inclusive_ms','exclusive_ms','calls'])
        for i,r in enumerate(profiler.rows):
            for k,v in r['stages'].items():writer.writerow([i,r['time']-profiler.rows[0]['time'],k,*v])
    for label in labels:
        values=np.array([r['stages'].get(label,[0,0,0]) for r in profiler.rows]);active=values[:,2]>0
        summary[label]={'inclusive_per_decision':stats(values[:,0]),'exclusive_per_decision':stats(values[:,1]),'active_decisions':int(active.sum()),'calls':int(values[:,2].sum()),'inclusive_when_active':stats(values[active,0]),'mean_per_call_ms':float(values[:,0].sum()/values[:,2].sum())}
    window_guard_manifest=guards[0].manifest()
    uncalled_nn=[name for name,_ in agents[0].model.named_modules() if name and 'nn.'+name not in summary]
    # Measure profiler perturbation on identical saved image features, without a game.
    objects,masks,globals_=[np.array(v,copy=True) for v in last_features]
    z={'objects':objects,'masks':masks,'globals':globals_,'seeds':np.full(len(objects),args.seed)}
    np.savez_compressed(out/'calibration_features.npz',**z,bullet_count=args.bullets)
    a=agents[0];timings={};reference={};rare=[]
    for active in [False,True,False,True]:
        samples=[]
        for i in range(60):
            j=i%len(z['seeds']);a.reset_state();profiler.current={};profiler.active=active
            started=time.perf_counter();action=a.act_features(z['objects'][j:j+1],z['masks'][j:j+1],z['globals'][j:j+1]);elapsed=(time.perf_counter()-started)*1000;profiler.active=False
            if j in reference:assert reference[j]==int(action[0])
            reference[j]=int(action[0])
            if i>=10:
                samples.append(elapsed)
                if active:rare.append(dict(profiler.current))
        timings.setdefault(str(active),[]).extend(samples)
    import copy
    tracking_calibration={}
    original_features=None
    for active in [False,True,False,True]:
        samples=[]
        for i in range(30):
            extractor=copy.deepcopy(controllers[0].tracked_extractor)
            det=last_detection[0];profiler.current={};profiler.active=active
            started=time.perf_counter()
            features=extractor.step_detections(det.bullet_positions,det.plane_position)
            elapsed=(time.perf_counter()-started)*1000;profiler.active=False
            if original_features is None:original_features=features
            else:
                for x,y in zip(features,original_features):np.testing.assert_array_equal(x,y)
            if i>=5:samples.append(elapsed)
        tracking_calibration.setdefault(str(active),[]).extend(samples)
    rare_summary={k:stats([r.get(k,[0,0,0])[0] for r in rare]) for k in ['recovery.young','recovery.established','recovery.compiled_kernel','guard.hazards','model.total']}
    changed=[p for p,h in source_hashes.items() if sha256_file(ROOT/p)!=h]
    if changed or sha256_file(checkpoint)!=checkpoint_hash:raise RuntimeError(f'Sources/checkpoint changed during profiling: {changed}')
    report={'tracking_calibration':{k:stats(v) for k,v in tracking_calibration.items()},'calibration_replay':rare_summary,'calibration_source':'last current-window image features; state reset before each calibration call; no historical features reused','uncalled_nn':uncalled_nn,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'source_hashes':source_hashes,'source_files_changed':changed,'desktop_verification':desktop_verification,'bullet_count':args.bullets,'targeted_bullet_probability':0.10,'seed':args.seed,'requested_seconds':args.seconds,'threads':args.threads,'device':'cpu','decisions':n,'total':stats([r['total_ms'] for r in profiler.rows]),'stages':summary,'calibration':{k:stats(v) for k,v in timings.items()},'guard':window_guard_manifest,'scope':'visible window current RGB policy only; game physics, rendering and pacing excluded; raw instrumented CPU wall time; child inclusive time must not be added to parent inclusive time'}
    (out/'breakdown.json').write_text(json.dumps(report,indent=2))
    with (out/'summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['stage','inclusive_mean_ms','exclusive_mean_ms','active_mean_ms','active_p99_ms','calls','active_decisions'])
        for k,v in summary.items():writer.writerow([k,v['inclusive_per_decision']['mean_ms'],v['exclusive_per_decision']['mean_ms'],v['inclusive_when_active']['mean_ms'],v['inclusive_when_active']['p99_ms'],v['calls'],v['active_decisions']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top=sorted(summary,key=lambda k:summary[k]['exclusive_per_decision']['mean_ms'],reverse=True)[:18]
    fig,ax=plt.subplots(figsize=(12,8),constrained_layout=True)
    ax.barh(top[::-1],[summary[k]['exclusive_per_decision']['mean_ms'] for k in top[::-1]],color='#248b85')
    ax.set(xlabel='Exclusive mean latency per decision (ms)',title=f'Visible window AI pipeline: {args.bullets} bullets, top exclusive costs (game excluded)');ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'breakdown.png',dpi=160);plt.close(fig)
    print('OUTPUT:',out,flush=True)


if __name__=='__main__':main()
