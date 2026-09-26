"""Correlate visible decision stages with cyclic-GC pauses and CPU time."""
from __future__ import annotations
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TailProbe:
    def __init__(self):
        self.rows = []
        self.events = []
        self.stack = []
        self.current = None
        self.restores = []
        self.gc_start = None

    def gc_event(self, phase, info):
        if phase == 'start':
            self.gc_start = (time.perf_counter(), time.thread_time(),
                             len(self.rows)+1 if self.current is not None else None,
                             list(self.stack))
        elif self.gc_start is not None:
            start, cpu, decision, stack = self.gc_start
            self.events.append(dict(start=start, duration_ms=(time.perf_counter()-start)*1000,
                                    cpu_ms=(time.thread_time()-cpu)*1000, decision=decision,
                                    stages=stack, **info))
            self.gc_start = None

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            if self.current is None:
                return original(*args, **kwargs)
            start, cpu = time.perf_counter(), time.thread_time()
            self.stack.append(label)
            try:
                return original(*args, **kwargs)
            finally:
                self.stack.pop()
                self.current['stages'].setdefault(label, []).append(dict(
                    start=start, ms=(time.perf_counter()-start)*1000,
                    cpu_ms=(time.thread_time()-cpu)*1000))
        setattr(obj,name,measured)
        self.restores.append((obj,name,original))

    def install(self):
        import barrage_rl.live_screen as live
        from barrage_rl.tracked_policy import TrackedFeatureExtractor, TrackedPolicyAgent
        from barrage_rl.image_oracle import PersistentImageTracker
        for obj, name, label in [
            (live,'snapshot_surface_rgb','capture'),
            (live.LiveVisualController,'_prediction_hints','prediction_hints'),
            (live.DominantBackgroundSemanticizer,'detect','rgb_detection'),
            (TrackedFeatureExtractor,'step_detections','tracking_features'),
            (PersistentImageTracker,'_fit_velocities','velocity_batch'),
            (TrackedPolicyAgent,'act_features','model_and_planner'),
        ]:
            self.wrap(obj,name,label)
        original = live.LiveVisualController.observe_due_surface
        def observe(controller, surface):
            start, cpu = time.perf_counter(), time.thread_time()
            self.current = dict(decision=len(self.rows)+1,start=start,stages={},
                                gc_count=gc.get_count())
            try:
                return original(controller,surface)
            finally:
                self.current.update(ms=(time.perf_counter()-start)*1000,
                                    cpu_ms=(time.thread_time()-cpu)*1000)
                self.rows.append(self.current)
                self.current = None
        live.LiveVisualController.observe_due_surface = observe
        self.restores.append((live.LiveVisualController,'observe_due_surface',original))
        gc.callbacks.append(self.gc_event)

    def close(self):
        gc.callbacks.remove(self.gc_event)
        for obj,name,original in reversed(self.restores):
            setattr(obj,name,original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--deep-model', action='store_true')
    args = parser.parse_args()
    output = ROOT/'diagnostics'/('decision_tail_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    probe = TailProbe()
    probe.install()
    previous = sys.argv
    try:
        if args.deep_model:
            sys.argv = [str(ROOT/'tools/diagnose_window_pacing.py'), '--seconds', str(args.seconds),
                        '--record-window-state', '--deep-model', '--output-dir', str(output)]
        else:
            sys.argv = [str(ROOT/'tools/benchmark_tracker_batch.py'),'--variant','batch',
                        '--seconds',str(args.seconds),'--skip-replay','--output-dir',str(output)]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        probe.close()
        sys.argv = previous
        output.mkdir(parents=True,exist_ok=True)
        data = dict(decisions=probe.rows,gc_events=probe.events)
        (output/'tail_trace.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('Slow decisions: '+json.dumps(sorted(probe.rows,key=lambda r:r['ms'],reverse=True)[:3]),flush=True)
    print('Long GC: '+json.dumps([e for e in probe.events if e['duration_ms']>10]),flush=True)
    print('Results: '+str(output),flush=True)


if __name__ == '__main__':
    main()
