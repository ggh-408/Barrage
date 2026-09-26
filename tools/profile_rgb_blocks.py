"""Sample exact RGB detector blocks and retain image-only regression samples."""
import argparse
import ast
import copy
from datetime import datetime
import json
import inspect
from pathlib import Path
import pickle
import runpy
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.profile_tracker_blocks import BlockProfiler


class RGBProfiler(BlockProfiler):
    specs={
        'detect':[(461,470,'input_and_background'),(471,473,'foreground'),
                  (477,477,'classify'),(478,482,'semantic_buffer'),(483,509,'bullet_centers'),
                  (517,527,'normalize_bullets'),(528,535,'semantic_bullets'),
                  (536,548,'plane_center'),(549,555,'semantic_plane_and_result')],
        '_foreground_mask':[], '_classify_foreground':[], '_candidate_hypotheses':[],
        '_validated_candidates':[(106,110,'prepare'),(111,129,'template_validation')],
        '_predictive_centers':[(268,275,'hypotheses_and_validation'),(276,301,'predicted_matching'),
                               (305,317,'recovery_scan_mask'),(318,320,'erase_predicted'),
                               (321,339,'bounded_recovery'),(345,348,'residual_recovery'),
                               (349,360,'merge_unique')],
        '_select_cover':[(163,176,'prepare'),(181,207,'coverage_and_gains'),
                         (208,216,'reverse_pixel_index'),(217,236,'greedy_cover'),(237,240,'result')],
        '_erase_sprite_coverage':[], '_squared_distance_2d':[],
    }
    def __init__(self,sample_every=10,capture_every=60,blocks=True):
        super().__init__(sample_every)
        self.capture_every=capture_every;self.samples=[];self.blocks=blocks;self.in_decision=False
        # Anchor ranges to their method definitions so inserting wrappers does
        # not silently attribute timings to unrelated source lines.
        from barrage_rl.live_screen import DominantBackgroundSemanticizer as Detector
        tree=ast.parse(Path(inspect.getsourcefile(Detector)).read_text(encoding='utf-8'))
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==Detector.__name__)
        starts={n.name:n.lineno for n in owner.body if isinstance(n,ast.FunctionDef)}
        anchors={'detect':453,'_validated_candidates':101,'_predictive_centers':261,'_select_cover':156}
        self.specs=copy.deepcopy(type(self).specs)
        for name,old_start in anchors.items():
            if name in ('_validated_candidates','_select_cover') and hasattr(Detector,name+'_numpy'):
                self.specs[name]=[]
            else:
                shift=starts[name]-old_start
                self.specs[name]=[(a+shift,b+shift,label) for a,b,label in self.specs[name]]

    def install(self):
        from barrage_rl.live_screen import LiveVisualController
        from barrage_rl.live_screen import DominantBackgroundSemanticizer as Detector
        if self.blocks:
            for name in self.specs:self.compile_method(Detector,name)
        original=Detector.detect
        def detect(detector,image,**kwargs):
            if not self.in_decision:return original(detector,image,**kwargs)
            self.decisions+=1
            sampled=self.blocks and (self.decisions-1)%self.sample_every==0
            capture=self.capture_every>0 and (self.decisions-1)%self.capture_every==0
            record=None
            if capture:
                record=dict(state=copy.deepcopy(vars(detector)),image=image.copy(),kwargs=copy.deepcopy(kwargs))
            self.active=sampled;self.current={};started=time.perf_counter()
            try:result=original(detector,image,**kwargs)
            finally:
                elapsed=(time.perf_counter()-started)*1000;self.active=False
                self.rows.append(dict(decision=self.decisions,sampled=sampled,total_ms=elapsed,blocks=self.current if sampled else {}))
            if record is not None:
                record.update(result=copy.deepcopy(result),after=copy.deepcopy(vars(detector)))
                self.samples.append(record)
            return result
        Detector.detect=detect;self.restores.append((Detector,'detect',original))
        observe=LiveVisualController.observe_due_surface
        def measured_observe(controller,surface):
            self.in_decision=True
            try:return observe(controller,surface)
            finally:self.in_decision=False
        LiveVisualController.observe_due_surface=measured_observe
        self.restores.append((LiveVisualController,'observe_due_surface',observe))

    def timing(self):
        values=np.asarray([r['total_ms'] for r in self.rows])
        return dict(count=len(values),mean_ms=float(values.mean()),p95_ms=float(np.percentile(values,95)),
                    p99_ms=float(np.percentile(values,99)),max_ms=float(values.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--no-blocks',action='store_true')
    parser.add_argument('--capture-every',type=int,default=60)
    args=parser.parse_args()
    output=ROOT/'diagnostics'/('rgb_blocks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    profiler=RGBProfiler(capture_every=args.capture_every,blocks=not args.no_blocks)
    profiler.install();previous=sys.argv
    try:
        sys.argv=[str(ROOT/'tools/diagnose_window_pacing.py'),'--seconds',str(args.seconds),
                  '--record-window-state','--output-dir',str(output)]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        profiler.close();sys.argv=previous
        output.mkdir(parents=True,exist_ok=True)
        (output/'rgb_updates.json').write_text(json.dumps(profiler.rows),encoding='utf-8')
        if not args.no_blocks:
            (output/'rgb_blocks.json').write_text(json.dumps(profiler.report(),indent=2),encoding='utf-8')
        (output/'rgb_timing.json').write_text(json.dumps(profiler.timing(),indent=2),encoding='utf-8')
        if profiler.samples:
            with (output/'rgb_samples.pkl').open('wb') as stream:pickle.dump(profiler.samples,stream,protocol=5)
    print('RGB timing: '+json.dumps(profiler.timing()),flush=True)
    print('Results: '+str(output),flush=True)


if __name__=='__main__':main()
