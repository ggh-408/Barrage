"""Exact image replay and same-configuration foreground before/after tests."""
import ast
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'diagnostics/five_stage_20260925'


def before_function(name):
    import barrage_rl.runtime_core as core
    path=OUT/'runtime_core_before.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
    namespace=dict(vars(core));exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace[name]


def validation():
    import pygame
    from barrage_rl.live_screen import DominantBackgroundSemanticizer as D
    from barrage_rl.runtime_core import snapshot_surface_rgb,tracker_prediction_hints
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    from tools.benchmark_tracker_exact import exact
    old_capture=before_function('snapshot_surface_rgb');old_hints=before_function('tracker_prediction_hints')
    with (ROOT/'diagnostics/rgb_blocks_20260925_112807_783291/rgb_samples.pkl').open('rb') as f:samples=pickle.load(f)
    original_classify=D._classify_foreground
    surface=pygame.Surface((820,820),depth=32)
    for sample in samples:
        image=sample['image'];pygame.surfarray.blit_array(surface,image.transpose(1,0,2))
        exact(old_capture(surface),snapshot_surface_rgb(surface));assert not surface.get_locked()
        a,b=D(),D();a.__dict__.update(copy.deepcopy(sample['state']));b.__dict__.update(copy.deepcopy(sample['state']))
        try:
            D._classify_foreground=staticmethod(D._classify_foreground_numpy)
            expected=a.detect(image,**sample['kwargs'])
        finally:D._classify_foreground=staticmethod(original_classify)
        actual=b.detect(image,**sample['kwargs'])
        exact(asdict(expected),asdict(actual));exact(vars(a),vars(b))
    with (ROOT/'diagnostics/tracker_exact_20260925/after/image_detections.pkl').open('rb') as f:records=pickle.load(f)
    extractor=TrackedFeatureExtractor()
    for i,(d,kw,_) in enumerate(records):
        extractor.step_detections(d.bullet_positions,d.plane_position,**kw)
        exact(old_hints(extractor.tracker,(820,820,3)),tracker_prediction_hints(extractor.tracker,(820,820,3)))
        if i%1200==1199:print(f'Hints: {i+1}/{len(records)} bitwise equal',flush=True)
    result=dict(rgb_snapshots_and_complete_detector_state_exact=len(samples),sequential_prediction_hints_exact=len(records))
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


def window(mode):
    import barrage_rl.runtime_core as core
    import barrage_rl.live_screen as live
    if mode=='before':
        core.snapshot_surface_rgb=live.snapshot_surface_rgb=before_function('snapshot_surface_rgb')
        core.tracker_prediction_hints=live.tracker_prediction_hints=before_function('tracker_prediction_hints')
        live.DominantBackgroundSemanticizer._classify_foreground=staticmethod(live.DominantBackgroundSemanticizer._classify_foreground_numpy)
    sys.argv=[str(ROOT/'tools/diagnose_window_pacing.py'),'--seconds','120',
              '--record-window-state','--deep-model','--output-dir',str(OUT/mode)]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='validate':validation()
    elif mode in ('before','after'):window(mode)
    else:raise ValueError(mode)
