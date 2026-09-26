"""Install isolated existing-component corrections; no simulator input."""
import sys
import types
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHECKPOINT=ROOT/'diagnostics/risk_removal_20260920/policy_teacher_cost.pt'
def install_candidate(agent):
    sys.path.insert(0,str(HERE)) if str(HERE) not in sys.path else None
    spec=importlib.util.spec_from_file_location('readiness_controller',HERE/'candidate_controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    guard=agent._receding_pixel_guard
    for name in ('apply','_assess','_geometry'):
        setattr(guard,name,types.MethodType(getattr(module.ContinuationMixin,name),guard))
    original_manifest=guard.manifest
    def manifest():
        return {**original_manifest(), 'experimental_source':str(HERE/'candidate_controller.py'),
            'safety_certificate':'conditional sensitivity test for observed tracks; excludes future births and association errors',
            'ranking_sensitivity':'integrated rectangle overlap; not a probability',
            'velocity_error_age_cap_seconds':7/30}
    guard.manifest=manifest
    return guard
