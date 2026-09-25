"""Shared image-controller installation for collection, evaluation and game."""
from dataclasses import asdict
import sys
from pathlib import Path


def configure_image_controller(agent, kind="receding", *, search_workers=1):
    if kind == "off":
        return dict(kind="off")
    if kind != "receding":
        raise ValueError(f"Unsupported shared image controller: {kind}")
    try:
        import numba
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime/pixel_search'))
        import numba
    # Compilation is process-local; source-derived persistent cache identifiers
    # are unnecessary for these experiments.
    if not getattr(numba.njit, '_barrage_uncached', False):
        original = numba.njit
        def uncached(*args, **kwargs):
            kwargs['cache'] = False
            return original(*args, **kwargs)
        uncached._barrage_uncached = True
        numba.njit = uncached
    from tools.pixel_guard_receding import RecedingGuardConfig, install_receding_guard
    config = RecedingGuardConfig(search_workers=int(search_workers))
    guard = install_receding_guard(agent, config)
    return dict(kind='receding', algorithm='receding_continuation', config=asdict(config))
