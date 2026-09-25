"""Trace process-local JIT compilation while running the unchanged visible window."""
import json
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from numba.core.dispatcher import Dispatcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15)
    args = parser.parse_args()
    output = ROOT / 'diagnostics' / ('window_compile_trace_' + time.strftime('%Y%m%d_%H%M%S'))
    records = []
    original = Dispatcher.compile
    origin = time.perf_counter()
    import barrage_rl.live_screen as live
    original_observe = live.LiveVisualController.observe_due_surface
    in_decision = False
    def observe_traced(controller, surface):
        nonlocal in_decision
        in_decision = True
        try:
            return original_observe(controller, surface)
        finally:
            in_decision = False
    live.LiveVisualController.observe_due_surface = observe_traced
    def compile_traced(dispatcher, signature):
        before = len(dispatcher.signatures)
        start = time.perf_counter()
        try:
            return original(dispatcher, signature)
        finally:
            elapsed = time.perf_counter() - start
            if elapsed > .01:
                records.append(dict(function=dispatcher.py_func.__module__ + '.' + dispatcher.py_func.__name__,
                                    during_decision=in_decision,
                                    start_seconds=start-origin, elapsed_ms=elapsed*1000,
                                    signatures_before=before, signatures_after=len(dispatcher.signatures),
                                    signature=str(signature)))
    Dispatcher.compile = compile_traced
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / 'tools/test_visible_window.py'), '--seconds', str(args.seconds),
                    '--breakdown', '--output-dir', str(output)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        Dispatcher.compile = original
        live.LiveVisualController.observe_due_surface = original_observe
        sys.argv = previous
        output.mkdir(parents=True, exist_ok=True)
        (output / 'compilation.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Compilation trace: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
