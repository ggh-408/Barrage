"""Measure frame pacing and decision placement without changing game source."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--no-ai', action='store_true')
    parser.add_argument('--immune', action='store_true')
    parser.add_argument('--window-state', action='store_true', help='Record actual foreground/minimized state and create a small focus target')
    parser.add_argument('--record-window-state', action='store_true', help='Record focus without creating another window')
    parser.add_argument('--deep-model', action='store_true', help='Enable module-level model timing')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    import pygame
    import barrage_rl.live_screen as live
    import barrage_rl.runtime_core as core
    output = args.output_dir or ROOT / 'diagnostics' / ('window_pacing_' + time.strftime('%Y%m%d_%H%M%S'))
    frames = []
    current = None
    restores = []
    focus_target = None
    user32 = None
    if args.window_state or args.record_window_state:
        import ctypes
        from ctypes import wintypes
        import subprocess
        user32 = ctypes.WinDLL('user32')
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        if args.window_state:
            focus_target = subprocess.Popen([sys.executable, '-B', str(ROOT/'tools/window_focus_target.py')])
    def wrap(obj, name, label):
        original = getattr(obj, name)
        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                if current is not None:
                    current[label] = current.get(label, 0.) + (time.perf_counter()-start)*1000
                    if label == 'decision_ms':
                        current['decisions'] += 1
        setattr(obj, name, measured)
        restores.append((obj, name, original))
    original_clock = pygame.time.Clock
    class Clock:
        def __init__(self):
            self.clock = original_clock()
        def tick_busy_loop(self, fps):
            nonlocal current
            start = time.perf_counter()
            current = dict(start=start, decisions=0, busy_loop=True)
            if user32 is not None:
                hwnd = pygame.display.get_wm_info()['window']
                current.update(foreground=(user32.GetForegroundWindow() == hwnd),
                               minimized=bool(user32.IsIconic(hwnd)),
                               visible=bool(user32.IsWindowVisible(hwnd)))
            dt = self.clock.tick_busy_loop(fps)
            current.update(tick_ms=(time.perf_counter()-start)*1000,
                           tick_dt_ms=dt, previous_work_ms=self.clock.get_rawtime())
            return dt
    pygame.time.Clock = Clock
    restores.append((pygame.time, 'Clock', original_clock))
    wrap(live.LiveVisualController, 'observe_due_surface', 'decision_ms')
    for name, label in [('render_world_surface', 'world_render_ms'),
                        ('advance_bullet_field', 'bullet_physics_ms'),
                        ('advance_plane', 'plane_physics_ms'),
                        ('colliding_bullet_indices', 'collision_ms')]:
        wrap(core, name, label)
    original_flip = pygame.display.flip
    def flip():
        started = time.perf_counter()
        original_flip()
        if current is not None:
            current['flip_ms'] = (time.perf_counter()-started)*1000
            current['frame_ms'] = (time.perf_counter()-current['start'])*1000
            frames.append(current.copy())
    pygame.display.flip = flip
    restores.append((pygame.display, 'flip', original_flip))
    previous = sys.argv
    try:
        sys.argv = [str(ROOT/'tools/test_visible_window.py'), '--seconds', str(args.seconds),
                    '--output-dir', str(output)]
        if args.deep_model:
            sys.argv.append('--deep-model')
        if args.immune:
            sys.argv.append('--immune')
        if args.no_ai:
            sys.argv.append('--no-ai')
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = previous
        for obj, name, original in reversed(restores):
            setattr(obj, name, original)
        if focus_target is not None:
            focus_target.terminate()
            focus_target.wait(timeout=5)
        output.mkdir(parents=True, exist_ok=True)
        (output/'frame_pacing.json').write_text(json.dumps(frames), encoding='utf-8')
    print('Pacing diagnostic: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
