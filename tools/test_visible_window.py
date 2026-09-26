"""Standalone visible-window FPS/latency diagnostic; leaves game source unchanged."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_for_test_foreground(pygame, is_foreground):
    """Keep activation denial recoverable; start timing only after actual focus."""
    if is_foreground():
        return True
    pygame.display.set_caption('Barrage test - click this window to start')
    surface = pygame.display.get_surface()
    pygame.font.init()
    font = pygame.font.Font(None, 30)
    surface.fill((20, 24, 32))
    surface.blit(font.render('Click this window to start the foreground test.', True,
                             (240, 240, 240)), (25, 40))
    pygame.display.flip()
    print('Waiting for game window focus. Click the game window to start; timing has not started.', flush=True)
    while not is_foreground():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
        time.sleep(0.02)
    return True


def foreground_test_window(pygame):
    """Activate this diagnostic's own window before the measurement starts."""
    if sys.platform != 'win32':
        from pygame._sdl2.video import Window
        Window.from_display_module().focus()
        return wait_for_test_foreground(pygame, pygame.key.get_focused)
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = pygame.display.get_wm_info()['window']
    user32.ShowWindow(hwnd, 9)  # Restore this test window if minimized.
    user32.SetForegroundWindow(hwnd)
    pygame.event.pump()
    return wait_for_test_foreground(pygame, lambda: user32.GetForegroundWindow() == hwnd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=120.0)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--breakdown', action='store_true', help='Record exclusive decision-stage timings')
    parser.add_argument('--deep-model', action='store_true', help='Time all invoked neural modules and shared geometry')
    parser.add_argument('--immune', action='store_true', help='Disable collision damage for timing comparisons')
    parser.add_argument('--no-ai', action='store_true', help='Use manual control through the same game entry point')
    parser.add_argument('--rgb-workers', type=int, default=None)
    parser.add_argument('--foreground', action=argparse.BooleanOptionalAction, default=True,
                        help='Activate the test window before timing (default); disable for background diagnostics')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')

    # Reuse only the desktop check, never the historical diagnostic entry point.
    from tools.measure_visible_latency import verify_visible_desktop
    desktop = verify_visible_desktop()
    output = (args.output_dir or ROOT / 'diagnostics' /
              ('visible_window_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'window.json'
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    import pygame
    original_set_mode = pygame.display.set_mode
    foreground_confirmed = None
    def set_mode(*a, **kw):
        nonlocal foreground_confirmed
        surface = original_set_mode(*a, **kw)
        if args.foreground:
            foreground_confirmed = foreground_test_window(pygame)
        return surface
    pygame.display.set_mode = set_mode
    profiler = None
    if args.breakdown or args.deep_model:
        from tools.window_latency_breakdown import WindowProfiler
        profiler = WindowProfiler(deep_model=args.deep_model)
        profiler.install()
    try:
        os.chdir(ROOT)
        sys.argv = [str(ROOT / 'Barrage.py'), '--ai', '--no-music', '--latency-allow-damage',
                    '--bullets', '300', '--targeted-probability', '0.10',
                    '--seed', str(args.seed), '--latency-test-seconds', str(args.seconds),
                    '--latency-output', str(report_path)]
        if args.immune:
            sys.argv.remove('--latency-allow-damage')
        if args.no_ai:
            sys.argv[1] = '--no-ai'
        if args.rgb_workers is not None:
            sys.argv.extend(['--rgb-workers', str(args.rgb_workers)])
        namespace = runpy.run_path(str(ROOT / 'Barrage.py'), run_name='__main__')
    finally:
        pygame.display.set_mode = original_set_mode
        if profiler is not None:
            profiler.close()
        sys.argv = previous_argv
        os.chdir(previous_cwd)

    game = namespace['Barrage']
    import torch
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if profiler is not None:
        breakdown = profiler.report()
        (output / 'latency_breakdown.json').write_text(json.dumps(breakdown, indent=2), encoding='utf-8')
        (output / 'decision_timings.json').write_text(json.dumps(profiler.rows), encoding='utf-8')
    with report_path.with_suffix('.frames.csv').open(encoding='utf-8', newline='') as stream:
        frames = list(csv.DictReader(stream))
    if len(frames) != report['rendered_frames']:
        raise RuntimeError('Frame count differs from the window report')
    complete_seconds = int(report['duration_seconds'])
    buckets = [0] * complete_seconds
    for row in frames:
        second = int(float(row['elapsed_seconds']))
        if 0 <= second < complete_seconds:
            buckets[second] += 1
    report.update(
        checkpoint=None if args.no_ai else str(namespace['DEFAULT_AI_CHECKPOINT']),
        requested_seconds=args.seconds,
        foreground_requested=args.foreground,
        foreground_confirmed_at_window_creation=foreground_confirmed,
        alive_at_end=bool(game.KEY),
        alive_physics_steps=int(game.ALIVE_PHYSICS_STEPS),
        seed=args.seed, ai_device=None if args.no_ai else str(game.AI_CONTROLLER.agent.device),
        torch_threads=torch.get_num_threads(), desktop_verification=desktop,
        rgb_workers=None if args.no_ai else game.AI_CONTROLLER.rgb_workers,
        runtime_stages={} if args.no_ai else game.AI_CONTROLLER.runtime_stage_report(),
        fps_complete_seconds=buckets,
        timing_scope={
            'fps': 'Completed pygame display flips per wall-clock second; partial final second excluded from buckets.',
            'ai_decision': 'RGB surface capture through action selection at synchronous decision boundaries.',
            'frame_processing': 'Full frame cycle including FPS pacing wait.',
            'exclusions': 'Window initialization precedes measurement; lazy compilation during gameplay remains included. Monitor scanout and external input latency are unmeasured.',
            'damage': ('Collision damage disabled.' if args.immune else 'Collision damage enabled. Measurement continues to the time limit after death; post-death frames may have a different workload.'),
        },
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    ai = report['ai_decision']
    lines = [
        f"Checkpoint: {report['checkpoint']}",
        f"Visible window: {report['duration_seconds']:.2f} s, {report['rendered_frames']} frames",
        f"Average FPS: {report['rendered_fps']:.2f}",
        f"AI latency mean / P95 / P99 / max: {ai['mean_ms']:.2f} / {ai['p95_ms']:.2f} / {ai['p99_ms']:.2f} / {ai['max_ms']:.2f} ms",
        f"Decision budget: {report['decision_budget_ms']:.2f} ms; over budget: {report['decision_over_budget_count']}/{ai['count']}",
        'Frame-cycle timing includes FPS pacing. Display hardware latency is unmeasured.',
        'First-use compilation during gameplay is included. Slow frames may extend the requested duration.',
        f"Collision damage {'disabled' if args.immune else 'enabled'}; alive at end: {bool(game.KEY)}. Post-death frames remain in timing statistics.",
    ]
    if buckets:
        lines.append(f'Full-second FPS min / max: {min(buckets)} / {max(buckets)}')
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
