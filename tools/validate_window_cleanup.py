"""Direct old/new closed-loop comparison; no training or summary fingerprints."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'diagnostics/window_cleanup_20260925'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_function(path, name, namespace, owner=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = tree.body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    env = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env[name]


def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    import numpy as np
    import pygame
    import torch
    torch.set_num_threads(10)
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.image_oracle import PersistentImageTracker
    from barrage_rl import runtime_core
    from barrage_rl.window_runtime import CONTROLLER, configure_window_controller
    from barrage_rl.deployment import configure_image_controller
    from tools.benchmark_tracker_exact import exact
    source = BASE/'before_source'
    old_game = load_module('cleanup_before_game', source/'Barrage.py')
    new_game = load_module('cleanup_after_game', ROOT/'Barrage.py')
    planner = load_module('cleanup_before_planner', source/'diagnostics/planner_readiness_20260921/candidate_parallel.py')
    sys.path.insert(0, str(ROOT/'diagnostics/planner_readiness_20260921'))
    old_update = old_function(source/'barrage_rl/image_oracle.py', '_update_measurements',
        PersistentImageTracker._update_measurements.__globals__, 'PersistentImageTracker')
    old_hints = old_function(source/'barrage_rl/runtime_core.py', 'tracker_prediction_hints',
        runtime_core.tracker_prediction_hints.__globals__)
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    controllers = []
    for i, module in enumerate((old_game, new_game)):
        module.PROJECT_ROOT = ROOT
        game = module.Barrage
        game.SCREEN_WIDTH = game.SCREEN_HEIGHT = 820
        game.QUANTITY = 300
        game.BULLET_SIZE = 5
        game.PLANE_SPEED = game.BULLET_SPEED = 240.
        game.TARGETED_BULLET_PROBABILITY = .10
        game.TARGETED_PREDICTION_SCALE_MIN = .65
        game.TARGETED_PREDICTION_SCALE_MAX = 1.
        game.TARGETED_ANGULAR_NOISE = .08
        game.COLLISION = False
        game.INVINCIBLE = False
        game.KEY = True
        game.MUSIC = False
        game.AI_PIPELINE = None
        game.RNG = np.random.default_rng(20260925)
        game.window = pygame.Surface((820, 820))
        module.Plane.SKIN = 0
        started = time.perf_counter()
        controller = LiveVisualController(str(ROOT/'best.pt'),
            experimental_controller=CONTROLLER)
        if i == 0:
            controller.agent._window_skip_unused_teacher_cost = False
            tracker = controller.tracked_extractor.tracker
            tracker._update_measurements = types.MethodType(old_update, tracker)
            controller._prediction_hints = lambda rgb, steps, c=controller: old_hints(c.tracked_extractor.tracker, rgb.shape, steps)
            configure_image_controller(controller.agent, search_workers=9)
            guard = controller.agent._receding_pixel_guard
            for name in ('apply', '_assess', '_geometry'):
                setattr(guard, name, types.MethodType(getattr(planner.ContinuationMixin, name), guard))
            planner.warmup_parallel_controller(guard)
        else:
            configure_window_controller(controller.agent)
        print(f'{i}: controller ready in {time.perf_counter()-started:.3f}s', flush=True)
        game.AI_CONTROLLER = controller
        game.reset_game()
        if i == 0:
            tracker = controller.tracked_extractor.tracker
            tracker._update_measurements = types.MethodType(old_update, tracker)
        controllers.append(controller)
    timings = [[], []]
    physics_times = [[], []]
    rng = np.random.default_rng(572)
    old_guard, new_guard = [c.agent._receding_pixel_guard for c in controllers]
    for case in range(24):
        paths = rng.integers(0, 9, (27, 15), dtype=np.int64)
        lengths = rng.integers(1, 16, 27, dtype=np.int64)
        plane = rng.uniform(20, 800, 2).astype(np.float32)
        bullets = plane + rng.uniform(-100, 100, (case * 3, 2)).astype(np.float32)
        velocity = rng.uniform(-240, 240, bullets.shape).astype(np.float32)
        error = rng.uniform(2, 240, len(bullets)).astype(np.float32)
        args = (plane, old_guard.half_size, bullets, velocity, error, paths, lengths)
        expected = old_guard._assess(*args)
        exact(expected, new_guard._assess(*args))
        certificate = new_guard._certificate(*args)
        exact(expected[:, 4:6], certificate[:, 4:6])
        exact(expected, new_guard._assess(*args, certificate=certificate))
    print('24 varied path metric/certificate cases exactly equal', flush=True)
    for frame in range(3601):
        if frame:
            for i, module in enumerate((old_game, new_game)):
                game = module.Barrage
                started = time.perf_counter()
                for _ in range(4):
                    game.advance_physics(runtime_core.ACTION_VECTORS[game.AI_ACTION], 1/120)
                physics_times[i].append((time.perf_counter()-started)*1000)
                game.render_world()
                started = time.perf_counter()
                game.AI_ACTION = controllers[i].observe_due_surface(game.window)
                timings[i].append((time.perf_counter()-started)*1000)
        a, b = old_game.Barrage, new_game.Barrage
        assert a.AI_ACTION == b.AI_ACTION, ('action', frame, a.AI_ACTION, b.AI_ACTION)
        exact(np.asarray(old_game.Bullet.LIST, np.float32), np.asarray(new_game.Bullet.LIST, np.float32))
        exact(a.PLANE.position, b.PLANE.position)
        exact(a.RNG.bit_generator.state, b.RNG.bit_generator.state)
        exact(pygame.surfarray.array3d(a.window), pygame.surfarray.array3d(b.window))
        features = [c.tracked_extractor._features() for c in controllers]
        try:
            exact(*features)
        except AssertionError:
            for k, (left, right) in enumerate(zip(*features)):
                if left.tobytes() != right.tobytes():
                    indices = np.argwhere(left != right)
                    print('feature mismatch', frame, k, indices[:8].tolist(),
                          [(float(left[tuple(ix)]), float(right[tuple(ix)])) for ix in indices[:8]], flush=True)
            raise
        exact(controllers[0].agent._receding_pixel_guard._plans, controllers[1].agent._receding_pixel_guard._plans)
        if frame and frame % 1200 == 0:
            print(f'{frame}/3600 closed-loop decisions exactly equal', flush=True)
    def stats(values):
        return dict(mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values, 95)))
    result = dict(decisions=3601, physics_steps=14400, seed=20260925,
        path_metric_cases=24,
        actions_pixels_features_plans_world_rng_equal=True,
        decision_before=stats(timings[0]), decision_after=stats(timings[1]),
        four_physics_steps_before=stats(physics_times[0]), four_physics_steps_after=stats(physics_times[1]),
        scope='Paired dummy-display correctness and compute timing; visible FPS measured separately. Collision damage disabled to complete the trajectory.')
    (BASE/'closed_loop.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    pygame.quit()


if __name__ == '__main__':
    main()
