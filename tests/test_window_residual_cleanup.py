"""Exact specialized commitment checks and bounded window state."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def test_selected_commitment_matches_all_directions_with_young_and_known_tracks():
    from barrage_rl.window_pixel_geometry import WindowPixelGeometry
    from tools.pixel_guard_refined import RefinedPixelGuard
    from tools.pixel_guard_candidate import PixelGuardConfig
    geometry = WindowPixelGeometry()
    original = object.__new__(RefinedPixelGuard)
    original.__dict__.update(vars(geometry), config=PixelGuardConfig(physics_steps=4))
    rng = np.random.default_rng(540)
    for case in range(40):
        objects = np.zeros((3, 384, 16), np.float32)
        masks = rng.random((3, 384)) < .8
        globals_ = np.zeros((3, 16), np.float32)
        globals_[:, :2] = rng.uniform(.01, .99, (3, 2))
        globals_[:, 6:8] = rng.uniform(-1, 1, (3, 2))
        objects[:, :, :2] = rng.uniform(-.1, .1, (3, 384, 2))
        objects[:, :, 2:4] = rng.uniform(-1, 1, (3, 384, 2))
        objects[:, :, 8] = rng.choice([.14, .15, .49, .5, 1.], (3, 384))
        objects[:, :, 9] = rng.choice([0., 0., 1.], (3, 384))
        objects[:, :, 10] = rng.uniform(0, 1, (3, 384))
        objects[:, :, 15] = rng.choice([0., .5, 1.], (3, 384))
        expected = original.hazards(objects, masks, globals_)
        actual = geometry.hazards(objects, masks, globals_)
        for a, b in zip(expected, actual):
            assert a.tobytes() == b.tobytes()
        actions = np.asarray([(case+i) % 9 for i in range(3)])
        nominal, possible = geometry.hazards(objects, masks, globals_, actions=actions, include_nominal=False)
        assert nominal is None
        assert np.array_equal(possible[:, 0], expected[1][np.arange(3), actions])


def test_bounded_history_preserves_features_and_velocities():
    from barrage_rl.tracked_policy import TrackedFeatureExtractor
    from tools.benchmark_tracker_exact import exact
    rng = np.random.default_rng(323)
    old = TrackedFeatureExtractor()
    new = copy.deepcopy(old)
    new.tracker.history_limit = 8
    initial = rng.uniform(.1, .9, (300, 2)).astype(np.float32)
    velocity = rng.uniform(-.001, .001, initial.shape).astype(np.float32)
    for frame in range(180):
        points = initial + velocity*frame
        if frame % 11 == 0:
            points = points[20:]
        plane = None if frame % 17 == 0 else np.array([.5, .5], np.float32)
        exact(old.step_detections(points, plane, decision_steps=1+frame%2),
              new.step_detections(points, plane, decision_steps=1+frame%2))
        assert max(len(t.history) for t in new.tracker.tracks) <= 8
    assert max(len(t.history) for t in old.tracker.tracks) > 8


def test_meta_loading_preserves_every_checkpoint_tensor_and_output():
    from barrage_rl.checkpoint_loader import load_tracked_agent
    from barrage_rl.window_runtime import CONTROLLER
    root = Path(__file__).resolve().parents[1]
    path = root/'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agents = [load_tracked_agent(str(path), torch.device('cpu'), checkpoint_data=checkpoint,
            experimental_controller=CONTROLLER, materialize_state=flag)[0] for flag in (False, True)]
        for name, value in agents[0].model.state_dict().items():
            assert torch.equal(value, agents[1].model.state_dict()[name]), name
        inputs = (torch.zeros(1, 384, 16), torch.zeros(1, 384, dtype=torch.bool), torch.zeros(1, 16))
        with torch.inference_mode():
            for a, b in zip(agents[0].model(*inputs), agents[1].model(*inputs)):
                assert torch.equal(a, b)
    finally:
        torch.set_num_threads(previous)


def test_window_import_does_not_require_simulator_or_teacher_modules():
    source = "import sys,json; import barrage_rl.window_runtime,barrage_rl.live_screen; print(json.dumps([n for n in ['gymnasium','barrage_rl.env','barrage_rl.baselines','barrage_rl.recovery_planner','tools.pixel_guard_receding'] if n in sys.modules]))"
    result = subprocess.run([sys.executable, '-B', '-c', source], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_window_pauses_decisions_after_death_and_resumes_after_restart():
    source = '''
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
import json
import numpy as np
from Barrage import Barrage as game
class Controller:
    decision_interval=4
    action_delay_steps=0
    primes=0
    decisions=0
    def reset(self): pass
    def prime_surface(self, surface):
        self.primes+=1
        return 0
    def observe_due_surface(self, surface):
        assert game.KEY, 'Decision executed after death'
        self.decisions+=1
        return 0
c=Controller()
game.AI_CONTROLLER=c
game.SCREEN_WIDTH=game.SCREEN_HEIGHT=820
game.MUSIC=False
game.INVINCIBLE=False
game.KEY=True
game.LATENCY_TEST_SECONDS=.30
game.LATENCY_OUTPUT=None
advance=game.advance_physics
def step(direction, dt):
    advance(direction, dt)
    if game.ALIVE_PHYSICS_STEPS>=5: game.KEY=False
game.advance_physics=staticmethod(step)
frames=0
def events():
    global frames
    frames+=1
    restarted=frames==12
    if restarted:
        game.KEY=True
        game.reset_game()
    return False,np.zeros(2,np.float32),restarted
game.get_event=staticmethod(events)
game.start_game()
assert c.primes==2 and c.decisions==2, (c.primes,c.decisions)
print('death_restart_passed')
'''
    result = subprocess.run([sys.executable, '-B', '-c', source], capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith('death_restart_passed')
