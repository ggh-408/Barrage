"""Physical duration stays consistent across action-repeat schedules."""
from unittest.mock import patch

import pytest

from barrage_rl.env import BarrageVisionEnv
from barrage_rl.task_spec import TARGET_TASK
from barrage_rl.timing import PHYSICS_FPS
from barrage_rl.tracked_policy import TrackedFeatureExtractor, TrackedPolicySpec


@pytest.mark.parametrize('repeats', [(4,), (3,), (5,), (3, 4, 5)])
def test_episode_stops_at_physical_limit(repeats):
    limit_steps = PHYSICS_FPS + 1
    limit_seconds = limit_steps / PHYSICS_FPS
    env = BarrageVisionEnv(**TARGET_TASK.env_kwargs(episode_limit_seconds=limit_seconds, observation_size=96))
    try:
        env.reset(seed=991000001)
        with patch.object(env, '_has_collision', return_value=False):
            for decision in range(100):
                env.action_repeat = repeats[decision % len(repeats)]
                _, _, terminated, truncated, info = env.step(0)
                assert not terminated
                if truncated:
                    break
            else:
                pytest.fail('episode did not terminate at the physical time limit')
        assert env.physics_steps == limit_steps
        assert info['survival_seconds'] == limit_seconds
        assert env.decision_dt == pytest.approx(env.action_repeat / PHYSICS_FPS)
    finally:
        env.close()


def test_production_horizon_and_tracker_share_timebase():
    env = BarrageVisionEnv(**TARGET_TASK.env_kwargs())
    try:
        tracker = TrackedFeatureExtractor(TrackedPolicySpec()).tracker
        assert env.physics_fps == TARGET_TASK.manifest()['physics_fps'] == 120
        assert env.max_episode_steps == 3600
        assert env.max_episode_physics_steps == 14400
        assert env.decision_dt == tracker.decision_dt == 1.0 / 30.0
    finally:
        env.close()
