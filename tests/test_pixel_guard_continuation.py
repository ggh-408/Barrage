import numpy as np
import pytest
from tools.pixel_guard_receding import RecedingGuardConfig,RecedingPixelGuard
from tests.test_pixel_guard_receding import features,selection


@pytest.fixture
def guard():
    return RecedingPixelGuard(RecedingGuardConfig(search_depth=3,beam_width=2,search_workers=1))


def test_retained_suffix_survives_batch_reordering_and_targeted_reset(guard):
    o,m,g=features(guard,np.array([410,410],np.float32),np.array([[380,410]],np.float32),np.zeros((1,2),np.float32))
    guard._plans[11]=dict(path=np.array([2,3]),expected_plane=np.array([410,410]))
    guard._plans[22]=dict(path=np.array([3,2]),expected_plane=np.array([410,410]))
    for key,expected in ((22,3),(11,2)):
        guard._context_indices=np.array([key])
        result=guard.apply(selection(1),o[None],m[None],g[None])
        assert result.actions.item()==expected
    guard.reset([11])
    assert 11 not in guard._plans and 22 in guard._plans
    guard.reset()
    assert not guard._plans


@pytest.mark.parametrize('skipped,moved',[(2,False),(1,True)])
def test_stale_plan_is_discarded_after_skipped_frame_or_wrong_motion(guard,skipped,moved):
    o,m,g=features(guard,np.array([410,410],np.float32),np.empty((0,2)),np.empty((0,2)))
    guard._context_steps=skipped
    guard._plans[0]=dict(path=np.array([2,3]),expected_plane=np.array([400 if moved else 410,410]))
    result=guard.apply(selection(0),o[None],m[None],g[None])
    assert result.actions.item()==0
    assert not guard._plans
    assert guard.counters['plan_invalidations']==1


def test_reused_prefix_keeps_full_lookahead_and_bounded_commitment(guard):
    o,m,g=features(guard,np.array([410,410],np.float32),np.array([[380,410]],np.float32),np.zeros((1,2),np.float32))
    guard._plans[0]=dict(path=np.array([2,3]),expected_plane=np.array([410,410]))
    result=guard.apply(selection(1),o[None],m[None],g[None])
    assert result.actions.item()==2
    assert len(guard._plans[0]['path'])==guard.recovery_config.search_depth-1
    assert guard._plans[0]['remaining']==1


def test_certified_model_direction_releases_unnecessary_plan(guard):
    o,m,g=features(guard,np.array([410,410],np.float32),np.empty((0,2)),np.empty((0,2)))
    guard._plans[0]=dict(path=np.array([2,3]),expected_plane=np.array([410,410]))
    result=guard.apply(selection(0),o[None],m[None],g[None])
    assert result.actions.item()==0
    assert not guard._plans
    assert guard.counters['model_releases']==1


def test_future_collision_triggers_search_before_short_gate():
    guard=RecedingPixelGuard(RecedingGuardConfig(search_depth=18,beam_width=2,search_workers=1))
    o,m,g=features(guard,np.array([410,410],np.float32),np.array([[500,410]],np.float32),np.zeros((1,2),np.float32))
    _,possible=guard.hazards(o[None],m[None],g[None])
    assert not possible[0].any()
    guard.apply(selection(2),o[None],m[None],g[None])
    assert guard.counters['searches']==1
    assert guard.counters['horizon_searches']==1


def test_coherent_scenarios_prefer_an_executable_escape(guard):
    from tools.pixel_receding_kernel import assess_paths
    from barrage_rl.runtime_core import ACTION_VECTORS
    plane=np.array([410,410],np.float32)
    bullets=np.array([[440,410]],np.float32)
    velocity=np.array([[-240,0]],np.float32)
    paths=np.array([[0]*9,[1]*9])
    metrics=assess_paths(plane,guard.half_size,bullets,velocity,np.array([2],np.float32),
        guard.table,ACTION_VECTORS,paths,np.array([9,9]))
    assert metrics[1,2]==9
    assert metrics[0,2]==0
    assert metrics[1,1]>metrics[0,1]


def test_full_interval_catches_directions_between_nine_samples(guard):
    from tools.pixel_receding_kernel import assess_paths,assess_path_intervals
    from barrage_rl.runtime_core import ACTION_VECTORS
    # A newly detected bullet has an unknown velocity within the component
    # bounds. Sampling only zero and extreme components misses interior rays.
    plane=np.array([645,114],np.float32)
    bullets=np.array([[692,8]],np.float32)
    velocity=np.zeros((1,2),np.float32)
    error=np.array([240],np.float32)
    path=np.array([[4,5,3,6,2,2,2,2,3,8,1,7,4,4,4,4,4,4]])
    lengths=np.array([18])
    scenarios=assess_paths(plane,guard.half_size,bullets,velocity,error,guard.table,
                            ACTION_VECTORS,path,lengths)
    interval=assess_path_intervals(plane,guard.half_size,bullets,velocity,error,guard.table,
                                    guard.integral,ACTION_VECTORS,path,lengths)
    assert scenarios[0,2]==9
    assert interval[0]<1


def test_retained_plan_must_preserve_terminal_wall_reserve(guard):
    o,m,g=features(guard,np.array([410,55],np.float32),np.empty((0,2)),np.empty((0,2)))
    guard._plans[0]=dict(path=np.array([3,3,3]),expected_plane=np.array([410,55]))
    result=guard.apply(selection(0),o[None],m[None],g[None])
    assert result.actions.item()!=3


def test_path_tracking_preserves_existing_beam_costs(guard):
    from tools.pixel_receding_kernel import beam_paths,beam_costs
    from barrage_rl.runtime_core import ACTION_VECTORS
    plane=np.array([410,410],np.float32)
    bullets=np.array([[438,419],[383,395]],np.float32)
    velocities=np.array([[-220,-30],[160,110]],np.float32)
    args=(plane,guard.half_size,bullets,velocities,np.array([2,4],np.float32),
          guard.table,guard.integral,ACTION_VECTORS,3,2,48.,1.)
    costs,paths=beam_paths(*args)
    np.testing.assert_array_equal(costs,beam_costs(*args))
    np.testing.assert_array_equal(paths[:,0],np.arange(9))
    assert np.all((paths>=0)&(paths<9))


def test_public_calls_propagate_episode_identity_and_reset():
    from tools.pixel_guard_receding import install_receding_guard
    class Agent:
        safety_threshold=.18
        def _select_actions(self,*args,**kwargs): return selection(0)
        def act_features(self,o,m,g,deterministic=True,episode_indices=None,**kwargs):
            return self._receding_pixel_guard._context_indices,self._receding_pixel_guard._context_steps
        def act_features_with_diagnostics(self,o,m,g,episode_indices=None):
            return self._receding_pixel_guard._context_indices
        def reset_state(self,episode_indices=None): self.reset_indices=episode_indices
    agent=Agent()
    guard=install_receding_guard(agent,RecedingGuardConfig(search_depth=2,beam_width=2,search_workers=1))
    ids=np.array([19])
    actual,steps=agent.act_features(None,None,None,True,ids,decision_steps=2)
    np.testing.assert_array_equal(actual,ids)
    assert steps==2
    np.testing.assert_array_equal(agent.act_features_with_diagnostics(None,None,None,ids),ids)
    assert guard._context_indices is None and guard._context_steps==1
    guard._plans[19]={}
    guard._plans[20]={}
    agent.reset_state(ids)
    assert 19 not in guard._plans and 20 in guard._plans
    np.testing.assert_array_equal(agent.reset_indices,ids)
