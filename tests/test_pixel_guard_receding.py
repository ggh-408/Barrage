import numpy as np
import pytest
import torch
from types import SimpleNamespace

from barrage_rl.action_selector import ActionSelection
from barrage_rl.runtime_core import ACTION_VECTORS, advance_plane, colliding_bullet_indices
from tools.pixel_guard_receding import RecedingGuardConfig, RecedingPixelGuard, install_receding_guard


@pytest.fixture(scope='module')
def guard():
    return RecedingPixelGuard(RecedingGuardConfig(search_depth=6,beam_width=4))


def features(guard, plane, bullets, velocities):
    observed = plane+guard.centroid_bias
    o = np.zeros((len(bullets),16),np.float32)
    o[:,:2]=(bullets-observed)/820
    o[:,2:4]=velocities/240
    o[:,8]=o[:,10]=o[:,15]=1
    g=np.zeros(16,np.float32)
    g[:2]=observed/820
    return o,np.ones(len(o),bool),g


def selection(action):
    return ActionSelection(actions=torch.tensor([action]),scores=torch.zeros(1,9),
        raw_actions=torch.tensor([action]),immediate_risk=torch.zeros(1,9),
        all_unsafe=torch.tensor([False]),analytic_clearance=None,
        learned_actions=torch.tensor([action]),collision_risk=torch.zeros(1,4,9),
        teacher_cost=torch.zeros(1,9),counter_values=torch.zeros(5,dtype=torch.int64))


def test_empty_interior_retains_incumbent_and_does_not_search(guard):
    o,m,g=features(guard,np.array([410,410],np.float32),np.empty((0,2)),np.empty((0,2)))
    before=guard.counters['searches']
    for action in range(9):
        result=guard.apply(selection(action),o[None],m[None],g[None])
        assert result.actions.item()==action
    assert guard.counters['searches']==before


def test_near_edge_search_moves_inward_without_dynamic_state(guard):
    plane=np.array([410,8],np.float32)
    o,m,g=features(guard,plane,np.empty((0,2)),np.empty((0,2)))
    result=guard.apply(selection(3),o[None],m[None],g[None])
    assert ACTION_VECTORS[result.actions.item(),1]>0


def test_recovery_preserves_a_safe_committed_prefix_against_real_masks(guard):
    rng=np.random.default_rng(851)
    plane=np.array([410,410],np.float32)
    cases=0
    for _ in range(25):
        bullets=plane+rng.uniform(-35,35,(4,2)).astype(np.float32)
        angles=rng.uniform(0,2*np.pi,4)
        velocities=np.column_stack((np.cos(angles),np.sin(angles))).astype(np.float32)*240
        o,m,g=features(guard,plane,bullets,velocities)
        o_before=o.copy()
        nominal,possible=guard.commit_guard.hazards(o[None],m[None],g[None])
        if possible[0].all():
            continue
        action=guard.search(o,m,g,np.zeros(9),0,possible[0],nominal[0])
        assert not possible[0,action]
        p=plane.copy()
        b=bullets.copy()
        for _ in range(4):
            p,_=advance_plane(p,ACTION_VECTORS[action],240,1/120,guard.half_size,820,820)
            b+=velocities/120
            assert not len(colliding_bullet_indices(p,b,guard.plane,guard.bullet,guard.plane_mask,guard.bullet_mask))
        np.testing.assert_array_equal(o,o_before)
        assert action==guard.search(o,m,g,np.zeros(9),0,possible[0],nominal[0])
        cases+=1
    assert cases>=10


@pytest.mark.parametrize('config',[
    RecedingGuardConfig(search_depth=0),RecedingGuardConfig(beam_width=0),
    RecedingGuardConfig(trigger_safe_actions=10),RecedingGuardConfig(wall_cost_weight=-1),
])
def test_invalid_search_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        RecedingPixelGuard(config)


def test_window_wrapper_and_game_entry_share_one_guard():
    calls=[]
    agent=SimpleNamespace(safety_threshold=.18,_select_actions=lambda *args,**kwargs:(calls.append(1),selection(0))[1])
    config=RecedingGuardConfig(search_depth=2,beam_width=2)
    first=install_receding_guard(agent,config)
    assert install_receding_guard(agent,config) is first
    o,m,g=features(first,np.array([410,410],np.float32),np.empty((0,2)),np.empty((0,2)))
    agent._select_actions(torch.tensor(o[None]),torch.tensor(m[None]),torch.tensor(g[None]),deterministic=True)
    assert calls==[1]
    assert first.counters['decisions']==1
    with pytest.raises(ValueError):
        install_receding_guard(agent,RecedingGuardConfig(search_depth=3))
