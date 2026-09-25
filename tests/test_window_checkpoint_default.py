from Barrage import DEFAULT_AI_CHECKPOINT, PROJECT_ROOT


def test_window_checkpoint_is_pinned():
    assert DEFAULT_AI_CHECKPOINT == PROJECT_ROOT / "best.pt"


def test_dashboard_matches_all_tie_selection():
    from barrage_rl.plot import _dagger_best_index
    data = dict(round=[2, 0, 1], model_iqm=[120.] * 3,
                model_p1=[120.] * 3, success_at_limit=[1.] * 3)
    assert _dagger_best_index(data, 'success_at_limit', 120.) == 0
    data['success_at_limit'] = [.995] * 3
    assert _dagger_best_index(data, 'success_at_limit', 120.) == 0
