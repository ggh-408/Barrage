import numpy as np
from tools.commit_safe_ranking import rank_routes


def choose(prefix_steps, exposure, nominal=None, lengths=None, interval=None):
    n = len(prefix_steps)
    lengths = np.full(n, 15) if lengths is None else np.asarray(lengths)
    metrics = np.ones((n, 7))
    metrics[:, 3] = np.asarray(prefix_steps) / (lengths * 4)
    metrics[:, 5] = metrics[:, 3] if interval is None else np.asarray(interval)/(lengths*4)
    metrics[:, 6] = exposure
    if nominal is not None:
        metrics[:, 3] = nominal
    return rank_routes(metrics, lengths, np.arange(n), np.full(n, 100.),
                       48., False, 0, -1)


def test_immediate_safety_beats_low_exposure():
    ranking, eligible = choose([0, 45], [.449, .570])
    assert ranking == [1]
    assert eligible.tolist() == [False, True]


def test_four_steps_are_required_and_sufficient():
    ranking, eligible = choose([3, 4], [0., 9.])
    assert ranking == [1]
    assert eligible.tolist() == [False, True]


def test_emergency_prefers_nominal_survival_time():
    ranking, _ = choose([0, 3], [0., 9.], interval=[60, 0])
    assert ranking[0] == 1


def test_full_nominal_route_beats_short_nominal_route():
    ranking, _ = choose([60, 59], [9., 0.])
    assert ranking[0] == 0


def test_safe_candidates_retain_nominal_then_exposure_priority():
    ranking, _ = choose([4, 30, 40], [2., 3., 0.], nominal=[1., 1., .9])
    assert ranking == [0, 1, 2]


def test_emergency_compares_steps_and_ties_are_deterministic():
    ranking, _ = choose([2, 3, 3], [0., 1., 1.], lengths=[10, 20, 20])
    assert ranking == [1, 2, 0]


def test_interval_rejection_does_not_exclude_nominally_safe_route():
    # Recorded 115.6s decision: left-down has 60 nominal steps but 2 interval
    # steps; right-up has 7 nominal steps and 5 interval steps.
    ranking, eligible = choose([60, 7], [.1, 2.6631066], interval=[2, 5])
    assert eligible.tolist() == [True, True]
    assert ranking[0] == 0


def test_interval_certificate_cannot_override_lower_exposure_full_nominal_route():
    ranking, _ = choose([60, 60], [.1, 2.], interval=[2, 60])
    assert ranking[0] == 0
