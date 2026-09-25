"""Shared commitment-safe arbitration for the current image-only pixel planner."""
import numpy as np


def rank_routes(metrics, lengths, roots, first_walls, wall_reserve,
                wall_gate, incumbent, prior_index):
    """Require four nominally safe physics steps whenever such a route exists.

    Prefix metrics are fractions of each route's length. In the emergency
    branch compare physical step counts, including nominally unsafe routes.
    Uncertainty is a soft ranking signal, never an eligibility requirement.
    Exposure is only a sensitivity score, never a probability.
    """
    prefix_steps = metrics[:, 3] * np.asarray(lengths) * 4
    eligible = prefix_steps >= 4 - 1e-12
    emergency = not eligible.any()
    if emergency:
        eligible = np.ones(len(metrics), dtype=bool)

    def key(i):
        return (
            -prefix_steps[i] if emergency else 0.,
            -int(metrics[i, 3] >= 1.),
            metrics[i, 6], -prefix_steps[i], -metrics[i, 2],
            -metrics[i, 1], -metrics[i, 3],
            -min(metrics[i, 4], wall_reserve),
            -float(first_walls[i]) if wall_gate else 0.,
            int(i != 9 + incumbent), int(prior_index < 0 or i < prior_index),
            int(roots[i]),
        )

    return sorted(np.flatnonzero(eligible), key=key), eligible
