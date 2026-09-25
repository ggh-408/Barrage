"""Current-window contract: reject retired checkpoint and option inputs."""
from unittest.mock import patch

import pytest

from barrage_rl.live_screen import LiveVisualController


@pytest.mark.parametrize('version', [10, 11])
def test_window_rejects_retired_checkpoint_versions(version):
    with patch('barrage_rl.live_screen.torch.load', return_value={'model_version': version}):
        with pytest.raises(ValueError, match='model_version 12'):
            LiveVisualController('unused.pt')


def test_window_rejects_retired_inert_options_before_loading():
    with patch('barrage_rl.live_screen.torch.load') as load:
        with pytest.raises(TypeError):
            LiveVisualController('unused.pt', analytic_max_model_risk_increase=0.005)
        load.assert_not_called()
