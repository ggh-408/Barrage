from unittest.mock import Mock

import Barrage as window


def test_manual_timed_entry_uses_cli_settings_without_dialog(monkeypatch):
    start = Mock()
    dialog = Mock(side_effect=AssertionError('Unexpected settings dialog'))
    monkeypatch.setattr(window.Barrage, 'start_game', start)
    monkeypatch.setattr(window, 'settings', dialog)
    window.main(['--no-ai', '--no-music', '--bullets', '300',
                 '--seed', '20260925', '--latency-test-seconds', '1'])
    start.assert_called_once()
    assert window.Barrage.AI_CONTROLLER is None
    assert window.Barrage.QUANTITY == 300
    assert window.Barrage.TARGETED_BULLET_PROBABILITY == .10
    assert window.Barrage.INVINCIBLE is False
    window.main(['--no-ai', '--latency-test-seconds', '1', '--latency-allow-damage'])
    assert window.Barrage.INVINCIBLE is True


def test_no_mode_retains_settings_dialog(monkeypatch):
    start = Mock()
    monkeypatch.setattr(window.Barrage, 'start_game', start)
    monkeypatch.setattr(window, 'settings', Mock(return_value=False))
    window.main([])
    start.assert_not_called()
