from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import test_visible_window as diagnostic


def fake_pygame():
    return SimpleNamespace(display=Mock(), font=Mock(), event=Mock(),
                           QUIT=256, quit=Mock())


def test_already_foreground_starts_without_waiting():
    pygame = fake_pygame()
    assert diagnostic.wait_for_test_foreground(pygame, lambda: True)
    pygame.display.flip.assert_not_called()


def test_activation_denied_waits_for_focus_without_quitting(monkeypatch):
    pygame = fake_pygame()
    pygame.event.get.return_value = []
    sleep = Mock()
    monkeypatch.setattr(diagnostic.time, 'sleep', sleep)
    assert diagnostic.wait_for_test_foreground(pygame, Mock(side_effect=[False, False, True]))
    pygame.display.flip.assert_called_once()
    pygame.event.get.assert_called_once()
    sleep.assert_called_once_with(.02)
    pygame.quit.assert_not_called()


def test_waiting_window_can_be_closed():
    pygame = fake_pygame()
    pygame.event.get.return_value = [SimpleNamespace(type=pygame.QUIT)]
    with pytest.raises(SystemExit) as error:
        diagnostic.wait_for_test_foreground(pygame, lambda: False)
    assert error.value.code == 0
    pygame.quit.assert_called_once()
