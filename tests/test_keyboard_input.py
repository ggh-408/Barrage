"""Physical-key input remains independent of the active text input language."""

import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pygame

from Barrage import Barrage


class KeyboardInputTests(unittest.TestCase):
    def setUp(self) -> None:
        Barrage.AI_CONTROLLER = None
        Barrage.KEY = True
        Barrage.PRESSED_SCANCODES.clear()

    @staticmethod
    def _no_keys() -> defaultdict[int, bool]:
        return defaultdict(bool)

    def test_physical_w_scancode_moves_up_with_untranslated_key_value(self) -> None:
        down = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_UNKNOWN,
            scancode=pygame.KSCAN_W,
            unicode="",
            mod=0,
        )
        with patch("pygame.key.get_pressed", return_value=self._no_keys()), patch(
            "pygame.event.get", return_value=[down]
        ):
            _, direction, _ = Barrage.get_event()
        np.testing.assert_array_equal(direction, np.asarray([0.0, -1.0]))

        up = pygame.event.Event(
            pygame.KEYUP,
            key=pygame.K_UNKNOWN,
            scancode=pygame.KSCAN_W,
            unicode="",
            mod=0,
        )
        with patch("pygame.key.get_pressed", return_value=self._no_keys()), patch(
            "pygame.event.get", return_value=[up]
        ):
            _, direction, _ = Barrage.get_event()
        np.testing.assert_array_equal(direction, np.asarray([0.0, 0.0]))

    def test_focus_loss_clears_physical_key_state(self) -> None:
        Barrage.PRESSED_SCANCODES.add(pygame.KSCAN_D)
        focus_lost = pygame.event.Event(pygame.WINDOWFOCUSLOST)
        with patch("pygame.key.get_pressed", return_value=self._no_keys()), patch(
            "pygame.event.get", return_value=[focus_lost]
        ):
            _, direction, _ = Barrage.get_event()
        np.testing.assert_array_equal(direction, np.asarray([0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
