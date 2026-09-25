"""Run the existing no-AI immune diagnostic on the foreground desktop."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pygame
from tools.test_visible_window import foreground_test_window
from tools.measure_visible_latency import main

if __name__ == '__main__':
    original = pygame.display.set_mode
    def set_mode(*args, **kwargs):
        surface = original(*args, **kwargs)
        foreground_test_window(pygame)
        return surface
    pygame.display.set_mode = set_mode
    try:
        main()
    finally:
        pygame.display.set_mode = original
