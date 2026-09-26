"""Owned RGB snapshot from a temporary SDL view, preserving channel values."""
import numpy as np

try:
    from tools.pixel_search_kernel import njit
except (ImportError, RuntimeError):
    copy_rgb = None
else:
    from numba import prange
    from .rgb_parallel import dispatch

    @njit(cache=False, fastmath=False)
    def copy_rgb_serial(image):
        result = np.empty((image.shape[0], image.shape[1], 3), np.uint8)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(3):
                    result[y, x, c] = image[y, x, c]
        return result

    @njit(cache=False, fastmath=False, parallel=True)
    def copy_rgb_parallel(image):
        result = np.empty((image.shape[0], image.shape[1], 3), np.uint8)
        for y in prange(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(3):
                    result[y, x, c] = image[y, x, c]
        return result

    def copy_rgb(image):
        return dispatch(copy_rgb_serial, copy_rgb_parallel, image)
