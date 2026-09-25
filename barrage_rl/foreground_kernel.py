"""Optional compiled integer foreground scan; no fast-math transformations."""
import numpy as np

try:
    from tools.pixel_search_kernel import njit
except (ImportError, RuntimeError):
    fused_foreground = None
else:
    from numba import prange
    from .rgb_parallel import dispatch

    @njit(cache=False, fastmath=False)
    def fused_foreground_serial(image, lower, upper):
        result = np.empty(image.shape[:2], np.bool_)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                hit = False
                for channel in range(3):
                    value = image[y, x, channel]
                    if value <= lower[channel] or value >= upper[channel]:
                        hit = True
                result[y, x] = hit
        return result

    @njit(cache=False, fastmath=False, parallel=True)
    def fused_foreground_parallel(image, lower, upper):
        result = np.empty(image.shape[:2], np.bool_)
        for y in prange(image.shape[0]):
            for x in range(image.shape[1]):
                hit = False
                for channel in range(3):
                    value = image[y, x, channel]
                    if value <= lower[channel] or value >= upper[channel]:
                        hit = True
                result[y, x] = hit
        return result

    def fused_foreground(image, lower, upper):
        return dispatch(fused_foreground_serial, fused_foreground_parallel, image, lower, upper)
