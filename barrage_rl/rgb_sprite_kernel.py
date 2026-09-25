"""Optional exact integer sprite kernels; preserve candidate and tie order."""
import numpy as np

try:
    from tools.pixel_search_kernel import njit
except (ImportError, RuntimeError):
    validate_candidates = erase_coverage = select_cover = classify_sparse = None
else:
    @njit(cache=False, fastmath=False)
    def classify_sparse(image, indices):
        bullets = np.zeros(image.shape[:2], np.bool_)
        planes = np.empty((len(indices), 2), np.int64)
        count = 0
        for i in indices:
            y, x = i // image.shape[1], i % image.shape[1]
            r, g, b = np.int64(image[y, x, 0]), np.int64(image[y, x, 1]), np.int64(image[y, x, 2])
            if abs(r-g) <= 16 and abs(r-b) <= 16:
                bullets[y, x] = True
            else:
                planes[count, 0], planes[count, 1] = y, x
                count += 1
        return bullets, planes[:count]

    @njit(cache=False, fastmath=False)
    def validate_candidates(mask, candidates):
        height, width = mask.shape
        output = np.empty((len(candidates), 2), np.int64)
        count = 0
        for i in range(len(candidates)):
            y, x = candidates[i, 0], candidates[i, 1]
            visible = 0
            matched = True
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dx) == 2 and abs(dy) == 2:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        visible += 1
                        if not mask[yy, xx]:
                            matched = False
                            break
                if not matched:
                    break
            if matched and visible >= 8:
                output[count, 0], output[count, 1] = y, x
                count += 1
        return output[:count]

    @njit(cache=False, fastmath=False)
    def erase_coverage(residual, centers):
        height, width = residual.shape
        for i in range(len(centers)):
            y, x = centers[i, 0], centers[i, 1]
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dx) == 2 and abs(dy) == 2:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        residual[yy, xx] = False

    @njit(cache=False, fastmath=False)
    def select_cover(residual, candidates):
        height, width = residual.shape
        count = len(candidates)
        coverage = np.full((count, 21), -1, np.int64)
        for i in range(count):
            offset = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dx) == 2 and abs(dy) == 2:
                        continue
                    y, x = candidates[i, 0]+dy, candidates[i, 1]+dx
                    if 0 <= y < height and 0 <= x < width:
                        coverage[i, offset] = y*width+x
                    offset += 1
        uncovered = residual.flatten()
        selected = np.empty((count, 2), np.float32)
        selected_count = 0
        while True:
            best, best_gain = -1, 1
            # Strict > retains the first maximum, exactly as np.argmax.
            for i in range(count):
                gain = 0
                for offset in range(21):
                    pixel = coverage[i, offset]
                    if pixel >= 0 and uncovered[pixel]:
                        gain += 1
                if gain > best_gain:
                    best, best_gain = i, gain
            if best < 0:
                break
            selected[selected_count, 0] = candidates[best, 1]
            selected[selected_count, 1] = candidates[best, 0]
            selected_count += 1
            for offset in range(21):
                pixel = coverage[best, offset]
                if pixel >= 0:
                    uncovered[pixel] = False
        return selected[:selected_count]


def warmup_sprite_kernels():
    """Compile all live C-layout signatures before the first timed decision."""
    if validate_candidates is None:
        return
    mask = np.ones((8, 8), np.bool_)
    centers = np.asarray([[4, 4]], np.int64)
    validate_candidates(mask, centers)
    select_cover(mask, centers)
    erase_coverage(mask, centers)
