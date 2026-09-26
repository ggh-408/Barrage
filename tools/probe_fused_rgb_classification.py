"""Isolated one-pass foreground and sprite-color classification candidate."""
from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pixel_search_kernel import njit


@njit(cache=False, fastmath=False)
def classify_scan(image, lower, upper):
    height, width = image.shape[:2]
    bullets = np.zeros((height, width), np.bool_)
    # Retain row-major ordering. Only the populated prefix is returned, owned.
    plane = np.empty((height * width, 2), np.int64)
    count = 0
    for y in range(height):
        for x in range(width):
            foreground = False
            for channel in range(3):
                value = image[y, x, channel]
                if value <= lower[channel] or value >= upper[channel]:
                    foreground = True
            if foreground:
                red = int(image[y, x, 0])
                green = int(image[y, x, 1])
                blue = int(image[y, x, 2])
                if max(red, green) - min(red, green) <= 16 and max(red, blue) - min(red, blue) <= 16:
                    bullets[y, x] = True
                else:
                    plane[count, 0] = y
                    plane[count, 1] = x
                    count += 1
    return bullets, plane[:count].copy()


def classify(image, background, threshold):
    from barrage_rl.live_screen import DominantBackgroundSemanticizer as Detector
    lower = np.full(3, -1, np.int64)
    upper = np.full(3, 256, np.int64)
    for channel in range(3):
        lo = background[channel] - threshold
        hi = background[channel] + threshold
        if lo >= 255 or hi <= 0:
            return Detector._classify_foreground(image, np.ones(image.shape[:2], np.bool_))
        if lo >= 0:
            lower[channel] = int(np.floor(lo))
        if hi <= 255:
            upper[channel] = int(np.ceil(hi))
    return classify_scan(image, lower, upper)


def main():
    from barrage_rl.live_screen import DominantBackgroundSemanticizer as Detector
    from tools.benchmark_tracker_exact import exact
    rng = np.random.default_rng(92543)
    cases = 0
    for shape in ((1, 1, 3), (32, 48, 3), (64, 65, 4)):
        image = rng.integers(0, 256, shape, dtype=np.uint8)
        image[:1, :1, :3] = [128, 144, 112]
        for frame in (image, image[:, ::-1], image.transpose(1, 0, 2)):
            for background in (np.zeros(3, np.int16), np.full(3, 128, np.int16), np.full(3, 255, np.int16)):
                for threshold in (0., 16., 24.5, 256., float('nan'), float('inf'), -1.):
                    before = frame.tobytes()
                    expected = Detector._classify_foreground(frame,
                        Detector._foreground_mask(frame, background, threshold))
                    actual = classify(frame, background, threshold)
                    exact(expected, actual)
                    assert frame.tobytes() == before
                    cases += 1
    output = ROOT / 'diagnostics/rgb_shared_scan_20260925'
    output.mkdir(exist_ok=True)
    result = dict(cases=cases, masks_and_ordered_plane_pixels_bitwise_equal=True,
                  inputs_unchanged=True, deployment_modified=False,
                  limitation='Boundary-level verification only; realistic timing and complete detector replay pending.')
    (output / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
