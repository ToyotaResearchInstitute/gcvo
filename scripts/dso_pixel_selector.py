"""DSO PixelSelector port (level-0 only; matches RKHS_BA `dso_select_pixels`).

Ported from `~/code/docker_home/cvo/gcvo/rkhs_ba/src/utils/CvoPixelSelector.cpp`.
The C++ implementation has commented-out level-1 / level-2 branches; only the
level-0 (`bestIdx2`) branch is active, so we only port that.

Algorithm:
  1. Compute gradient magnitude squared via central differences (mapmax0).
  2. For each 32×32 image block, build a 50-bin histogram of clipped sqrt(grad²)
     and pick the 50%-quantile + 7 as the block threshold.
  3. Smooth thresholds with a 3×3 (self-included) neighborhood mean, then square.
  4. Iterate over pot×pot cells; within each cell select the pixel with the
     largest grad² that exceeds `thsSmoothed * thFactor`. Yields one point per
     cell.
  5. Outer loop adjusts `pot` (potential) so the total point count is in
     [2/3 * num_want, num_want].

We expose `dso_select_pixels(gray, num_want)` returning a list of (x, y) coords.
"""
import numpy as np

SETTING_MIN_GRAD_HIST_CUT = 0.5
SETTING_MIN_GRAD_HIST_ADD = 7


def _make_thresholds(grad_sq, w32, h32, w, h):
    """Build per-32x32-block adaptive threshold map (smoothed, squared)."""
    ths = np.zeros((h32, w32), dtype=np.float32)
    grad_mag_clipped = np.minimum(np.sqrt(grad_sq), 48).astype(np.int32)

    for y in range(h32):
        for x in range(w32):
            it0, it1 = max(1, 32 * x), min(w - 1, 32 * x + 32)
            jt0, jt1 = max(1, 32 * y), min(h - 1, 32 * y + 32)
            if it1 <= it0 or jt1 <= jt0:
                continue
            block = grad_mag_clipped[jt0:jt1, it0:it1]
            hist = np.bincount(block.ravel(), minlength=49)  # bins 0..48
            total = int(block.size)
            # DSO stores hist[0] = total and shifts bins by +1; reproduce:
            th_count = int(total * SETTING_MIN_GRAD_HIST_CUT + 0.5)
            cum = 0
            q_idx = 90  # default in DSO if not reached
            for i in range(90):
                if i < 49:
                    cum += hist[i]
                if th_count - cum < 0:
                    q_idx = i
                    break
            ths[y, x] = q_idx + SETTING_MIN_GRAD_HIST_ADD

    # 3x3 neighborhood mean (self-included), then square.
    ths_smoothed = np.zeros_like(ths)
    for y in range(h32):
        for x in range(w32):
            s, n = 0.0, 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h32 and 0 <= xx < w32:
                        s += ths[yy, xx]
                        n += 1
            s += ths[y, x]
            n += 1
            avg = s / n
            ths_smoothed[y, x] = avg * avg
    return ths_smoothed


def _select(grad_sq, ths_smoothed, pot, th_factor=1.0):
    """Pick at most one point per pot×pot cell where grad_sq > smoothed_th×th_factor.

    Vectorized — equivalent to the innermost-`bestIdx2` branch of DSO's `select`.
    """
    h, w = grad_sq.shape
    # Tile per-block threshold to full image size, padding any unaligned
    # trailing rows/cols with the last block's threshold (edge-replicate).
    th_full = np.repeat(np.repeat(ths_smoothed, 32, axis=0), 32, axis=1)
    if th_full.shape[0] < h:
        th_full = np.concatenate(
            [th_full, np.broadcast_to(th_full[-1:, :], (h - th_full.shape[0], th_full.shape[1]))],
            axis=0)
    if th_full.shape[1] < w:
        th_full = np.concatenate(
            [th_full, np.broadcast_to(th_full[:, -1:], (th_full.shape[0], w - th_full.shape[1]))],
            axis=1)
    th_full = th_full[:h, :w] * th_factor

    # Border mask: xf in [4, w-5], yf in [4, h-4] (matches C++ check).
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    border = (xx >= 4) & (xx < w - 5) & (yy >= 4) & (yy <= h - 4)
    above = (grad_sq > th_full) & border

    # For each pot×pot cell, find argmax of grad_sq where above-threshold.
    # Iterate over cells (cheap enough since pot >= 1 and total cells ≤ h*w).
    coords = []
    masked = np.where(above, grad_sq, -1.0)
    for y0 in range(0, h, pot):
        for x0 in range(0, w, pot):
            cell = masked[y0:y0 + pot, x0:x0 + pot]
            if cell.size == 0:
                continue
            best = cell.max()
            if best < 0:
                continue
            idx = int(np.argmax(cell))
            ch, cw = cell.shape
            dy, dx = divmod(idx, cw)
            coords.append((x0 + dx, y0 + dy))
    return coords


def dso_select_pixels(image_gray, num_want):
    """Return list of (x, y) pixel coords selected by the DSO procedure.

    image_gray: HxW uint8 grayscale image.
    num_want: target point count.
    """
    img = image_gray.astype(np.float32)
    h, w = img.shape
    # Central difference gradient (matches RawImage::gradient_square()).
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    grad_sq = gx * gx + gy * gy

    w32 = w // 32
    h32 = h // 32
    ths_smoothed = _make_thresholds(grad_sq, w32, h32, w, h)

    # Outer auto-tuning loop on `potential` (matches dso_select_pixels in C++).
    pot = 3
    coords = _select(grad_sq, ths_smoothed, pot)

    # Too many: increase pot up to +4
    times = 1
    while len(coords) > num_want and times < 5:
        pot_try = 3 + times
        coords = _select(grad_sq, ths_smoothed, pot_try)
        times += 1

    # Too few (< 2/3 * num_want): decrease pot down to 1
    times = 1
    while len(coords) < (num_want * 2) // 3 and times > -2:
        pot_try = max(1, 1 + times)
        coords = _select(grad_sq, ths_smoothed, pot_try)
        times -= 1

    return coords
