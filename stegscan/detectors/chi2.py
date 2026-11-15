"""Chi-square attack (Westfeld & Pfitzmann 1999) with a quantitative extension.

Two things are computed on the value histogram h[0..255]:

1. ``p_uniform`` -- the classic attack.  Pairs of values (2k, 2k+1) are
   expected to have equal counts once every LSB carries an independent
   random message bit.  A chi-square goodness-of-fit test against that
   "flattened pairs" model gives a p-value near 1 for fully embedded
   images and near 0 for untouched photographs.

2. ``rate`` -- a *quantitative* estimate.  The observed difference
   d_i = h(2i) - h(2i+1) shrinks by exactly (1 - p) under LSB
   replacement, while the pair total S_i = h(2i) + h(2i+1) is unchanged.
   Fitting  d_i ~ beta * D_i  with D_i an estimate of the pair difference
   the cover would have shown gives beta = 1 - p.

   D_i is obtained from the smoothness of S:  for a locally smooth value
   distribution, h(v) - h(v+1) ~ -1/2 dh/dv, and the pair difference of
   the *cover* is approximately  -dS/dv evaluated at the pair, divided by
   two.  Since S is invariant, this predictor does not depend on p -- it
   can be computed from the stego image alone.  A central difference over
   a stride of w bins is used; the factor of 1/(4w) turns the count
   difference of a stride-2w interval into a per-unit-value slope.

Caveats (they matter for how the number is reported):

* the estimator is biased low at high rates and noisy at low rates, so
  the scanner applies a calibration curve fitted on reference images;
* the classic p-value saturates: even a clean photo rarely looks perfectly
  like the model, so the p-value alone is a weak signal for small payloads.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2

from .result import Result

# How many histogram bins must a pair cover before we trust its difference.
MIN_PAIR_COUNT = 20
# A pair must also hold at least this fraction of the image's pixels / 128,
# so that small images are not dominated by a handful of huge bins.
MIN_PAIR_FRACTION = 0.01
# Each spatial tile must be at least this large on a side before the
# tiled estimator may split the plane.  See analyse_plane().
MIN_TILE_EDGE = 256
# Minimum correlation between the observed pair differences and the
# smooth-envelope predictor before the estimate is trusted.  See the note
# at the gate in estimate_rate().
MIN_CORRELATION = 0.35
# ...and only enforced while the predictor still carries signal.  Below
# this ratio of mean |D| to mean pair total, the histogram has flattened
# (i.e. the payload is large) and correlation is not diagnostic.
MIN_PREDICTOR_STRENGTH = 0.01


def pair_histogram(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.bincount(plane.reshape(-1), minlength=256).astype(np.float64)
    return h[0::2].copy(), h[1::2].copy()


def _threshold(plane: np.ndarray, min_count: int) -> float:
    return max(float(min_count), MIN_PAIR_FRACTION * plane.size / 128.0)


def _trim_support(S: np.ndarray, usable: np.ndarray,
                  min_count: int, margin: int = 3) -> np.ndarray:
    """Drop pairs outside the histogram's real support, with a margin.

    A pair whose total is below the counting threshold carries no usable
    slope: its own "difference" is a few pixels against a few pixels.  The
    central-difference predictor at the edge of the populated range instead
    reads the drop to zero and reports a huge slope -- e.g. S going
    ... 5369, 1692, 0, 0 ... produces a predictor of several hundred where
    the data supports tens.  Those bins, and ``margin`` bins either side,
    are excluded.

    This is a *support* test, not a spike test: it keys on the threshold the
    caller is already using for every other bin, so it cannot discard the
    dense interior the regression depends on.
    """
    n = len(S)
    usable = usable.copy()
    core = np.flatnonzero(S >= min_count)
    if core.size < max(4, 2 * margin + 2):
        return np.zeros(n, dtype=bool)
    lo = max(0, core[0] - margin)
    hi = min(n, core[-1] + 1 + margin)
    usable[:lo] = False
    usable[hi:] = False
    return usable


def _trim_clipping(S: np.ndarray, usable: np.ndarray, run: int = 4,
                   spike_factor: float = 4.0) -> np.ndarray:
    """Drop end bins dominated by clipping or by an isolated spike.

    Saturated highlights and crushed shadows pile pixels onto the end bins,
    and a tone curve can leave a single enormous bin at the histogram edge.
    Either produces a near-vertical edge that a central-difference predictor
    reads as a huge slope.  Two robust rules are applied:

    * a bin whose count exceeds ``spike_factor`` times the median of the
      neighbouring bins on its inner side is treated as a spike;
    * the outermost ``run`` bins are never used, because the predictor at
      those positions is one-sided.

    This deliberately does *not* walk inward looking for a trend: a
    stair-stepped histogram (common in smooth synthetic content) would make
    such a walk consume the whole usable range, which is how an earlier
    version produced NaN for entire images.
    """
    n = len(S)
    usable = usable.copy()
    if n < 2 * run + 2:
        return usable

    for side in ("left", "right"):
        for offset in range(run):
            idx = offset if side == "left" else n - 1 - offset
            usable[idx] = False

        for idx in (range(run, n // 2) if side == "left"
                    else range(n - 1 - run, n // 2, -1)):
            if not usable[idx]:
                continue
            # neighbours further inside the histogram
            if side == "left":
                nb = S[idx + 1: idx + 1 + run]
            else:
                nb = S[idx - run: idx]
            nb = nb[nb > 0]
            if nb.size == 0:
                usable[idx] = False
                continue
            if S[idx] > spike_factor * np.median(nb):
                usable[idx] = False
    return usable


def chi_square_uniformity(plane: np.ndarray, min_count: int = MIN_PAIR_COUNT) -> tuple[float, int]:
    """Westfeld chi-square attack.  Returns (p_value, #pairs_used)."""
    h0, h1 = pair_histogram(plane)
    S = h0 + h1
    exp = S / 2.0
    thresh = _threshold(plane, min_count)
    usable = (S >= thresh) & (exp > 0)
    usable = _trim_support(S, usable, thresh)
    usable = _trim_clipping(S, usable)
    n = int(usable.sum())
    if n < 4:
        return float("nan"), n
    stat = (((h0 - exp) ** 2 + (h1 - exp) ** 2)[usable] / exp[usable]).sum()
    return float(_chi2.sf(stat, max(1, n - 1))), n


def estimate_rate(
    plane: np.ndarray,
    stride: int = 1,
    min_count: int = MIN_PAIR_COUNT,
) -> tuple[float, int, float]:
    """Least-squares estimate of the embedding rate.

    Returns (rate, #bins_used, beta) where beta = 1 - rate.
    """
    h0, h1 = pair_histogram(plane)
    S = h0 + h1
    d = h0 - h1
    n = len(S)

    # Predictor: central difference of S over a stride of `stride` pairs
    # (= 2*stride value bins each side).  A stride of w pair-indices spans
    # 2w value units each side, so the slope is (S[i+w]-S[i-w]) / (4w) in
    # "counts per unit value"; the cover pair difference is half of that.
    D = np.full(n, np.nan)
    if n <= 2 * stride:
        return float("nan"), 0, float("nan")
    D[stride:n - stride] = -(S[2 * stride:] - S[:n - 2 * stride]) / (8.0 * stride)

    thresh = _threshold(plane, min_count)
    usable = np.isfinite(D) & (S >= thresh) & (np.abs(D) > 1e-9)
    usable = _trim_support(S, usable, thresh)
    usable = _trim_clipping(S, usable)
    # A pair difference cannot exceed the pair total it came from.
    usable &= np.abs(D) <= np.maximum(S, 1.0)

    if usable.sum() < 6:
        return float("nan"), int(usable.sum()), float("nan")

    dd = d[usable]
    DD = D[usable]
    # Weight by 1/S: under the null, var(d) is approximately S, so a pair
    # with twice the count is twice as precise.  Without this weighting the
    # sparse tail bins (where the slope estimate is mostly noise) dominate.
    w = 1.0 / np.maximum(S[usable], 1.0)
    beta = float((w * dd * DD).sum() / (w * DD * DD).sum())

    # Model-validity gate.  The estimator assumes the cover's pair
    # differences follow the smooth envelope.  Content that breaks that --
    # a painting with dithering, a halftone scan, a posterised gradient --
    # has a jagged histogram where d and D are uncorrelated; the fitted
    # slope then collapses towards zero and the formula reports a *full*
    # payload.  Requiring positive correlation between d and D catches that.
    #
    # The gate is only applied when the *predictor* still carries signal:
    # at high embedding rates the LSB plane is randomised, the pair totals
    # flatten out, D -> 0, and d becomes pure noise, so correlation is
    # meaningless there (and low).  In that regime the slope is near zero
    # because the payload really is there, so let the gate pass.
    if DD.size >= 8:
        corr = float(np.corrcoef(dd, DD)[0, 1])
        predictor_strength = float(np.mean(np.abs(DD)) / max(np.mean(S[usable]), 1e-9))
        if (not np.isfinite(corr) or corr < MIN_CORRELATION) and \
                predictor_strength >= MIN_PREDICTOR_STRENGTH:
            return float("nan"), int(DD.size), float("nan")

    rate = float(np.clip(1.0 - beta, 0.0, 1.0))
    return rate, int(DD.size), beta


def analyse_plane(
    plane: np.ndarray,
    stride: int = 1,
    min_count: int = MIN_PAIR_COUNT,
    min_tile_edge: int = MIN_TILE_EDGE,
    max_tiles: int = 4,
) -> Result:
    """Run both tests on one plane.

    The uniformity p-value uses the whole plane.  The *rate* is estimated on
    a grid of spatial tiles and combined by median: the single-histogram
    regression is noisy at low payloads (its per-bin differences are
    counting noise), and a median over tiles is far more stable than one
    global fit while still being consistent -- an embedding tool spreads its
    payload across the image, so every tile carries roughly the same density.

    The grid is chosen so that each tile is at least ``min_tile_edge`` on a
    side.  Splitting a 512 px image into 4x4 gives 128 px tiles, which have
    too few samples per histogram bin for the slope fit to mean anything
    (measured: the estimate collapses toward zero even at a 25 % payload);
    the same image split into 2x2 is accurate.  On a 900 px photograph the
    tile size makes little difference, which is why the floor rather than a
    fixed grid is what matters.
    """
    p_uniform, n_pairs = chi_square_uniformity(plane, min_count=min_count)

    h, w = plane.shape
    t = min(max_tiles, h // min_tile_edge, w // min_tile_edge)
    t = max(1, t)
    if t == 1:
        rate, n_bins, beta = estimate_rate(plane, stride=stride,
                                           min_count=min_count)
    else:
        th, tw = h // t, w // t
        rates, n_bins_total = [], 0
        for by in range(t):
            for bx in range(t):
                tile = plane[by * th:(by + 1) * th, bx * tw:(bx + 1) * tw]
                r, n, _ = estimate_rate(tile, stride=stride,
                                        min_count=min_count)
                n_bins_total += n
                if np.isfinite(r):
                    rates.append(r)
        rate = float(np.median(rates)) if rates else float("nan")
        n_bins = n_bins_total
        beta = 1.0 - rate if np.isfinite(rate) else float("nan")

    return Result(
        method="chi2",
        rate=rate,
        n_obs=n_bins,
        p_uniform=p_uniform,
        details={"n_pairs": n_pairs, "beta": beta, "tiles": t},
    )
