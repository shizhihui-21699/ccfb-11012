"""DCT coefficient-histogram analysis for JPEG (JSteg-family embedding).

Motivation
----------
JSteg and its relatives replace the LSB of quantised DCT coefficients,
skipping the values 0 and 1 (which would otherwise produce visible
artefacts and break the run-length coding).  Within each pair

    (2k, 2k+1)   and   (-2k, -2k-1)

LSB replacement is closed: a coefficient can only move between the two
members of its pair.  If a fraction p of the eligible coefficients is
modified, then for every pair

    S = h(2k) + h(2k+1)          is unchanged
    d = h(2k) - h(2k+1)          is multiplied by (1 - p)

The pair (0, 1) and the pair (-2, -1) are protected: -1, 0 and 1 all
survive embedding untouched, so their differences are not scaled.  They
are excluded from the regression.

Calibration
-----------
The cover histogram itself is unknown.  The standard trick (Fridrich's
calibrated steganalysis) is to build a reference from the image itself:
crop a few pixels -- *not* a multiple of 8, so the 8x8 block grid shifts
by a sub-block offset -- and recompress with the file's own quantisation
tables.  The shifted grid destroys the block-aligned LSB structure while
preserving the statistics of the content, so the recompressed file is a
good proxy for the cover.  Regressing the observed pair differences on
the calibrated ones gives (1 - p) directly.

The estimate is reported per component (Y, Cb, Cr) and pooled.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .result import Result

try:  # jpeglib gives lossless access to the quantised coefficients
    import jpeglib
except Exception:  # pragma: no cover - optional dependency
    jpeglib = None

# Coefficient magnitude range entering the histogram.  Large coefficients
# are rare and their pair counts are too small to be informative.
VMAX = 40
# Minimum pair total before a pair enters the regression.
MIN_PAIR_COUNT = 30
# Crop offset in pixels.  4 is not a multiple of 8, which is what shifts
# the block grid.
CALIB_CROP = 4
# Trim the most extreme calibration differences landed badly in testing:
# the largest |D| bins carry most of the payload signal, so trimming them
# biases the estimate upward on clean images.  Left at 0; the protected
# pairs and the MIN_PAIR_COUNT gate are what keep the fit sane.
TRIM_FRACTION = 0.0


def available() -> bool:
    return jpeglib is not None


@dataclass
class _Paired:
    h0: np.ndarray        # counts of the even member of each pair
    h1: np.ndarray        # counts of the odd member
    lo: int               # value of the first even bin


def signed_pairs(coeffs: np.ndarray, vmax: int = VMAX) -> _Paired:
    """Pair counts over signed coefficient values in [-vmax, vmax]."""
    a = coeffs.reshape(-1)
    a = a[(a >= -vmax) & (a <= vmax)].astype(np.int32)
    lo = -vmax
    hi = vmax + 1
    h = np.bincount(a - lo, minlength=hi - lo).astype(np.float64)
    n = (hi - lo) // 2
    return _Paired(h[0:2 * n:2].copy(), h[1:2 * n:2].copy(), lo)


def _recompress_shifted(path: Path, tmpdir: Path) -> "jpeglib.DCTJPEG":
    """Crop by a non-block-multiple offset and recompress with the file's tables."""
    j = jpeglib.read_dct(str(path))
    arr = np.asarray(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    m = CALIB_CROP
    if h <= 2 * m or w <= 2 * m:
        raise ValueError("image too small for calibration crop")
    crop = arr[m:h - m, m:w - m]

    s = jpeglib.from_spatial(crop)
    out = tmpdir / "calibrated.jpg"
    s.write_spatial(str(out), qt=np.stack(j.qt), quant_tbl_no=j.quant_tbl_no)
    return j, jpeglib.read_dct(str(out))


def _estimate_component(
    obs: _Paired,
    cal: _Paired,
    min_pair_count: int = MIN_PAIR_COUNT,
) -> tuple[float, float, int, float]:
    """Return (rate, chi2 p-value, bins used, beta)."""
    n = min(len(obs.h0), len(obs.h1), len(cal.h0), len(cal.h1))
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan")
    h0, h1 = obs.h0[:n], obs.h1[:n]
    c0, c1 = cal.h0[:n], cal.h1[:n]
    S = h0 + h1
    d = h0 - h1
    Sc = c0 + c1
    Dc = c0 - c1

    i_skip = (0 - obs.lo) // 2          # pair (0, 1)
    protected = {i_skip - 1, i_skip}    # (-2,-1) and (0,1) hold untouched values

    usable = (S >= min_pair_count) & (Sc >= min_pair_count) & (np.abs(Dc) > 0)
    for j in protected:
        if 0 <= j < n:
            usable[j] = False
    if usable.sum() < 6:
        return float("nan"), float("nan"), int(usable.sum()), float("nan")

    dd = d[usable]
    DD = Dc[usable]
    keep = np.ones(DD.shape, dtype=bool)
    if TRIM_FRACTION > 0:
        cut = np.quantile(np.abs(DD), 1.0 - TRIM_FRACTION)
        keep = np.abs(DD) <= cut
        dd, DD = dd[keep], DD[keep]
    if DD.size < 4:
        return float("nan"), float("nan"), int(DD.size), float("nan")

    beta = float((dd * DD).sum() / (DD * DD).sum())
    rate = float(np.clip(1.0 - beta, 0.0, 1.0))

    # Goodness of fit of the observed pairs against the calibrated shape
    # stretched by the estimated (1-p).  Only bins with a workable expected
    # count enter; the statistic is reported as a diagnostic, not a gate.
    Sc_u = Sc[usable][keep]
    ratio = DD / Sc_u
    e0 = Sc_u * (1 + ratio) / 2.0
    e1 = Sc_u * (1 - ratio) / 2.0
    good = e0 > 5
    if good.sum() >= 3:
        h0u = h0[usable][keep][good]
        h1u = h1[usable][keep][good]
        stat = (((h0u - e0[good]) ** 2) / e0[good]).sum() + \
               (((h1u - e1[good]) ** 2) / e1[good]).sum()
        from scipy.stats import chi2
        pv = float(chi2.sf(stat, max(1, 2 * int(good.sum()) - 1)))
    else:
        pv = float("nan")
    return rate, pv, int(good.sum()), beta


def analyse_file(path: str | Path) -> dict:
    """Analyse a JPEG file in the DCT domain.

    Returns a dict with per-component rates plus pooled values.  Raises
    RuntimeError when jpeglib is unavailable or the file cannot be parsed.
    """
    if jpeglib is None:
        raise RuntimeError(
            "DCT analysis needs the 'jpeglib' package (pip install jpeglib)"
        )

    path = Path(path)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        j, k = _recompress_shifted(path, tmpdir)

        comps = {"Y": j.Y, "Cb": j.Cb, "Cr": j.Cr}
        cal = {"Y": k.Y, "Cb": k.Cb, "Cr": k.Cr}
        per = {}
        for name, coeff in comps.items():
            if coeff is None:
                continue
            # Chroma planes are subsampled and quantised more coarsely, so
            # their histograms are much thinner; use a lower gate for them.
            gate = MIN_PAIR_COUNT if name == "Y" else 10
            obs = signed_pairs(coeff.astype(np.int32)[..., 1:].reshape(-1))
            ref = signed_pairs(cal[name].astype(np.int32)[..., 1:].reshape(-1))
            rate, pv, bins, beta = _estimate_component(obs, ref, min_pair_count=gate)
            per[name] = Result(
                method=f"dct-{name}",
                rate=rate,
                n_obs=bins,
                p_uniform=pv,
                details={"beta": beta},
            )

        # Pool: weight each component by the number of coefficients it
        # contributes, since chroma planes are subsampled.
        weighted = 0.0
        weight = 0.0
        for name, r in per.items():
            coeff = comps[name]
            n_coef = int(coeff.astype(np.int32)[..., 1:].size)
            if r.valid:
                weighted += r.rate * n_coef
                weight += n_coef
        pooled = weighted / weight if weight else float("nan")
        return {"components": per, "rate": pooled, "nbins": sum(r.n_obs for r in per.values())}
