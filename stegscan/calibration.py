"""Empirical calibration for the quantitative rate estimators.

Each detector reads a raw statistic out of the image and maps it to an
embedding rate.  The raw estimators are not all unbiased:

* chi-square (pair-difference regression) is close to unbiased from ~10 %
  capacity upward but is compressed at very low rates;
* sample-pairs is accurate up to about half capacity, then saturates
  (Dumitrescu's Assumption 1 degrades once the LSB plane is randomised);
* RS underestimates from roughly 20 % capacity onward, and badly above
  half capacity;
* the DCT pair-difference estimator is close to linear but slightly
  compressed at low rates.

The tables below were produced with ``tools/calibrate.py`` on a generated
corpus of natural-style photographs and hard textures, embedded with both
sequential and scattered payloads at PNG/BMP and several JPEG qualities.
Each entry is (median raw estimate -> true rate).  The scanner inverts the
curve, so the number in the report estimates the *true* rate rather than
the estimator's own scale.

These maps are content-dependent; they are a default, not ground truth.
``scan_file(..., calibrate=False)`` exposes the raw output, and the report
records which mode was used.
"""

from __future__ import annotations

import numpy as np


# method -> (raw_estimate_points, true_rate_points)
#
# Measured on 16 real photographs (public-domain Commons images, all three
# channels, capacity-pooled), which is the reference that matters: the
# synthetic corpus in ``corpus.py`` is a stress test, not a calibration
# source, because its histograms are smoother than photography and bias the
# fit.  The bold rows show how close each raw estimator is to the truth:
#
#   true rate      0.05   0.10   0.25   0.50   0.75   1.00
#   chi-square     0.075  0.107  0.268  0.466  0.714  0.883   (compressed high)
#   sample pairs   0.055  0.105  0.258  0.502  0.758  0.730   (flattens at full)
#   RS             0.042  0.084  0.195  0.364  0.545  0.866   (compressed)
#
# So chi-square and SPA need only gentle correction; RS needs a real curve.
CURVES: dict[str, tuple[list[float], list[float]]] = {
    "chi2": (
        [0.028, 0.035, 0.075, 0.107, 0.215, 0.360, 0.466, 0.714, 0.883],
        [0.000, 0.020, 0.050, 0.100, 0.200, 0.350, 0.500, 0.750, 1.000],
    ),
    "spa": (
        # SPA saturates hard: it reads ~0.73 at full capacity, not 1.0.  The
        # last point is the value at which it stops increasing; a fully
        # embedded plane is normally caught by the saturation path in
        # scan.py, not by this curve.
        [0.006, 0.025, 0.055, 0.105, 0.210, 0.355, 0.502, 0.758, 0.800],
        [0.000, 0.020, 0.050, 0.100, 0.200, 0.350, 0.500, 0.750, 1.000],
    ),
    "rs": (
        [0.015, 0.021, 0.042, 0.084, 0.150, 0.275, 0.364, 0.545, 0.866],
        [0.000, 0.020, 0.050, 0.100, 0.200, 0.350, 0.500, 0.750, 1.000],
    ),
    "dct": (
        [0.0152, 0.0409, 0.0736, 0.1364, 0.2585, 0.3994, 0.5557, 0.7844, 1.0000],
        [0.000, 0.020, 0.050, 0.100, 0.200, 0.350, 0.500, 0.750, 1.000],
    ),
}

# The raw value at which each method is still essentially reporting noise.
# One to two standard deviations above the clean-image median, measured on
# the same real-photo reference.
NOISE_FLOOR: dict[str, float] = {
    "chi2": 0.060,
    "spa": 0.045,
    "rs": 0.055,
    "dct": 0.050,
}


def calibrate(method: str, raw: float, enabled: bool = True) -> float:
    """Map a raw estimator output to a corrected rate in [0, 1]."""
    if not enabled or raw != raw:          # NaN passes through
        return raw
    xs, ys = CURVES[method]
    x = float(np.clip(raw, xs[0], xs[-1]))
    return float(np.clip(np.interp(x, xs, ys), 0.0, 1.0))


def is_above_noise(method: str, raw: float) -> bool:
    return raw == raw and raw > NOISE_FLOOR.get(method, 0.0)
