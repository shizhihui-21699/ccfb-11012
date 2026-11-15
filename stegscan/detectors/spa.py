"""Sample-pairs analysis (Dumitrescu, Wu & Memon, ICIP 2002).

The image is split into horizontally and vertically adjacent pixel pairs
(u, v).  Four "primary sets" are counted:

    X = { (u,v) : v even and u < v, or v odd and u > v }
    Y = { (u,v) : v even and u > v, or v odd and u < v }
    W = { (u,v) in Y : u and v differ only in the LSB }   (|u-v| == 1 and u even, v odd)
    Z = { (u,v) : u == v }

For a natural image |X| = |Y| statistically.  Writing P for the number of
pairs and gamma = |W| + |Z|, the embedding rate p (message bits per pixel)
is the smaller root of

    0.5 * gamma * p^2 + (2|X| - |P|) * p + (|Y| - |X|) = 0

which is equation (9) of the paper.  |X|, |Y|, |W|, |Z| are measured on
the image under examination (primed sets in the paper).

Practical notes:

* W must be the subset of Y that differs only in the LSB, i.e. the pair
  must be (2k, 2k+1).  Counting every adjacent pair with |u-v| == 1 would
  add pairs that belong to X and break the algebra.
* The quadratic has no real root when the image is far from the model
  (very high rates on some content).  We then fall back to the linear
  term, which is the paper's own degenerate case, and flag it.
* SPA systematically underestimates near p -> 1 (the paper's Assumption 1
  degrades once the LSB plane is fully randomised).  The scanner applies a
  calibration curve.
"""

from __future__ import annotations

import numpy as np

from .result import Result

# Above this |Y-X|/P, SPA is saturated: measured on real photographs the
# imbalance is +0.0003 on clean images, +0.039 at half capacity and +0.069
# at full capacity, and SPA's rate stops increasing past ~0.5 capacity.
# 0.05 sits clear of the half-capacity reading and inside the full-capacity
# range.
SPA_SATURATION_IMBALANCE = 0.050


def _pairs(plane: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Horizontal and vertical adjacent pairs, flattened to 1-D."""
    h, w = plane.shape
    out = [(plane[:, :-1].reshape(-1), plane[:, 1:].reshape(-1))]
    if h > 1:
        out.append((plane[:-1, :].reshape(-1), plane[1:, :].reshape(-1)))
    return out


def count_primary_sets(plane: np.ndarray) -> dict[str, int]:
    X = Y = W = Z = P = 0
    for u, v in _pairs(plane):
        u = u.astype(np.int32)
        v = v.astype(np.int32)
        v_even = (v & 1) == 0
        x = ((v_even & (v > u)) | (~v_even & (v < u)))
        y = ((v_even & (v < u)) | (~v_even & (v > u)))
        # W is Y restricted to LSB-only differences: (2k, 2k+1) with u even.
        w = y & (np.abs(u - v) == 1)
        X += int(x.sum())
        Y += int(y.sum())
        W += int(w.sum())
        Z += int((u == v).sum())
        P += int(u.size)
    return {"X": X, "Y": Y, "W": W, "Z": Z, "P": P}


def rate_from_sets(sets: dict[str, int]) -> tuple[float, str]:
    """Solve the SPA quadratic; returns (rate, note)."""
    X, Y, W, Z, P = sets["X"], sets["Y"], sets["W"], sets["Z"], sets["P"]
    gamma = W + Z
    a = 0.5 * gamma
    b = 2.0 * X - P
    c = Y - X

    if a <= 0:
        # gamma == 0: the method fails by construction (paper, section 2).
        if abs(b) < 1e-12:
            return float("nan"), "gamma=0 (no LSB-only pairs; SPA undefined)"
        return float(np.clip(-c / b, 0.0, 1.0)), "gamma=0 (linear fallback)"

    disc = b * b - 4.0 * a * c
    if disc < 0:
        # Outside the model's validity.  The linear term is the best
        # available answer and is marked as low confidence.
        if abs(b) < 1e-12:
            return float("nan"), "no real root; discriminant<0"
        return float(np.clip(-c / b, 0.0, 1.0)), "no real root; linear fallback"

    root = np.sqrt(disc)
    r1 = (-b - root) / (2.0 * a)
    r2 = (-b + root) / (2.0 * a)
    cand = [r for r in (r1, r2) if -0.05 <= r <= 1.05]
    if cand:
        return float(np.clip(min(cand), 0.0, 1.0)), ""
    # Both roots outside [0,1]: clip the one closest to the feasible range.
    r = min((r1, r2), key=lambda t: min(abs(t), abs(t - 1.0)))
    return float(np.clip(r, 0.0, 1.0)), "root outside [0,1]"


def assumption_residual(sets: dict[str, int]) -> float:
    """The rate the estimator would report from the |X| = |Y| mismatch alone.

    SPA's Assumption 1 is that |X| = |Y| for a natural image.  It holds only
    approximately, and the estimate is sensitive to the residual: on a
    typical plane (gamma/P ~ 0.09) a 1 % excess of Y over X moves the
    answer by about 0.10.  Measured against real photographs, (Y-X)/P runs
    from +0.0001 to +0.0100 depending on content, so any SPA reading below
    roughly 0.1 can be produced entirely by the assumption's failure rather
    than by a payload.

    Returning this as a diagnostic lets the scanner tell "SPA sees a
    payload" apart from "SPA's baseline is off for this image".
    """
    X, Y, W, Z, P = sets["X"], sets["Y"], sets["W"], sets["Z"], sets["P"]
    if P <= 0:
        return float("nan")
    # Evaluate the same quadratic with the idealised X == Y.
    ideal = dict(sets)
    balanced = (X + Y) // 2
    ideal["X"] = balanced
    ideal["Y"] = balanced
    rate_ideal, _ = rate_from_sets(ideal)
    return rate_ideal


def analyse_plane(plane: np.ndarray) -> Result:
    sets = count_primary_sets(plane)
    rate, note = rate_from_sets(sets)
    baseline = assumption_residual(sets)
    details = dict(sets)
    details["assumption_baseline"] = baseline

    # Saturation test.  Assumption 1 says |X| = |Y|, so (Y-X)/P should be
    # zero at every payload level.  Measured across 16 real photographs it
    # grows monotonically with the true rate -- median +0.0003 clean,
    # +0.039 at half capacity, +0.069 at full capacity -- because once
    # enough bits are embedded the set membership is no longer the natural
    # one the derivation assumes.  SPA's own output is flat from ~0.5 upward
    # (it reads 0.50 at a true 50 % and 0.50 at 100 %), so a high (Y-X)/P
    # means the estimate has stopped tracking the payload and must not
    # constrain the consensus.
    P = sets["P"]
    imbalance = (sets["Y"] - sets["X"]) / P if P else float("nan")
    details["xy_imbalance"] = imbalance
    saturated = bool(np.isfinite(imbalance) and abs(imbalance) > SPA_SATURATION_IMBALANCE)
    if saturated:
        note = (f"{note}; " if note else "") + \
               "saturated (|X-Y| large: SPA no longer tracks the payload)"

    if np.isfinite(baseline) and rate < baseline:
        note = (f"{note}; " if note else "") + \
               f"below the |X|=|Y| baseline ({baseline:.3f})"
    return Result(
        method="spa",
        rate=rate,
        n_obs=int(P),
        note=note,
        details=details,
        model_saturated=saturated,
    )
