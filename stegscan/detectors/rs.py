"""RS analysis (Fridrich, Goljan & Du, 2001).

Pixels are split into disjoint groups of n adjacent samples.  A
discrimination function f(G) = sum |x_{i+1} - x_i| measures group
noisiness, and two flipping operators are applied to a mask M of the
group:

    F1  : LSB flip  (x -> x-1 for odd x, x+1 for even x)
    F-1 : shifted flip (x -> x+1 for even x, x-1 for odd x)

Groups whose noisiness rises under F1 are "regular" (R_M), those whose
noisiness falls are "singular" (S_M); same for F-1, giving R_{-M} and
S_{-M}.  For a natural image R_M ~ R_{-M} and S_M ~ S_{-M}.  Embedding
equalises them.  With the notation

    d0  = R_M(p/2)   - S_M(p/2)
    d-0 = R_-M(p/2)  - S_-M(p/2)
    d1  = R_M(1-p/2) - S_M(1-p/2)     (all pixels flipped)
    d-1 = R_-M(1-p/2)- S_-M(1-p/2)

the paper's quadratic is

    2(d1 + d0) x^2 + (d-0 - d-1 - d1 - 3 d0) x + d0 - d-0 = 0

and the embedding rate is p = |x / (x - 1/2)|.

Notes:

* Two masks are averaged (the checkerboard mask and its complement) as
  in the reference implementation, which reduces sensitivity to image
  structure aligned with the mask.
* RS mirrors SPA: reliable up to roughly half capacity, increasingly
  biased low beyond that.  The scanner applies a calibration curve.
"""

from __future__ import annotations

import numpy as np

from .result import Result


def _groups(plane: np.ndarray, mh: int, mw: int) -> np.ndarray:
    """Non-overlapping mh x mw groups, flattened to (n_groups, mh*mw)."""
    h, w = plane.shape
    gh, gw = h // mh, w // mw
    if gh == 0 or gw == 0:
        return np.empty((0, mh * mw), dtype=np.int32)
    blocks = plane[: gh * mh, : gw * mw]
    blocks = blocks.reshape(gh, mh, gw, mw).transpose(0, 2, 1, 3)
    return blocks.reshape(-1, mh * mw).astype(np.int32)


def _variation(groups: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(groups, axis=1)).sum(axis=1)


def _flip(groups: np.ndarray, mask: np.ndarray, negative: bool) -> np.ndarray:
    odd = (groups & 1).astype(bool)
    if negative:
        step = np.where(odd, 1, -1)
    else:
        step = np.where(odd, -1, 1)
    return np.where(mask, groups + step, groups)


def _rs_rates(groups: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    base = _variation(groups)
    pos = _variation(_flip(groups, mask, negative=False))
    neg = _variation(_flip(groups, mask, negative=True))
    R = float(np.mean(pos > base))
    S = float(np.mean(pos < base))
    Rn = float(np.mean(neg > base))
    Sn = float(np.mean(neg < base))
    return R, S, Rn, Sn


def count_rs(plane: np.ndarray, mh: int = 2, mw: int = 2) -> dict[str, float]:
    groups = _groups(plane, mh, mw)
    if groups.shape[0] < 16:
        return {}

    mask = np.zeros((mh, mw), dtype=bool)
    mask[::2, ::2] = True
    mask[1::2, 1::2] = True
    mask = mask.reshape(-1)
    mask_c = ~mask
    mask_all = np.ones(mh * mw, dtype=bool)

    r1, s1, rn1, sn1 = _rs_rates(groups, mask)
    r2, s2, rn2, sn2 = _rs_rates(groups, mask_c)
    r_all, s_all, rn_all, sn_all = _rs_rates(groups, mask_all)

    # Average the two complementary masks (reference implementation).
    R = (r1 + r2) / 2.0
    S = (s1 + s2) / 2.0
    Rn = (rn1 + rn2) / 2.0
    Sn = (sn1 + sn2) / 2.0

    return {
        "R": R, "S": S, "Rn": Rn, "Sn": Sn,
        "R_all": r_all, "S_all": s_all, "Rn_all": rn_all, "Sn_all": sn_all,
        "n_groups": float(groups.shape[0]),
    }


def rate_from_rs(stats: dict[str, float]) -> tuple[float, str]:
    if not stats:
        return float("nan"), "image too small for RS groups"

    d0 = stats["R"] - stats["S"]
    dm0 = stats["Rn"] - stats["Sn"]
    d1 = stats["R_all"] - stats["S_all"]
    dm1 = stats["Rn_all"] - stats["Sn_all"]

    a = 2.0 * (d1 + d0)
    b = dm0 - dm1 - d1 - 3.0 * d0
    c = d0 - dm0

    note = ""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return float("nan"), "degenerate quadratic"
        x = c / b
        note = "linear fallback"
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0:
            if abs(b) < 1e-12:
                return float("nan"), "no real root"
            x = c / b
            note = "no real root; linear fallback"
        else:
            root = np.sqrt(disc)
            r1 = (-b + root) / (2.0 * a)
            r2 = (-b - root) / (2.0 * a)
            x = r1 if abs(r1) <= abs(r2) else r2

    if abs(x - 0.5) < 1e-9:
        return float("nan"), "x=0.5 (rate undefined)"
    rate = abs(x / (x - 0.5))
    return float(np.clip(rate, 0.0, 1.0)), note


def analyse_plane(plane: np.ndarray, mh: int = 2, mw: int = 2) -> Result:
    stats = count_rs(plane, mh=mh, mw=mw)
    rate, note = rate_from_rs(stats)
    return Result(
        method="rs",
        rate=rate,
        n_obs=int(stats.get("n_groups", 0)),
        note=note,
        details=stats,
    )
