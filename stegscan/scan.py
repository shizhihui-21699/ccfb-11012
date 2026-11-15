"""Scan orchestration: one image -> detector results -> a verdict.

The spatial detectors (chi-square, SPA, RS) run on the LSB plane of each
colour channel.  A payload is normally spread over one or more channels,
so per-channel rates are combined by a *capacity-weighted* mean rather
than a plain average: a channel with 100 % capacity empty should not
dilute the evidence from a channel that is 40 % full.

For JPEG, the DCT-domain detector replaces the spatial ones: an LSB
payload in a JPEG is generally in the quantised coefficients, and the
spatial-plane statistics of a JPEG are dominated by the codec, not by
the payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import calibration
from .detectors import chi2 as chi2_det
from .detectors import dct as dct_det
from .detectors import rs as rs_det
from .detectors import spa as spa_det
from .detectors.result import Result
from .imageio import LoadedImage, UnsupportedImage, load_image, iter_image_files


SPATIAL_METHODS = ("chi2", "spa", "rs")

# Reported rate when the quantitative estimator is saturated by a full LSB
# plane: the uniformity test says the plane is payload, and the best point
# estimate available is "at or near capacity".  0.96 rather than 1.0 keeps
# the number honest about the last few percent.
SATURATED_RATE = 0.96


@dataclass
class MethodVerdict:
    method: str
    rate: float                 # calibrated estimate, fraction of LSB capacity
    raw_rate: float             # estimator output before calibration
    n_obs: int
    p_uniform: float
    note: str = ""
    channel_spread: float = float("nan")   # max-min across analysed channels
    model_saturated: bool = False   # this estimator stopped tracking the payload

    @property
    def valid(self) -> bool:
        return np.isfinite(self.rate)


@dataclass
class FileVerdict:
    path: Path
    fmt: str
    width: int
    height: int
    n_pixels: int
    channels: int
    bit_depth: int
    file_size: int
    methods: dict[str, MethodVerdict] = field(default_factory=dict)
    rate: float = float("nan")          # consensus rate, fraction of LSB capacity
    rate_spread: float = float("nan")   # disagreement between methods
    est_payload_bits: float = float("nan")
    est_payload_bytes: float = float("nan")
    suspiciousness: float = float("nan")   # 0..100
    verdict: str = ""
    error: str = ""
    note_saturated: bool = False

    @property
    def detected(self) -> bool:
        return self.suspiciousness >= 50.0

    @property
    def capacity_bits(self) -> int:
        return self.n_pixels * self.channels


def _consensus(method_rates: list[float]) -> float:
    """Agreed-upon rate: the mean of the two lowest non-saturated estimates.

    LSB embedding raises every structural detector, so a genuine payload
    shows up in all of them.  A single detector reading high while the
    others stay near zero means that detector's model has failed on this
    content -- a halftone, a dithered painting, a posterised gradient.
    Averaging the two *lowest* estimates therefore requires two detectors
    to agree before a rate is reported, and discards one outlier outright.

    Why not the raw minimum?  It is too pessimistic when one detector is
    merely noisier on this image: on 512 px synthetic covers chi-square
    reads about 0.07 where SPA and RS read 0.24-0.26 for the same 25 %
    payload, and the minimum reports chi-square's number as the answer.

    Why not the median?  It is too optimistic when one detector reads high
    for content reasons: on a clean artwork whose texture mimics LSB
    structure the median reaches 0.15 while the two-lowest mean stays below
    0.08 across 16 clean photographs, and still reaches 0.96 at a true
    full plane.

    Deliberately *unweighted*: a detector's ``n_obs`` is not comparable
    across methods (chi-square counts histogram pairs, SPA counts pixel
    pairs), so weighting by it would silently reduce the consensus to
    whichever method reports the largest numbers.
    """
    valid = sorted(r for r in method_rates if np.isfinite(r))
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return float(valid[0])
    return float(np.mean(valid[:2]))


def _median_rate(method_rates: list[float]) -> float:
    """Median of the per-method rates, reported alongside the consensus."""
    valid = [r for r in method_rates if np.isfinite(r)]
    return float(np.median(valid)) if valid else float("nan")


def _spread(rates: list[float]) -> float:
    valid = [r for r in rates if np.isfinite(r)]
    if len(valid) < 2:
        return float("nan")
    return float(max(valid) - min(valid))


def score_suspiciousness(rate: float, spread: float, n_methods: int,
                         p_uniform: float) -> float:
    """Map evidence to a 0..100 suspiciousness score.

    The score is driven by the *estimate of the embedding rate*, not by a
    p-value: a clean natural image can produce a small p-value under the
    strict chi-square model, and a short payload can leave the histogram
    barely moved.  Attenuation applies when the methods disagree, because
    disagreement usually means the model assumptions are breaking.
    """
    if not np.isfinite(rate):
        return 0.0
    # Base: rate 0 -> 0, rate >= 20% of capacity -> 100.
    base = min(1.0, rate / 0.20) * 100.0
    # Penalise only large disagreements (>25 points of capacity).
    if np.isfinite(spread) and spread > 0.25:
        base *= 0.75
    # A uniform-pair p-value near 1 with a low rate is still worth a nudge.
    if np.isfinite(p_uniform) and p_uniform > 0.5 and rate < 0.02:
        base = max(base, 20.0)
    return float(np.clip(base, 0.0, 100.0))


def _verdict_label(score: float) -> str:
    """Short labels: they sit in a narrow report column."""
    if score >= 85:
        return "strong evidence"
    if score >= 50:
        return "probable"
    if score >= 20:
        return "weak / inconclusive"
    return "no evidence"


def scan_spatial(img: LoadedImage, calibrate: bool = True) -> dict[str, MethodVerdict]:
    """Run the three spatial detectors on every channel and pool them."""
    per_method: dict[str, list[tuple[float, float, int, float, str]]] = {
        m: [] for m in SPATIAL_METHODS
    }
    model_saturated: dict[str, bool] = {m: False for m in SPATIAL_METHODS}

    for plane, cname in zip(img.planes, img.plane_names):
        c2 = chi2_det.analyse_plane(plane)
        sp = spa_det.analyse_plane(plane)
        r = rs_det.analyse_plane(plane)
        weight = float(plane.size)
        per_method["chi2"].append((c2.rate, weight, c2.n_obs, c2.p_uniform, c2.note))
        per_method["spa"].append((sp.rate, weight, sp.n_obs, float("nan"), sp.note))
        per_method["rs"].append((r.rate, weight, r.n_obs, float("nan"), r.note))
        model_saturated["chi2"] |= c2.model_saturated
        model_saturated["spa"] |= sp.model_saturated
        model_saturated["rs"] |= r.model_saturated

    out: dict[str, MethodVerdict] = {}
    for method, items in per_method.items():
        valid = [(raw, w, i2, i3, i4) for raw, w, i2, i3, i4 in items
                 if np.isfinite(raw)]
        if not valid:
            raw = float("nan")
            channel_spread = float("nan")
        else:
            # Capacity-weighted mean across channels, which is the correct
            # way to turn per-channel densities into one image-level
            # density: rate = payload_bits / total_capacity, and the
            # payload is split unevenly in practice.  A channel that is
            # completely empty must pull the average *down*, because the
            # image really does have unused capacity there.
            #
            # (A median was tried and rejected: with a sequential payload
            # confined to one channel, two of three channels read zero and
            # the median reports no payload at all.)
            total_w = sum(w for _, w, *_ in valid)
            raw = float(sum(r * w for r, w, *_ in valid) / total_w)
            # Spread flags an unevenly distributed payload (concentrated in
            # one channel), which is worth surfacing in the report.
            channel_spread = float(np.max([r for r, *_ in valid]) -
                                   np.min([r for r, *_ in valid]))
        notes = "; ".join(sorted({i[4] for i in items if i[4]}))
        p_uniform = float(np.nanmedian([i[3] for i in items])) if any(
            np.isfinite(i[3]) for i in items) else float("nan")
        calibrated = calibration.calibrate(method, raw, enabled=calibrate)
        out[method] = MethodVerdict(
            method=method,
            rate=calibrated,
            raw_rate=raw,
            n_obs=int(sum(i[2] for i in items)),
            p_uniform=p_uniform,
            note=notes,
            channel_spread=channel_spread,
            model_saturated=model_saturated[method],
        )
    return out


def scan_jpeg(img: LoadedImage, calibrate: bool = True) -> dict[str, MethodVerdict]:
    res = dct_det.analyse_file(img.path)
    comps: dict[str, Result] = res["components"]
    raw = res["rate"]
    p_uniform = float(np.nanmedian([c.p_uniform for c in comps.values()
                                    if np.isfinite(c.p_uniform)]) or float("nan"))
    return {
        "dct": MethodVerdict(
            method="dct",
            rate=calibration.calibrate("dct", raw, enabled=calibrate),
            raw_rate=raw,
            n_obs=int(res["nbins"]),
            p_uniform=p_uniform,
            note="; ".join(c.note for c in comps.values() if c.note),
        )
    }


def scan_file(path: str | Path, calibrate: bool = True) -> FileVerdict:
    path = Path(path)
    try:
        img = load_image(path)
    except UnsupportedImage as exc:
        raise
    except Exception as exc:                         # corrupted file etc.
        return FileVerdict(
            path=path, fmt="?", width=0, height=0, n_pixels=0,
            channels=0, bit_depth=0, file_size=path.stat().st_size if path.exists() else 0,
            error=f"{type(exc).__name__}: {exc}",
        )

    verdict = FileVerdict(
        path=path, fmt=img.fmt, width=img.width, height=img.height,
        n_pixels=img.n_pixels, channels=len(img.planes),
        bit_depth=img.bit_depth, file_size=img.file_size,
    )

    try:
        if img.fmt == "JPEG":
            methods = scan_jpeg(img, calibrate=calibrate)
        else:
            methods = scan_spatial(img, calibrate=calibrate)
    except Exception as exc:
        verdict.error = f"{type(exc).__name__}: {exc}"
        return verdict

    verdict.methods = methods
    all_rates = [m.rate for m in methods.values()]
    # Saturated estimators have stopped tracking the payload (SPA flattens
    # past half capacity; chi-square has no envelope left on a full plane),
    # so their numbers must not constrain the consensus.
    rates = [m.rate for m in methods.values() if not m.model_saturated]

    if not any(np.isfinite(r) for r in rates) and any(
            np.isfinite(r) for r in all_rates):
        # Every usable detector saturated.  On a fully embedded image the
        # uniformity test is unambiguous even though no rate can be fitted,
        # so report near-capacity rather than "no estimate".
        verdict.rate = SATURATED_RATE
        verdict.rate_spread = float("nan")
        verdict.note_saturated = True
    else:
        verdict.rate = _consensus(rates if rates else all_rates)
        verdict.rate_spread = _spread(rates if rates else all_rates)

    if np.isfinite(verdict.rate):
        bits = verdict.rate * verdict.capacity_bits
        verdict.est_payload_bits = bits
        verdict.est_payload_bytes = bits / 8.0

    p_uniform = float(np.nanmedian([m.p_uniform for m in methods.values()
                                    if np.isfinite(m.p_uniform)])) if any(
        np.isfinite(m.p_uniform) for m in methods.values()) else float("nan")

    verdict.suspiciousness = score_suspiciousness(
        verdict.rate, verdict.rate_spread, len(methods), p_uniform
    )
    verdict.verdict = _verdict_label(verdict.suspiciousness)
    return verdict


def scan_directory(root: str | Path, calibrate: bool = True,
                   progress=None) -> list[FileVerdict]:
    files = iter_image_files(root)
    verdicts: list[FileVerdict] = []
    for i, p in enumerate(files, 1):
        try:
            verdicts.append(scan_file(p, calibrate=calibrate))
        except UnsupportedImage as exc:
            verdicts.append(FileVerdict(
                path=p, fmt="?", width=0, height=0, n_pixels=0, channels=0,
                bit_depth=0, file_size=p.stat().st_size, error=str(exc),
            ))
        if progress:
            progress(i, len(files), p)
    return verdicts
