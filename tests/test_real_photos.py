"""Properties the detectors must hold on real photographs.

The synthetic corpus used elsewhere is deliberately harsh; these tests use
four public-domain photographs shipped in ``tests/data`` so that a
regression in the *realistic* case is caught, not just a change in the
synthetic case.

The assertions are about behaviour (near-zero on clean, monotone in
payload, roughly calibrated), not about exact output of one build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stegscan.corpus import embed_lsb
from stegscan.detectors import chi2 as chi2_det
from stegscan.detectors import rs as rs_det
from stegscan.detectors import spa as spa_det
from stegscan.imageio import load_image


REFERENCE = sorted((Path(__file__).parent / "data").glob("ref*.png"))


def _planes(path: Path) -> list[np.ndarray]:
    return load_image(path).planes


def _pooled(det, planes) -> float:
    vals = [det(p).rate for p in planes]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _embedded(path: Path, rate: float) -> list[np.ndarray]:
    arr = np.asarray(Image.open(path).convert("RGB"))
    stego, _ = embed_lsb(arr, rate, seed=5)
    return [stego[..., c] for c in range(3)]


def test_reference_set_present():
    assert len(REFERENCE) == 4, f"expected 4 fixtures, found {len(REFERENCE)}"


def test_clean_photos_score_zero():
    """Each detector is near zero on clean photographs.

    Thresholds are per-method because the methods' clean-image tails
    differ.  Measured over 48 real-photo channels, the 95th percentile of
    clean readings is 0.10 for chi-square, 0.17 for sample pairs and 0.14
    for RS: SPA's estimate is a quadratic whose constant term is the
    |X|-|Y| mismatch, which is small but not zero in any real image, and a
    1 % mismatch alone produces ~0.10.  Asserting a single tight bound for
    all three would either fail on honest data or hide the difference.
    """
    bounds = {chi2_det.analyse_plane: 0.12,
              spa_det.analyse_plane: 0.20,
              rs_det.analyse_plane: 0.16}
    for path in REFERENCE:
        for det, bound in bounds.items():
            r = _pooled(det, _planes(path))
            assert np.isfinite(r), f"{det.__module__} produced no estimate on {path.name}"
            assert r < bound, f"{det.__module__} read {r:.3f} on clean {path.name} (bound {bound})"


def test_all_detectors_rise_with_payload():
    for path in REFERENCE:
        planes = _planes(path)
        clean = {
            "chi2": _pooled(chi2_det.analyse_plane, planes),
            "spa": _pooled(spa_det.analyse_plane, planes),
            "rs": _pooled(rs_det.analyse_plane, planes),
        }
        half = {
            "chi2": _pooled(chi2_det.analyse_plane, _embedded(path, 0.5)),
            "spa": _pooled(spa_det.analyse_plane, _embedded(path, 0.5)),
            "rs": _pooled(rs_det.analyse_plane, _embedded(path, 0.5)),
        }
        for m in clean:
            assert half[m] > clean[m] + 0.15, (path.name, m, clean[m], half[m])


def test_chi2_and_spa_are_roughly_calibrated():
    """On real photographs, the 25 % and 50 % payloads are recovered closely."""
    for path in REFERENCE:
        for rate in (0.25, 0.5):
            planes = _embedded(path, rate)
            for det, name in ((chi2_det.analyse_plane, "chi2"),
                              (spa_det.analyse_plane, "spa")):
                got = _pooled(det, planes)
                assert abs(got - rate) < 0.12, f"{name} on {path.name}: {got:.3f} vs {rate}"


def test_rs_biases_low_at_half_capacity():
    for path in REFERENCE:
        got = _pooled(rs_det.analyse_plane, _embedded(path, 0.5))
        assert got < 0.45, f"RS read {got:.3f} at a true 50 % payload on {path.name}"
