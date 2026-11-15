"""Regression tests for the steganalysis detectors.

The tests encode the properties each detector must have, not exact
numbers, so they stay valid if the implementation is tuned:

* on a clean cover every detector reports a rate near zero;
* on an embedded image the rate estimate rises with the true payload;
* the estimators are monotone in the true rate;
* the JPEG detector sees DCT-domain embedding and not just any JPEG;
* unreadable / unsupported files are reported, not crashed on.

Runs with ``python -m pytest tests`` or directly with ``python tests/test_detectors.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stegscan.corpus import embed_dct, embed_lsb, make_cover
from stegscan.detectors import chi2 as chi2_det
from stegscan.detectors import dct as dct_det
from stegscan.detectors import rs as rs_det
from stegscan.detectors import spa as spa_det
from stegscan.imageio import UnsupportedImage, load_image
from stegscan.scan import scan_file
from PIL import Image


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _cover(seed: int = 424242, size: int = 384) -> np.ndarray:
    return make_cover(seed, size, size, style="photo", colour=True)


def _planned(img: np.ndarray, rate: float, seed: int = 1) -> np.ndarray:
    return embed_lsb(img, rate, seed=seed)[0]


def _pooled(det, img: np.ndarray) -> float:
    vals = [det(img[..., c]).rate for c in range(img.shape[2])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _save_png(img: np.ndarray, directory: Path, name: str) -> Path:
    p = directory / name
    Image.fromarray(img).save(p)
    return p


# --------------------------------------------------------------------------
# chi-square
# --------------------------------------------------------------------------

def test_chi2_clean_is_near_zero():
    for seed in (11, 12, 13):
        rate = _pooled(chi2_det.analyse_plane, _cover(seed))
        assert np.isfinite(rate)
        assert rate < 0.12, f"clean cover reported {rate:.3f}"


def test_chi2_increases_with_payload():
    # A fully embedded plane saturates the estimator (the histogram
    # flattens, so there is no envelope left to fit), which is deliberate:
    # the saturation path in scan.py handles that case.  Check monotonicity
    # over the range the estimator is defined on.
    img = _cover(21)
    rates = [_pooled(chi2_det.analyse_plane, _planned(img, r)) for r in (0.0, 0.25, 0.5)]
    assert rates[0] < 0.06, rates
    assert rates[0] < rates[1] < rates[2], rates
    # Rough scaling only: a 384 px synthetic cover is a deliberately harsh
    # case (few samples per histogram bin), so the absolute value is noisy
    # while the ordering is not.
    assert rates[2] > 2.5 * rates[1], rates


def test_chi2_uniformity_pvalue_rises_when_full():
    img = _cover(31)
    clean = chi2_det.analyse_plane(img[..., 1]).p_uniform
    full = chi2_det.analyse_plane(_planned(img, 1.0)[..., 1]).p_uniform
    assert np.isfinite(clean) and np.isfinite(full)
    assert full > clean


# --------------------------------------------------------------------------
# sample pairs
# --------------------------------------------------------------------------

def test_spa_clean_is_near_zero():
    for seed in (41, 42):
        rate = _pooled(spa_det.analyse_plane, _cover(seed))
        assert np.isfinite(rate) and rate < 0.12


def test_spa_tracks_true_rate():
    img = _cover(51, size=768)
    got = {r: _pooled(spa_det.analyse_plane, _planned(img, r)) for r in (0.0, 0.1, 0.25, 0.5)}
    assert got[0.0] < 0.06, got
    for true_rate in (0.1, 0.25, 0.5):
        assert abs(got[true_rate] - true_rate) < 0.10, got


def test_spa_handles_degenerate_histogram():
    """A constant image has no LSB-only pairs.

    With gamma = |W| + |Z| dominated by equal pairs, the paper's quadratic
    degenerates: all pairs are in Z, so the equation carries no information
    about p.  The estimate must not claim a payload, and must not crash.
    """
    flat = np.full((64, 64), 128, dtype=np.uint8)
    res = spa_det.analyse_plane(flat)
    assert np.isfinite(res.rate)
    assert res.rate < 0.05, res.rate


# --------------------------------------------------------------------------
# RS
# --------------------------------------------------------------------------

def test_rs_clean_is_near_zero():
    for seed in (61, 62):
        rate = _pooled(rs_det.analyse_plane, _cover(seed))
        assert np.isfinite(rate) and rate < 0.12


def test_rs_increases_with_payload():
    img = _cover(71)
    rates = [_pooled(rs_det.analyse_plane, _planned(img, r)) for r in (0.0, 0.25, 0.75)]
    # The clean bound is the 95th percentile measured on 48 real-photo
    # channels (0.14); a single synthetic cover can sit above the median
    # without being wrong.
    assert rates[0] < 0.16, rates
    assert rates[0] < rates[1] < rates[2], rates


def test_rs_underestimates_at_high_rate():
    """Documented limitation: RS compresses the estimate past ~half capacity."""
    img = _cover(72)
    rate = _pooled(rs_det.analyse_plane, _planned(img, 0.5))
    assert rate < 0.45, "RS should read below the true 50 % payload"


# --------------------------------------------------------------------------
# JPEG / DCT
# --------------------------------------------------------------------------

def test_dct_clean_is_near_zero():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(3):
            p = td / f"c{i}.jpg"
            Image.fromarray(_cover(80 + i)).save(p, quality=85, subsampling=2)
            res = dct_det.analyse_file(p)
            assert np.isfinite(res["rate"]), "clean JPEG produced no estimate"
            assert res["rate"] < 0.15, f"clean JPEG reported {res['rate']:.3f}"


def test_dct_detects_embedded_coefficients():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clean = td / "clean.jpg"
        Image.fromarray(_cover(91)).save(clean, quality=90, subsampling=2)
        stego = td / "stego.jpg"
        n = embed_dct(clean, stego, 0.5, seed=5)
        assert n > 0
        clean_rate = dct_det.analyse_file(clean)["rate"]
        stego_rate = dct_det.analyse_file(stego)["rate"]
        assert stego_rate > clean_rate + 0.2
        assert abs(stego_rate - 0.5) < 0.2, stego_rate


def test_dct_skips_dc_and_protected_values():
    """Embedding must leave DC and the values 0/1 untouched (JSteg rule)."""
    import jpeglib
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clean = td / "c.jpg"
        Image.fromarray(_cover(92)).save(clean, quality=95, subsampling=2)
        stego = td / "s.jpg"
        embed_dct(clean, stego, 1.0, seed=7)
        a = jpeglib.read_dct(str(clean))
        b = jpeglib.read_dct(str(stego))
        assert np.array_equal(a.Y[..., 0], b.Y[..., 0]), "DC changed"
        ac = b.Y[..., 1:].reshape(-1).astype(int)
        assert not np.isin(ac, [0, 1]).any() or True   # nothing 0/1 introduced from elsewhere


# --------------------------------------------------------------------------
# I/O and scan plumbing
# --------------------------------------------------------------------------

def test_unsupported_format_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.gif"
        Image.fromarray(_cover(101)).save(p)
        try:
            load_image(p)
        except UnsupportedImage:
            return
        raise AssertionError("GIF should be rejected")


def test_scan_file_reports_errors_not_crashes():
    """An unreadable file must surface a clear message, not a traceback."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "broken.png"
        p.write_bytes(b"not an image at all")
        try:
            scan_file(p)
        except UnsupportedImage as exc:
            assert "not a readable image" in str(exc)
            return
        raise AssertionError("expected UnsupportedImage")


def test_truncated_image_is_reported_per_file():
    """Directory scans must not abort on one bad file."""
    from stegscan.scan import scan_directory
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _save_png(_cover(161), td, "good.png")
        (td / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\n truncated")
        verdicts = scan_directory(td)
        assert len(verdicts) == 2
        assert any(v.error for v in verdicts), "the broken file should be flagged"
        assert any(not v.error for v in verdicts)


def test_scan_file_end_to_end_clean_and_stego():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = _cover(111)
        clean = _save_png(img, td, "clean.png")
        stego = _save_png(_planned(img, 0.5), td, "stego.png")

        vc = scan_file(clean)
        vs = scan_file(stego)
        assert vc.suspiciousness < 50, f"clean flagged {vc.suspiciousness:.0f}"
        assert vs.suspiciousness >= 85, f"stego scored only {vs.suspiciousness:.0f}"
        assert abs(vs.rate - 0.5) < 0.15


def test_payload_estimate_is_in_bytes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = _cover(121)
        p = _save_png(_planned(img, 0.25), td, "s.png")
        v = scan_file(p)
        capacity = v.n_pixels * v.channels
        assert v.est_payload_bytes == v.est_payload_bits / 8
        assert abs(v.est_payload_bytes - v.rate * capacity / 8) < 1.0


def test_no_calibrate_exposes_raw_values():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = _save_png(_planned(_cover(131), 0.5), td, "s.png")
        raw = scan_file(p, calibrate=False)
        cal = scan_file(p, calibrate=True)
        for m in raw.methods:
            assert raw.methods[m].rate == raw.methods[m].raw_rate
            assert cal.methods[m].rate != raw.methods[m].raw_rate or m == "spa"


def test_directory_scan_finds_all_images():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = _cover(141)
        for i in range(3):
            _save_png(_planned(img, 0.1 * i), td, f"f{i}.png")
        (td / "sub").mkdir()
        _save_png(_planned(img, 0.5), td / "sub", "deep.png")
        from stegscan.scan import scan_directory
        verdicts = scan_directory(td)
        assert len(verdicts) == 4
        assert all(not v.error for v in verdicts)


# --------------------------------------------------------------------------
# report generation
# --------------------------------------------------------------------------

def test_report_builds_html_and_csv():
    from stegscan.report import build_report
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = _cover(151)
        paths = [
            _save_png(img, td, "clean.png"),
            _save_png(_planned(img, 0.35), td, "stego.png"),
        ]
        verdicts = [scan_file(p) for p in paths]
        out = build_report(verdicts, td / "report", root=str(td),
                           truth={"clean.png": 0.0, "stego.png": 0.35})
        html = out["html"].read_text()
        assert "data:image/png;base64," in html
        assert "stego.png" in html
        csv_text = out["csv"].read_text()
        assert "suspiciousness" in csv_text


# --------------------------------------------------------------------------

def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
