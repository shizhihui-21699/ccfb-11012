#!/usr/bin/env python3
"""Measure the raw-estimator -> true-rate curves on a generated corpus.

Usage:
    python -m tools.calibrate            # default sweep
    python -m tools.calibrate --write    # print a calibration.py body

The scanner's default curves were produced by this script.  Re-run it if
you change an estimator or a detector's internal parameters.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stegscan.corpus import build_corpus            # noqa: E402
from stegscan.scan import scan_file, scan_spatial   # noqa: E402
from stegscan.imageio import load_image             # noqa: E402
import stegscan.detectors.dct as dct_det            # noqa: E402

RATES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
METHODS = ("chi2", "spa", "rs")


def measure_spatial(n_covers=10, seed=101, size=512):
    with tempfile.TemporaryDirectory() as td:
        samples = build_corpus(
            td, n_covers=n_covers, rates=RATES[1:], seed=seed, size=size,
            styles=("photo", "texture"), formats=("png",),
            embeddings=("sequential", "scattered"),
            jpeg_rates=(), jpeg_qualities=(),
        )
        rows = collections.defaultdict(lambda: collections.defaultdict(list))
        for s in samples:
            v = scan_file(s.path)
            key = (s.family, round(s.true_rate, 3))
            for m in METHODS:
                rows[key][m].append(v.methods[m].raw_rate)
    return rows


def measure_dct(n_covers=8, seed=202, size=512, qualities=(75, 85, 95),
                rates=(0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)):
    with tempfile.TemporaryDirectory() as td:
        samples = build_corpus(
            td, n_covers=0, rates=(), seed=seed, size=size,
            styles=("photo",), formats=(), jpeg_rates=(),
            jpeg_qualities=qualities, clean_controls=False,
        )
        # build_corpus writes clean JPEGs + embedded twins only for rates in
        # jpeg_rates; extend by embedding here for the full grid.
        from PIL import Image
        from stegscan.corpus import make_cover, embed_dct
        rows = collections.defaultdict(list)
        for i in range(n_covers):
            cover = make_cover(seed * 1000 + 300 + i, size, size, style="photo")
            src = Path(td) / f"c{i}.jpg"
            Image.fromarray(cover).save(src, quality=85, subsampling=2)
            for q in qualities:
                base = Path(td) / f"q{q}_c{i}.jpg"
                Image.fromarray(cover).save(base, quality=q, subsampling=2)
                rows[0.0].append(dct_det.analyse_file(base)["rate"])
                for r in rates[1:]:
                    out = Path(td) / f"q{q}_c{i}_r{int(r*100)}.jpg"
                    embed_dct(base, out, r, seed=seed + i * 17 + q)
                    rows[round(r, 3)].append(dct_det.analyse_file(out)["rate"])
    return rows


def summarize(rows, methods=METHODS):
    """Median raw estimate per (family, true rate), plus a pooled column.

    The pooled column is the median over *all* embeddings at that true rate,
    which is what the scanner must invert: at scan time the embedding
    scheme is unknown.
    """
    out = {}
    for (fam, rate), d in sorted(rows.items()):
        out.setdefault(fam, {})[rate] = {
            m: float(np.nanmedian(d[m])) for m in methods if d.get(m)
        }
    pooled = {}
    for (fam, rate), d in rows.items():
        for m in methods:
            if d.get(m):
                pooled.setdefault(rate, {}).setdefault(m, []).append(d[m])
    for rate, by_method in pooled.items():
        out.setdefault("pooled", {})[rate] = {
            m: float(np.nanmedian(v)) for m, v in by_method.items()
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="emit calibration.py curves")
    ap.add_argument("--json", type=Path, help="write raw measurements here")
    args = ap.parse_args()

    sp = summarize(measure_spatial())
    dc = measure_dct()

    result = {"spatial": sp, "dct": {str(k): float(np.nanmedian(v)) for k, v in sorted(dc.items())}}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))

    for fam, by_rate in sp.items():
        print(f"== {fam}")
        for rate, vals in by_rate.items():
            print(f"  true={rate:4.3f} " + "  ".join(f"{m}={v:.3f}" for m, v in vals.items()))
    print("== dct")
    for rate, v in result["dct"].items():
        print(f"  true={rate:>5} raw={v:.3f}")

    if args.write:
        print("\n# Calibration points: raw estimate -> true rate (pooled families).")
        for m in METHODS:
            by_rate = sp["pooled"]
            xs = [round(by_rate[r][m], 4) for r in sorted(by_rate) if m in by_rate[r]]
            ys = [r for r in sorted(by_rate) if m in by_rate[r]]
            print(f"{m}: x={xs}\ny={ys}")
        print(f"dct: x={[round(result['dct'][k], 4) for k in sorted(result['dct'])]}"
              f"\ny={[float(k) for k in sorted(result['dct'])]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
