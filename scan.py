#!/usr/bin/env python3
"""Command-line front end for stegscan.

Examples
--------
Scan a directory and write the HTML/CSV report:

    python scan.py /evidence/handoff --out /evidence/report

Generate a labelled test corpus first, then scan and benchmark it:

    python scan.py --make-testset /tmp/stegtest --out /tmp/stegreport

Scan without applying the calibration curves (raw estimator output):

    python scan.py /evidence/handoff --out /tmp/raw --no-calibrate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stegscan.corpus import build_corpus
from stegscan.imageio import UnsupportedImage
from stegscan.report import build_report
from stegscan.scan import scan_directory, scan_file


def _progress(i: int, n: int, path: Path) -> None:
    end = "\n" if i == n else "\r"
    print(f"[{i:>4}/{n}] {path.name[:70]:<70}", end=end, file=sys.stderr, flush=True)


def _truth_from_manifest(root: Path) -> dict[str, float]:
    """Read the truth map written by a testset build (truth.json)."""
    import json
    p = root / "truth.json"
    if not p.exists():
        return {}
    return {k: float(v) for k, v in json.loads(p.read_text()).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scan.py",
        description="Detect LSB and DCT-domain steganography in PNG/BMP/JPEG images.",
    )
    ap.add_argument("path", nargs="?", type=Path,
                    help="directory (recursively scanned) or a single image")
    ap.add_argument("--out", type=Path, default=Path("stegscan_report"),
                    help="output directory for report.html / findings.csv "
                         "(default: ./stegscan_report)")
    ap.add_argument("--title", default=None, help="report title")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="report raw estimator output instead of calibrated rates")
    ap.add_argument("--no-progress", action="store_true", help="suppress progress output")

    test = ap.add_argument_group("test corpus")
    test.add_argument("--make-testset", type=Path, metavar="DIR",
                      help="generate a labelled test corpus in DIR, then scan it")
    test.add_argument("--n-covers", type=int, default=8,
                      help="number of cover images for --make-testset (default: 8)")
    test.add_argument("--size", type=int, default=512,
                      help="cover edge length for --make-testset (default: 512)")
    test.add_argument("--seed", type=int, default=2024,
                      help="corpus seed (default: 2024)")

    args = ap.parse_args(argv)

    if not args.make_testset and args.path is None:
        ap.error("either supply a path or use --make-testset DIR")

    truth: dict[str, float] = {}
    root = args.path
    title = args.title

    if args.make_testset:
        print(f"generating test corpus in {args.make_testset} ...", file=sys.stderr)
        samples = build_corpus(
            args.make_testset,
            n_covers=args.n_covers,
            rates=(0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00),
            seed=args.seed,
            size=args.size,
            styles=("photo", "texture"),
            formats=("png", "bmp"),
            embeddings=("sequential", "scattered"),
            jpeg_rates=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90),
            jpeg_qualities=(75, 85, 95),
        )
        import json
        truth = {s.path.name: s.true_rate for s in samples}
        (args.make_testset / "truth.json").write_text(json.dumps(truth, indent=2))
        print(f"  {len(samples)} files, truth.json written", file=sys.stderr)
        root = args.make_testset
        title = title or "Steganalysis report — labelled test corpus"
    elif root is not None and root.is_dir():
        truth = _truth_from_manifest(root)
    title = title or f"Steganalysis report — {root or args.make_testset}"

    calibrate = not args.no_calibrate
    if root is None:
        ap.error("no input path")

    if root.is_file():
        verdicts = [scan_file(root, calibrate=calibrate)]
        root_label = str(root.parent)
    else:
        verdicts = scan_directory(
            root, calibrate=calibrate,
            progress=None if args.no_progress else _progress,
        )
        root_label = str(root)

    if not verdicts:
        print("no supported images found", file=sys.stderr)
        return 2

    out = build_report(verdicts, args.out, root=root_label, title=title,
                       truth=truth or None, calibrate=calibrate)

    flagged = [v for v in verdicts if not v.error and v.suspiciousness >= 50]
    print(f"\nscanned {len(verdicts)} files, {len(flagged)} flagged", file=sys.stderr)
    for v in sorted(flagged, key=lambda v: -v.suspiciousness)[:10]:
        print(f"  {v.suspiciousness:5.0f}  {v.path.name}  "
              f"rate={v.rate:.3f}  ~{v.est_payload_bytes:,.0f} B", file=sys.stderr)
    print(f"report: {out['html']}", file=sys.stderr)
    print(f"csv:    {out['csv']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UnsupportedImage as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
