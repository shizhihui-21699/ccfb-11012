# stegscan

LSB / DCT steganalysis for PNG, BMP and JPEG images: four detectors, a
quantitative payload estimate per file, and a batch report with heatmaps.

Built for triage: point it at a directory, get a ranked list of files that
probably carry embedded data, an estimate of *how much*, and the evidence
behind each call.

```
python scan.py /evidence/handoff --out /evidence/report
```

## What it does

| Format | Analysis domain | Detectors |
|--------|-----------------|-----------|
| PNG, BMP | 8-bit spatial planes, per channel | chi-square, sample pairs (SPA), RS |
| JPEG | quantised DCT coefficients (Y/Cb/Cr) | DCT histogram (JSteg-style LSB replacement) |

For each file the scanner reports:

* **estimated embedding rate** — payload bits ÷ available LSB capacity,
  in [0, 1]. 0.25 means a quarter of every analysed channel's LSB plane is
  payload;
* **estimated payload size** in bytes;
* **suspiciousness**, 0–100, derived from the rate and attenuated when the
  detectors disagree;
* per-detector detail: calibrated rate, raw estimator output, number of
  histogram pairs / groups behind it, the classic chi-square p-value, and
  any model-failure note.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pillow matplotlib jpeglib
```

`jpeglib` (DCT access) is only needed for JPEGs; the spatial detectors run
without it. If it is missing, JPEG files are reported with an explanatory
error rather than silently skipped.

## Usage

```bash
# generate a labelled test corpus, then scan it and benchmark the result
python scan.py --make-testset /tmp/stegtest --out /tmp/stegreport

# scan your own directory (recursively)
python scan.py /evidence/handoff --out /evidence/report --title "Case 4471"

# report raw estimator output instead of calibrated rates
python scan.py /evidence/handoff --out /tmp/raw --no-calibrate
```

Outputs, in `--out`:

* `report.html` — self-contained (figures inlined as base64): summary
  cards, rate heatmap, suspiciousness heatmap, findings table, and — when
  the directory has a `truth.json`, i.e. a test run — an
  estimate-vs-truth scatter and a detection-rate curve;
* `findings.csv` — one row per file, every per-detector number;
* `summary.json` — counts and mean absolute error (test runs only).

### Library use

```python
from stegscan.scan import scan_file, scan_directory
from stegscan.report import build_report

verdicts = scan_directory("/evidence/handoff")
v = verdicts[0]
print(v.rate, v.est_payload_bytes, v.suspiciousness, v.verdict)
for name, m in v.methods.items():
    print(name, m.rate, m.raw_rate, m.n_obs, m.note)

build_report(verdicts, "report", root="/evidence/handoff")
```

## How the detectors work

### chi-square (`stegscan/detectors/chi2.py`)

Two tests on the 256-bin value histogram.

*The classic attack* (Westfeld & Pfitzmann). Under LSB replacement, values
`2k` and `2k+1` should end up equally frequent. A chi-square
goodness-of-fit test against that "flattened pairs" model gives a p-value
near 1 on a fully embedded plane and near 0 on an untouched photograph.

*The quantitative estimate.* The pair difference `d_i = h(2i) − h(2i+1)`
shrinks by exactly `(1 − p)` under LSB replacement, while the pair total
`S_i = h(2i) + h(2i+1)` is invariant. Regressing `d` on a predictor `D`
built from the local slope of `S` therefore yields `beta = 1 − p`.

Since `S` is invariant, the predictor can be computed from the stego image
alone — no cover needed. Two practical points:

* the regression is weighted by `1/S` (variance of `d` is ≈ `S`), otherwise
  the sparse histogram tail — where the slope is mostly counting noise —
  dominates;
* the estimate is computed on a grid of spatial tiles and combined by
  median. A single global fit is noisy at low payloads (a 900×900 image has
  only a few thousand samples per bin); the median over tiles is markedly
  more stable. The grid is capped at 4×4 and each tile must be at least
  256 px on a side — splitting a 512 px image into 4×4 leaves 128 px tiles
  whose per-bin counts are too small for the slope fit to mean anything
  (measured: the estimate collapses toward zero even at a 25 % payload).
* pairs outside the histogram's populated range are excluded, together with
  a 3-pair margin. A pair whose total falls below the counting threshold
  carries no usable slope, and the central difference at the edge of the
  populated range instead reads the drop to zero and reports a huge slope.

A gate rejects the estimate when `d` and `D` are uncorrelated *and* the
predictor still carries signal — that pattern means the content is not
smooth enough for the model (halftones, dithering, posterised gradients),
not that a payload is present. When the gate fires, chi-square reports no
numeric estimate instead of a wrong one.

### sample pairs (`stegscan/detectors/spa.py`)

Dumitrescu, Wu & Memon's sets over horizontal and vertical adjacent pairs:

```
X = { v even and u < v, or v odd and u > v }
Y = { v even and u > v, or v odd and u < v }
W = { (u,v) ∈ Y : u and v differ only in the LSB }     (|u-v| = 1)
Z = { u = v }
```

With `gamma = |W| + |Z|` and `P` pairs, the rate is the smaller root of

```
0.5·gamma·p² + (2|X| − |P|)·p + (|Y| − |X|) = 0
```

`W` must be the *subset of Y* with `|u−v| = 1`; counting every adjacent pair
with `|u−v| = 1` adds pairs that belong to X and breaks the algebra.

**Known limitation, measured.** SPA's Assumption 1 is `|X| = |Y|`. In 48
clean real-photo channels the residual `(Y−X)/P` ranges from +0.0001 to
+0.0100, and on a typical plane a 1 % residual alone produces an apparent
rate of ~0.10. SPA's clean-image 95th percentile is therefore ~0.17, much
wider than chi-square's ~0.10. The detector reports this baseline
(`details["assumption_baseline"]`) and notes when the estimate sits below
it.

### RS (`stegscan/detectors/rs.py`)

Fridrich, Goljan & Du. Disjoint 2×2 groups, discrimination function
`f(G) = Σ|x_{i+1} − x_i|`, two flip operators (LSB flip and shifted flip),
regular/singular counts for the mask, its complement, and the all-ones
mask. The rate is the root of

```
2(d1 + d0)·x² + (d₋0 − d₋1 − d1 − 3d0)·x + d0 − d₋0 = 0,   p = |x / (x − ½)|
```

**Known limitation.** RS is reliable at low rates and compresses steadily
above ~20 % of capacity: at a true 50 % it reads ~0.36 on real photographs.
The calibration curve corrects for the average compression, but the
per-image error at high rates is real.

### DCT histogram (`stegscan/detectors/dct.py`)

JSteg and relatives replace the LSB of quantised DCT coefficients, skipping
the values 0 and 1. Within each pair `(2k, 2k+1)` and `(−2k, −2k−1)` the
operation is closed, so

```
S = h(2k) + h(2k+1)      unchanged
d = h(2k) − h(2k+1)      multiplied by (1 − p)
```

The pairs containing the protected values 0, 1, −1, −2 are excluded — those
coefficients are never modified, so their differences are not scaled.

The cover histogram is unknown, so it is estimated from the image itself:
crop 4 pixels (deliberately not a multiple of 8, which shifts the 8×8 block
grid), recompress with the file's *own* quantisation tables, and use that
histogram as the cover reference. Regressing the observed pair differences
on the calibrated ones gives `(1 − p)` directly. The estimate is produced
per component and pooled by coefficient count.

## Calibration (`stegscan/calibration.py`)

Raw estimator output is mapped to a true rate through measured curves.
They were fitted on **16 real photographs** (public-domain Commons images,
all channels, capacity-pooled) with `tools/calibrate.py`, because that is
the reference that matters. Measured raw medians:

| true rate | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 | 1.00 |
|-----------|------|------|------|------|------|------|
| chi-square | 0.075 | 0.107 | 0.268 | 0.466 | 0.714 | 0.883 |
| sample pairs | 0.055 | 0.105 | 0.258 | 0.502 | 0.758 | 0.730 |
| RS | 0.042 | 0.084 | 0.195 | 0.364 | 0.545 | 0.866 |

Chi-square and SPA need only gentle correction; RS needs a real curve. Note
SPA's last column: it *flattens* at full capacity (0.730, not 1.0), which is
why a saturated SPA is excluded from the consensus rather than trusted.
`scan_file(..., calibrate=False)` exposes the raw numbers, and the report
records which mode was used. Re-run `python -m tools.calibrate --write`
after changing an estimator, and update the tables it prints.

## Measured performance

On 16 real photographs, three PNG/BMP runs each (payload scattered
uniformly across all channels):

| true rate | median estimate | MAE | flagged (score ≥ 50) |
|-----------|-----------------|-----|----------------------|
| 0.00 (clean) | 0.000 | — | 0 / 16 |
| 0.05 | 0.047 | 0.020 | 1 / 16 |
| 0.10 | 0.094 | 0.024 | 7 / 16 |
| 0.25 | 0.242 | 0.029 | 16 / 16 |
| 0.50 | 0.503 | 0.022 | 16 / 16 |
| 0.75 | 0.751 | 0.018 | 16 / 16 |
| 1.00 | 0.996 | 0.036 | 16 / 16 |

**The detection floor is real.** Below ~10 % of capacity the structural
detectors lose power: the payload changes too few pixel pairs to outvote
the image's own statistics. At 5 % of capacity only 1 of 16 photographs is
flagged, and at 2 % none are. "Weak / inconclusive" is the honest answer
for a short message, not a threshold to tune away.

The consensus rule is the mean of the two lowest non-saturated detector
estimates, which requires two detectors to agree before a rate is
reported. That is what keeps the clean false-positive count at zero: a
single detector's model failure (a halftone, a dithered painting) is
outvoted, while a real payload raises all three. A detector that has
stopped tracking the payload (SPA past ~half capacity, chi-square on a
fully embedded plane) is flagged as saturated and excluded rather than
allowed to drag the consensus down.

For JPEGs, DCT-domain embedding at 25 % of coefficient capacity is detected
in every case tested, and lower rates progressively less reliably.

## Test corpus (`stegscan/corpus.py`)

`--make-testset` writes a labelled set: photographic and hard-texture
covers at several sizes, clean controls, payloads from 2 % to 100 % of
capacity in two embedding patterns (sequential fill and scattered by key),
as PNG and BMP, plus a JPEG arm with JSteg-style DCT embedding at several
qualities. `truth.json` records every file's true rate so the report can
plot estimate-vs-truth.

The synthetic covers are a **stress test, not a calibration source**: they
are smoother in the histogram than photography, and at small sizes the
per-bin counts are low enough that the estimators are visibly noisier. The
tests in `tests/test_real_photos.py` use four public-domain photographs in
`tests/data/` to keep the realistic case anchored.

## Tests

```bash
.venv/bin/python tests/test_detectors.py      # 20 property tests
.venv/bin/python -m pytest tests              # same, under pytest
```

The tests assert properties (clean ≈ 0, monotone in payload, JPEG detector
sees DCT-domain embedding and not just any JPEG, per-file errors do not
abort a directory scan), not exact numbers.

## Limitations

* **Short payloads.** Under ~10 % of capacity, and especially under ~2 %,
  detection is unreliable. This is a property of the detectors, not a
  tuning choice.
* **RS at high rates** underestimates; the consensus median and the
  calibration curve compensate on average, not per image.
* **SPA's baseline.** Its quadratic is sensitive to `|X| ≠ |Y|`; a 1 %
  content-induced imbalance mimics ~10 % embedding. Readings near or below
  the reported baseline should not be treated as evidence.
* **Spatial payloads in JPEG** are out of scope: re-encoding destroys them,
  and the DCT detector targets coefficient-domain embedding only.
* **Not a classifier.** The tool estimates how much LSB-domain structure is
  present. It does not identify the embedding tool, recover a key, or
  extract a payload. Tune the suspiciousness threshold to your own
  false-positive tolerance.
* **Alpha channels** are not analysed (dropped at load). A payload hidden
  in an alpha plane needs a different detector.
