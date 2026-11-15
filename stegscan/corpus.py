"""Synthetic test-corpus generation.

Two kinds of cover are produced:

* ``photo``  -- multi-octave value noise with a tone curve and film grain,
  which reproduces the smooth histograms and pixel correlation of a
  photograph well enough that the detectors' assumptions hold;
* ``texture`` -- directional structure (stripes/edges) plus texture, i.e.
  the "hard" case where pair statistics are less well behaved.

Payloads are embedded three ways, because a detector that only works for
one embedding scheme is not much use:

* ``sequential``  -- fill the LSB plane from the first pixel onward
  (the naive case; chi-square's blockwise variant targets it);
* ``scattered``   -- pseudo-random positions driven by a key
  (the usual real-world case);
* ``channel``     -- spread over the colour channels as one continuous
  stream, which is how most tools actually do it.

Everything is deterministic: a seed reproduces the exact corpus, which is
what makes the calibration curve and the regression tests meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# cover synthesis
# --------------------------------------------------------------------------

def _bilinear(g: np.ndarray, h: int, w: int) -> np.ndarray:
    gh, gw = g.shape
    ys = np.linspace(0, gh - 1, h)
    xs = np.linspace(0, gw - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx
    bot = g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _fractal(rng: np.random.Generator, h: int, w: int,
             base: tuple[int, int] = (4, 4), octaves: int = 7,
             decay: float = 0.6) -> np.ndarray:
    v = np.zeros((h, w))
    amp = 1.0
    for o in range(octaves):
        gh = min(h, max(2, base[0] * 2 ** o))
        gw = min(w, max(2, base[1] * 2 ** o))
        v += amp * _bilinear(rng.random((gh, gw)), h, w)
        amp *= decay
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else v


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.2:
        return img
    radius = max(1, int(3 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(xs ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


def _pink_noise(rng: np.random.Generator, h: int, w: int,
                beta: float = 1.0) -> np.ndarray:
    """1/f^beta noise via spectral synthesis, normalised to [0, 1].

    Value-noise fractals roll off far too fast above a few cycles per
    pixel: their 8x8 DCT blocks are almost all zero, which makes the JPEG
    detector see nothing to analyse.  Real photographs keep roughly 1/f
    amplitude up to Nyquist, so a spectral synthesiser is used for the
    fine detail and the low-frequency shape is composed separately.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = f[0, 1] if w > 1 else 1.0
    amp = np.zeros_like(f)
    nz = f > 0
    amp[nz] = 1.0 / (f[nz] ** beta)
    # random phases -> Gaussian field
    phase = rng.uniform(0, 2 * np.pi, (h, w))
    spec = amp * np.exp(1j * phase)
    spec[0, 0] = 0.0
    field = np.real(np.fft.ifft2(spec))
    lo, hi = field.min(), field.max()
    return (field - lo) / (hi - lo) if hi > lo else field


def _detail(rng: np.random.Generator, h: int, w: int,
            strength: float = 1.0) -> np.ndarray:
    """Multi-scale detail with roughly photographic frequency content."""
    fine = _pink_noise(rng, h, w, beta=rng.uniform(0.9, 1.4))
    coarse = _fractal(rng, h, w, base=(3, 3), octaves=6, decay=0.65)
    mix = 0.75 * fine + 0.25 * coarse
    lo, hi = mix.min(), mix.max()
    mix = (mix - lo) / (hi - lo) if hi > lo else mix
    return strength * mix


def _luminance(rng: np.random.Generator, h: int, w: int, style: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(float)

    if style == "photo":
        kind = int(rng.integers(0, 5))
        if kind == 0:                       # horizon / gradient sky
            img = 0.25 + 0.55 * (yy / h)
            ground = _detail(rng, h, w, 1.0)
            weight = 1.0 - 0.6 * np.exp(-((yy - 0.7 * h) / (0.2 * h)) ** 2)
            img = img + 0.45 * ground * weight
        elif kind == 1:                     # 1/f texture (foliage, rock)
            img = _detail(rng, h, w, 1.0)
        elif kind == 2:                     # soft blobs + fine detail
            img = np.zeros((h, w))
            for _ in range(int(rng.integers(10, 28))):
                cy, cx = rng.random() * h, rng.random() * w
                sy = rng.uniform(0.02, 0.14) * h
                sx = rng.uniform(0.02, 0.14) * w
                img += rng.normal(0.0, 0.5) * np.exp(
                    -(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2)
            img = 0.5 + img + 0.18 * (_detail(rng, h, w) - 0.5)
        elif kind == 3:                     # architecture / edges
            ang = rng.uniform(0, math.pi)
            period = rng.uniform(10, 45)
            img = 0.5 + 0.32 * np.sin(
                2 * math.pi * (xx * math.cos(ang) + yy * math.sin(ang)) / period)
            img = _blur(img, rng.uniform(0.6, 1.6)) + 0.30 * (_detail(rng, h, w) - 0.5)
        else:                               # vignette + texture
            cy, cx = h * rng.uniform(0.3, 0.7), w * rng.uniform(0.3, 0.7)
            r = np.sqrt(((yy - cy) / (0.6 * h)) ** 2 + ((xx - cx) / (0.6 * w)) ** 2)
            img = 0.7 * np.exp(-r * r) + 0.45 * (_detail(rng, h, w) - 0.5) + 0.25
    elif style == "texture":
        # Deliberately "hard" content: strong directional structure and
        # coarse-grain noise.  Unlike the earlier pure-sinusoid variant,
        # several scales are mixed so the value histogram still spans the
        # whole range (a narrow, comb-like histogram is not a photograph
        # and makes histogram-based estimators meaningless).
        kind = int(rng.integers(0, 3))
        if kind == 0:
            img = _fractal(rng, h, w, base=(2, 2), octaves=9, decay=0.72)
            img = img + 0.35 * (_detail(rng, h, w, 1.0) - 0.5)
        elif kind == 1:
            ang = rng.uniform(0, math.pi)
            freq = rng.uniform(0.04, 0.2)
            patch = np.sin(freq * (xx * math.cos(ang) + yy * math.sin(ang)))
            patch = patch * _fractal(rng, h, w, base=(4, 4), octaves=6, decay=0.6)
            img = 0.5 + 0.45 * patch + 0.25 * rng.normal(0, 1, (h, w))
            img = _blur(img, 0.7)
        else:
            img = _fractal(rng, h, w, base=(16, 16), octaves=7, decay=0.6)
            img = _blur(img + 0.35 * rng.normal(0, 1, (h, w)), 1.0)
    else:
        raise ValueError(f"unknown cover style {style!r}")

    img = np.clip(img, 0.0, 1.0)
    # Stretch contrast to a photographic range.  The compositions above are
    # normalised per-component, which leaves them far flatter than a real
    # photograph (measured: std ~18 grey levels versus ~60 for the reference
    # photos).  Flat images have almost no 8x8 DCT energy, so a JPEG made
    # from one is not a useful test of the DCT detector.
    img = 0.5 + (img - 0.5) * rng.uniform(2.0, 3.6)
    img = np.clip(img, 0.0, 1.0)
    img = img ** rng.uniform(0.85, 1.15)
    img = img * rng.uniform(0.85, 1.05) + rng.uniform(0.03, 0.15)

    # Soft-knee tone mapping to [0, 255].  A hard clip would pile tens of
    # thousands of pixels onto bins 0/255, which no photograph does: the
    # value histogram would show a cliff, and every histogram-based
    # estimator would be measuring that cliff rather than the content.
    img = _soft_knee(img, lo=0.02, hi=0.98)
    img = img * 235.0 + 8.0 + rng.normal(0, rng.uniform(1.0, 3.0), (h, w))
    return _hist_reshape(img, rng, style)


def _hist_reshape(img: np.ndarray, rng: np.random.Generator,
                  style: str) -> np.ndarray:
    """Flatten any dominant histogram mode, then re-quantise.

    The composed fields are dominated by a handful of large low-frequency
    terms, so after the tone curve a large share of the pixels sits in a
    narrow band: measured 35-57 %% of pixels inside a 5-bin window around
    the mode, versus 3-10 %% in the reference photographs.  That spike is
    what makes a central-difference predictor read a huge slope and report
    a false payload.  A real photograph has a broad, slowly varying
    histogram, so the fix belongs in the generator, not in the detector.

    The remedy is a monotone histogram-equalising remap (blended towards
    the original to keep the tonal character), followed by a fresh grain
    pass, because quantising to 8 bits after a steep remap would otherwise
    leave comb artefacts of its own.
    """
    flat = img.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(flat.size)
    cdf = ranks / max(flat.size - 1, 1)

    # Equalised value, kept inside the same tonal span.
    lo, hi = np.percentile(flat, 0.5), np.percentile(flat, 99.5)
    eq = lo + cdf * (hi - lo)

    # Blend: pure equalisation loses the composition's tonal relationships,
    # so mix it with the original and let the sensor grain break up the
    # remaining quantisation steps.
    mix = rng.uniform(0.45, 0.75)
    out = (1.0 - mix) * flat + mix * eq

    # Re-centre and stretch to a photographic tonal range.  The input is
    # already on the 0..255 scale, so these are absolute grey levels.
    #
    # The *range* matters as much as the spread: a real photograph uses
    # almost every one of the 256 bins, whereas a std-only stretch leaves
    # ~100 bins empty at the ends (measured: 147-156 non-zero bins for the
    # composed fields against 256 for the reference photos).  Empty bins
    # remove exactly the histogram pairs the chi-square estimator regresses
    # on -- 75 usable pairs instead of 122 -- which is why this matters to
    # the detector and not just to the eye.
    #
    # A percentile stretch alone is used; a second pass normalising to a
    # target standard deviation would undo it (that was the earlier bug).
    lo_pct, hi_pct = rng.uniform(0.3, 1.5), rng.uniform(98.5, 99.7)
    lo_val, hi_val = np.percentile(out, [lo_pct, hi_pct])
    span = max(hi_val - lo_val, 1e-9)
    out = 4.0 + (out - lo_val) / span * 247.0

    # Triangular dither before the 8-bit quantisation.  Without it, a
    # rank-based remap leaves a systematic even/odd imbalance in the value
    # histogram (measured up to 6 %% against ~0.2 %% in real photographs):
    # ranks are integers, so consecutive ranks land on alternating
    # parities.  An LSB-replacement detector reads exactly that imbalance
    # as payload, so an undithered cover is a false-positive generator.
    out = out + rng.uniform(-1.0, 1.0, out.shape) + rng.uniform(-1.0, 1.0, out.shape)
    return np.clip(out, 0, 255).reshape(img.shape)


def _soft_knee(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compress out-of-range mass smoothly into (lo, hi) instead of clipping.

    Uses a scaled tanh, which is monotone and has no flat saturated tail.
    """
    span = hi - lo
    mid = (hi + lo) / 2.0
    x = (img - mid) / max(span / 2.0, 1e-9)
    return mid + (span / 2.0) * np.tanh(x)


def make_cover(seed: int, h: int = 512, w: int = 512, style: str = "photo",
               colour: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lum = _luminance(rng, h, w, style)
    if not colour:
        return lum.round().astype(np.uint8)[..., None]

    tint = rng.uniform(0.8, 1.0, size=3)
    chans = []
    for c in range(3):
        if rng.random() < 0.6:
            layer = _fractal(rng, h, w, base=(5, 5), octaves=7, decay=0.6)
        else:
            layer = np.zeros((h, w))
        chans.append(_soft_knee(
            lum * (0.75 + 0.25 * tint[c]) / 255.0 + 0.05 * (layer - 0.5),
            lo=0.01, hi=0.99) * 255.0)
    return np.clip(np.stack(chans, -1), 0, 255).round().astype(np.uint8)


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

def _payload_bits(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed + 10_007).integers(0, 2, n, dtype=np.uint8)


def embed_lsb(cover: np.ndarray, rate: float, seed: int = 0,
              mode: str = "scattered", channels: tuple[int, ...] | None = None
              ) -> tuple[np.ndarray, int]:
    """LSB-replace a fraction of the available capacity.  Returns (stego, bits).

    ``rate`` is the fraction of *total* LSB capacity used, i.e.
    payload bits / (pixels * channels).  This is the same convention the
    detectors report, so a scan estimate of 0.25 against a file generated
    with rate=0.25 is directly comparable.
    """
    arr = cover.copy()
    h, w = arr.shape[:2]
    n_channels = arr.shape[2]
    if channels is None:
        channels = tuple(range(n_channels))

    # Candidate positions, in channel-major raster order.
    positions = []
    for c in channels:
        positions.append(np.arange(h * w, dtype=np.int64) * n_channels + c)
    idx_all = np.concatenate(positions)

    nbits = int(round(rate * h * w * len(channels)))
    nbits = min(nbits, idx_all.size)
    if nbits == 0:
        return arr, 0

    flat = arr.reshape(-1)
    if mode == "sequential":
        sel = idx_all[:nbits]
    elif mode == "scattered":
        sel = np.random.default_rng(seed + 77).choice(idx_all, nbits, replace=False)
    elif mode == "interleaved":
        # Round-robin across channels, i.e. one bit per channel per pixel
        # until the payload is exhausted -- the pattern most embedders
        # produce when they write a byte across successive channels.
        per = int(math.ceil(nbits / len(channels)))
        order = []
        for c_i, c in enumerate(channels):
            order.append(np.arange(h * w, dtype=np.int64) * n_channels + c)
        pick = []
        remaining = nbits
        for row in range(per):
            for c_i in range(len(channels)):
                if remaining <= 0:
                    break
                pick.append(order[c_i][row])
                remaining -= 1
        sel = np.array(pick, dtype=np.int64)
    else:
        raise ValueError(f"unknown embedding mode {mode!r}")

    bits = _payload_bits(seed, nbits)
    flat[sel] = (flat[sel] & 0xFE) | bits
    return arr, nbits


def embed_dct(src: Path, dst: Path, rate: float, seed: int = 0) -> int:
    """JSteg-style LSB replacement in quantised DCT coefficients (skip 0, 1)."""
    import jpeglib

    j = jpeglib.read_dct(str(src))
    comps = [c.copy() for c in (j.Y, j.Cb, j.Cr)]
    rng = np.random.default_rng(seed + 31_337)

    total = 0
    for c in comps:
        flat = c.reshape(-1)
        ac_mask = np.ones(c.shape, dtype=bool)
        ac_mask[..., 0] = False                    # never touch DC
        v0 = flat.astype(np.int32)
        eligible = ac_mask.reshape(-1) & (v0 != 0) & (v0 != 1)
        idx = np.flatnonzero(eligible)
        nbits = int(round(rate * idx.size))
        if nbits == 0:
            continue
        sel = np.sort(rng.choice(idx, nbits, replace=False))
        bits = rng.integers(0, 2, nbits, dtype=np.int32)
        flat[sel] = ((v0[sel] & ~1) | bits).astype(flat.dtype)
        total += nbits

    j.Y, j.Cb, j.Cr = comps
    j.write_dct(str(dst), quality=-1)
    return total


# --------------------------------------------------------------------------
# corpus construction
# --------------------------------------------------------------------------

@dataclass
class Sample:
    path: Path
    true_rate: float          # fraction of LSB capacity (0 for covers)
    cover_seed: int
    family: str               # "clean" | "sequential" | "scattered" | "channel"
    style: str
    fmt: str

    @property
    def is_clean(self) -> bool:
        return self.true_rate == 0.0


def build_corpus(root: str | Path, *, n_covers: int = 12, rates: tuple[float, ...] = (
        0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00),
        seed: int = 2024, size: int = 512, styles: tuple[str, ...] = ("photo",),
        formats: tuple[str, ...] = ("png", "bmp"),
        embeddings: tuple[str, ...] = ("sequential", "scattered"),
        clean_controls: bool = True,
        jpeg_rates: tuple[float, ...] = (0.05, 0.25, 0.50, 0.90),
        jpeg_qualities: tuple[int, ...] = (75, 85, 95),
) -> list[Sample]:
    """Write a labelled test corpus below *root*.

    The returned list describes every file, so a test run can compare the
    scanner's estimate against the true rate.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    rng = np.random.default_rng(seed)

    for i in range(n_covers):
        style = styles[i % len(styles)]
        cover_seed = seed * 1000 + i
        cover = make_cover(cover_seed, size, size, style=style, colour=True)
        base = f"{style}_{i:03d}_s{cover_seed}"

        for fmt in formats:
            stem = root / f"{base}_clean.{fmt}"
            _save(cover, stem, fmt)
            if clean_controls:
                samples.append(Sample(stem, 0.0, cover_seed, "clean", style, fmt))
            # clean stego twins are regenerated from the same cover, so a
            # false positive on a *pair* is directly comparable.
            else:
                stem.unlink(missing_ok=True)

        for mode in embeddings:
            for rate in rates:
                stego, _ = embed_lsb(cover, rate, seed=cover_seed, mode=mode)
                for fmt in formats:
                    path = root / f"{base}_{mode}_r{int(rate*100):03d}.{fmt}"
                    _save(stego, path, fmt)
                    samples.append(Sample(path, rate, cover_seed, mode, style, fmt))

    # JPEG arm: encode a cover, then embed in the DCT domain and rewrite,
    # so the file keeps a valid JPEG structure with modified coefficients.
    for i in range(min(n_covers, 8)):
        style = styles[i % len(styles)]
        cover_seed = seed * 1000 + 100 + i
        cover = make_cover(cover_seed, size, size, style=style, colour=True)
        base = f"{style}_j{i:03d}_s{cover_seed}"
        for q in jpeg_qualities:
            clean = root / f"{base}_clean_q{q}.jpg"
            Image.fromarray(cover).save(clean, quality=q, subsampling=2)
            samples.append(Sample(clean, 0.0, cover_seed, "clean", style, "jpg"))
            for rate in jpeg_rates:
                stego = root / f"{base}_dct_q{q}_r{int(rate*100):03d}.jpg"
                embed_dct(clean, stego, rate, seed=cover_seed + q)
                samples.append(Sample(stego, rate, cover_seed, "dct", style, "jpg"))

    return samples


def _save(arr: np.ndarray, path: Path, fmt: str) -> None:
    img = Image.fromarray(arr)
    if fmt.lower() in ("jpg", "jpeg"):
        img.save(path, quality=95, subsampling=0)
    else:
        img.save(path)
