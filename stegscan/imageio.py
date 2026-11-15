"""Image loading and colour-plane handling.

Every detector works on individual 8-bit planes.  For colour images the
LSB plane of each channel is analysed separately; results are combined
by the scanner.  JPEG files are also opened as spatial planes here
(used only for a sanity check - the real JPEG analysis works in the DCT
domain).

Only PNG, BMP and JPEG are accepted.  Anything else is rejected with a
clear message so an operator knows to convert first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"PNG", "BMP", "JPEG"}


class UnsupportedImage(Exception):
    """Raised when a file is not a supported raster image."""


@dataclass
class LoadedImage:
    path: Path
    fmt: str                     # "PNG" | "BMP" | "JPEG"
    mode: str                    # PIL mode, e.g. "RGB", "L", "RGBA"
    planes: list[np.ndarray]     # one uint8 array per analysed channel
    plane_names: list[str]       # "Y"/"R"/"G"/"B"...
    width: int
    height: int
    bit_depth: int               # significant bits per sample
    file_size: int

    @property
    def n_pixels(self) -> int:
        return self.width * self.height

    @property
    def is_greyscale(self) -> bool:
        return len(self.planes) == 1


def _pil_mode_planes(img: Image.Image) -> tuple[list[np.ndarray], list[str], int]:
    """Split a PIL image into the 8-bit planes we analyse.

    Alpha channels are returned separately by the caller and dropped here:
    transparency is a legitimate part of the file, but payload hiding in
    the alpha plane is a different detector, so we do not pool it with
    colour-channel statistics.
    """
    mode = img.mode
    if mode in ("RGB", "RGBA"):
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return [arr[..., 0], arr[..., 1], arr[..., 2]], ["R", "G", "B"], 8
    if mode in ("L", "LA", "1"):
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return [arr], ["Y"], 8
    if mode == "P":
        # palettised PNG: the palette index plane is what an LSB payload
        # would touch only if the tool re-mapped colours; analysing the
        # rendered luminance is more useful in practice.
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return [arr[..., 0], arr[..., 1], arr[..., 2]], ["R", "G", "B"], 8
    if mode.startswith("I") or mode == "F":
        # 16-bit or float data: PNG-16 stores >8 bits, so LSB analysis on
        # the low 8 bits is still meaningful, but we must say so.
        arr = np.asarray(img)
        arr16 = (arr.astype(np.uint16) if arr.dtype != np.uint16 else arr)
        low = (arr16 & 0xFF).astype(np.uint8)
        return [low], ["Y"], 16
    if mode == "CMYK":
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return [arr[..., 0], arr[..., 1], arr[..., 2]], ["R", "G", "B"], 8
    raise UnsupportedImage(f"unsupported PIL mode {mode!r}")


def load_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    try:
        with Image.open(path) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in SUPPORTED_FORMATS:
                raise UnsupportedImage(
                    f"{path.name}: format {fmt or 'unknown'} is not supported "
                    f"(expected one of {sorted(SUPPORTED_FORMATS)})"
                )
            probe.load()
            planes, names, depth = _pil_mode_planes(probe)
            width, height = probe.size
            mode = probe.mode
    except UnidentifiedImageError as exc:
        raise UnsupportedImage(f"{path.name}: not a readable image file") from exc

    return LoadedImage(
        path=path,
        fmt=fmt,
        mode=mode,
        planes=planes,
        plane_names=names,
        width=width,
        height=height,
        bit_depth=depth,
        file_size=path.stat().st_size,
    )


def iter_image_files(root: str | Path) -> list[Path]:
    """Recursively list supported image files under *root* (sorted, stable)."""
    root = Path(root)
    exts = {".png", ".bmp", ".jpg", ".jpeg", ".jpe", ".jfif"}
    files = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    ]
    return files
