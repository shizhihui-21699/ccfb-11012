"""图像读取与格式识别。

PNG / BMP 走 Pillow 读成 numpy 数组；JPEG 交给 :mod:`stegdetect.jpeg_codec`
以拿到量化 DCT 系数。所有空间域分析都基于 ``H x W x C`` 的 uint8 数组。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

SPATIAL_EXTS = {".png", ".bmp"}
JPEG_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif"}
SUPPORTED_EXTS = SPATIAL_EXTS | JPEG_EXTS


@dataclass
class SpatialImage:
    """空间域图像。"""

    path: str
    array: np.ndarray  # H x W x C, uint8
    kind: str          # "png" | "bmp"

    @property
    def channels(self) -> int:
        return self.array.shape[2]

    @property
    def capacity_bits(self) -> int:
        """LSB 替换容量：每个通道样本 1 bit。"""
        return int(self.array.size)


def kind_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in SPATIAL_EXTS:
        return "spatial"
    if ext in JPEG_EXTS:
        return "jpeg"
    return "unsupported"


def load_spatial(path: str) -> SpatialImage:
    """读取 PNG/BMP 为 HxWxC uint8（灰度保留单通道，其余转 RGB）。"""
    with Image.open(path) as im:
        if im.mode in ("L", "I;16"):
            arr = np.asarray(im.convert("L"), dtype=np.uint8)[:, :, np.newaxis]
        elif im.mode == "RGBA":
            # alpha 通道不参与常规 LSB 分析，只取 RGB
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    kind = os.path.splitext(path)[1].lower().lstrip(".")
    return SpatialImage(path=path, array=arr, kind=kind)


def iter_image_files(root: str):
    """递归遍历目录，按扩展名 yield 受支持的图片路径。"""
    if os.path.isfile(root):
        if kind_of(root) != "unsupported":
            yield root
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            if kind_of(path) != "unsupported":
                yield path
