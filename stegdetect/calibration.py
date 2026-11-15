"""经验校准加载与应用。

各空间域方法（SPA/RS/卡方配对估计）的原始输出在不同图像统计下存在系统偏差，
这里加载 tools/calibrate.py 在带真值数据集上拟合的保序回归分段线性映射，
把原始估计校正到嵌入率量纲。校准文件缺失时退化为恒等映射。
"""
from __future__ import annotations

import json
import os

import numpy as np

_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "calibration.json")
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(_CALIB_PATH):
        with open(_CALIB_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
    else:
        _cache = {}
    return _cache


def calibrate(method: str, raw: float) -> float:
    """把某方法的原始嵌入率估计映射为校准后的估计。"""
    spec = _load().get("spatial", {}).get(method)
    if spec is None:
        return float(np.clip(raw, 0.0, 1.0))
    return float(np.clip(np.interp(raw, spec["xp"], spec["yp"]), 0.0, 1.0))
