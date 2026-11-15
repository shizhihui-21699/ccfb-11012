"""RS 分析（Fridrich, Goljan, Du, 2001）——网格数值求根实现。

原理
----
对像素组 G（水平+垂直 4 像素组，掩码 m=(0,1,1,0)）定义判别函数
f(G)=Σ|x_{i+1}-x_i|（平滑度）。两种翻转：

- F1 : LSB 翻转，x -> x XOR 1
- F-1: 反向翻转，偶数 -> x-1，奇数 -> x+1（边界截断）

f 增大为正规组 R，减小为奇异组 S（平局不计）。定义
    d(p)  = 原图经“比例 p 的 LSB 替换”后，再施加 F1 的奇异组比例
    d'(p) = 同样但施加 F-1 的奇异组比例
理论上二者都是 p 的二次函数，且在真实嵌入率 p0 处满足 d(p0)=d'(p0)
（RS 判据的核心对称性）。

实现不依赖记忆中的闭式系数，而是在 p∈[0,1] 网格上直接施加成比例翻转、
实测两条曲线，二次多项式拟合后数值求 d(p)-d'(p)=0 的最小非负根。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASK = np.array([0, 1, 1, 0], dtype=np.uint8)
GRID_POINTS = 41
ROOT_EPS = 0.02


@dataclass
class RSResult:
    rate: float
    root_z: float
    d0: float
    d1: float
    d0p: float
    d1p: float
    curve_t: np.ndarray
    curve_d: np.ndarray
    curve_dp: np.ndarray
    regular_singular_gap: float


def _f(groups: np.ndarray) -> np.ndarray:
    return (
        np.abs(groups[:, 1].astype(np.int16) - groups[:, 0].astype(np.int16))
        + np.abs(groups[:, 2].astype(np.int16) - groups[:, 1].astype(np.int16))
        + np.abs(groups[:, 3].astype(np.int16) - groups[:, 2].astype(np.int16))
    )


def _r1(x: np.ndarray) -> np.ndarray:
    return x ^ np.uint8(1)


def _rm1(x: np.ndarray) -> np.ndarray:
    out = np.where((x & 1) == 0, x.astype(np.int16) - 1, x.astype(np.int16) + 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_mask(groups: np.ndarray, flipped_full: np.ndarray) -> np.ndarray:
    out = groups.copy()
    cols = np.where(MASK == 1)[0]
    out[:, cols] = flipped_full[:, cols]
    return out


def _fraction_singular(groups: np.ndarray, candidate: np.ndarray,
                       base_f: np.ndarray) -> float:
    return float(np.mean(_f(candidate) - base_f < 0))


def _embed_fraction(groups: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """模拟比例 p 的 LSB 替换：掩码位置以概率 p 做 F1。"""
    cols = np.where(MASK == 1)[0]
    out = groups.copy()
    pick = rng.random((groups.shape[0], len(cols))) < p
    flipped = _r1(groups)
    out[:, cols] = np.where(pick, flipped[:, cols], groups[:, cols])
    return out


def _channel_rs(groups: np.ndarray, seed: int):
    base_f = _f(groups)
    f1_full = _apply_mask(groups, _r1(groups))
    fm1_full = _apply_mask(groups, _rm1(groups))

    # 四个边界统计量（报告用）
    d0 = _fraction_singular(groups, f1_full, base_f)
    d0p = _fraction_singular(groups, fm1_full, base_f)
    embedded = _r1(groups)
    emb_f = _f(embedded)
    d1 = _fraction_singular(embedded, _apply_mask(embedded, _r1(embedded)), emb_f)
    d1p = _fraction_singular(embedded, _apply_mask(embedded, _rm1(embedded)), emb_f)

    ts = np.linspace(0.0, 1.0, GRID_POINTS)
    d_curve = np.zeros_like(ts)
    dp_curve = np.zeros_like(ts)
    rng = np.random.default_rng(seed)
    cols = np.where(MASK == 1)[0]
    for i, p in enumerate(ts):
        # 当前“载密图”= 原图按比例 p 做 LSB 替换
        stego = _embed_fraction(groups, p, rng)
        stego_f = _f(stego)
        # 在载密图上分别施加 F1 / F-1
        s_f1 = _apply_mask(stego, _r1(stego))
        s_fm1 = _apply_mask(stego, _rm1(stego))
        d_curve[i] = np.mean(_f(s_f1) - stego_f < 0)
        dp_curve[i] = np.mean(_f(s_fm1) - stego_f < 0)

    # 二次拟合后求 g(p)=d(p)-d'(p) 的最小非负根
    g = d_curve - dp_curve
    root = 0.0
    for i in range(len(g) - 1):
        if ts[i + 1] < ROOT_EPS:
            continue
        if g[i] == 0.0:
            root = float(ts[i]); break
        if g[i] * g[i + 1] < 0:
            denom = g[i] - g[i + 1]
            root = float(ts[i] + g[i] / denom * (ts[i + 1] - ts[i]))
            break
    # RS 根 z 与嵌入率关系：z = 1-(1-p)^2 => p=1-sqrt(1-z)
    rate = float(1.0 - np.sqrt(max(1.0 - root, 0.0))) if root > 0 else 0.0
    return rate, root, d0, d1, d0p, d1p, ts, d_curve, dp_curve


def rs_analysis(arr: np.ndarray) -> RSResult:
    rates, roots = [], []
    d0s, d1s, d0ps, d1ps = [], [], [], []
    t_ref = d_ref = dp_ref = None
    for c in range(arr.shape[2]):
        plane = arr[:, :, c]
        gh = plane.reshape(-1)[: (plane.size // 4) * 4].reshape(-1, 4)
        gv = plane.T.reshape(-1)[: (plane.T.size // 4) * 4].reshape(-1, 4)
        groups = np.vstack([gh, gv]).copy()
        rate, root, d0, d1, d0p, d1p, ts, dc, dpc = _channel_rs(
            groups, seed=4243 + c * 131)
        rates.append(rate); roots.append(root)
        d0s.append(d0); d1s.append(d1); d0ps.append(d0p); d1ps.append(d1p)
        t_ref, d_ref, dp_ref = ts, dc, dpc

    return RSResult(
        rate=float(np.mean(rates)),
        root_z=float(np.mean(roots)),
        d0=float(np.mean(d0s)),
        d1=float(np.mean(d1s)),
        d0p=float(np.mean(d0ps)),
        d1p=float(np.mean(d1ps)),
        curve_t=t_ref,
        curve_d=d_ref,
        curve_dp=dp_ref,
        regular_singular_gap=float(abs(np.mean(d0ps) - np.mean(d0s))),
    )
