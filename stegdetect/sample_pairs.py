"""样本对分析（Sample Pair Analysis, Dumitrescu, Wu, Wang, 2003）。

原理
----
取水平（并拼合垂直）相邻像素对 (u,v)，按差值与奇偶性分类：

- D  : 差值为奇数（两个像素 LSB 不同），再细分
- X_n: v-u=2n 且 u 为奇数；Y_n: v-u=2n 且 u 为偶数（差值为偶数）
- Z_n: 差 2n+1 且 u 奇；W_n: 差 2n+1 且 u 偶

对 LSB 替换建立多集合（multiset）的迹条件（trace condition），导出关于嵌入率
p 的二次方程 a p² + b p + c = 0。实现采用论文给出的聚合形式：

    a = 2 Σ W_n
    b = 2(Σ Z_n − Σ W_n) − D
    c = −Σ Z_n
其中 D = Σ(D_n + D_{-n}) 覆盖所有奇数差像素对。取 [0,1] 内最接近 0 的根。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SPAResult:
    rate: float          # 估计嵌入率 p，[0,1]
    n_pairs: int
    a: float
    b: float
    c: float
    discriminant: float
    clipped: bool        # 根被裁剪到 [0,1]（估计不可靠的信号）


def _collect(first: np.ndarray, second: np.ndarray):
    """从一组有序像素对 (first, second) 计算 D、ΣZ、ΣW、ΣX、ΣY。"""
    u = first.astype(np.int16)
    v = second.astype(np.int16)
    diff = v - u
    odd_mask = (diff & 1) != 0
    u_odd = (u & 1) != 0

    # D：所有奇数差对
    D = int(np.count_nonzero(odd_mask))
    # Z_n：差 2n+1 且 u 奇；W_n：差 2n+1 且 u 偶
    # 聚合量 Σ n·|Z_n| / Σ n·|W_n|。注意负差时 n 的符号：
    #   diff=2n+1 -> n=(diff-1)/2；diff=2n-1（即 -|2n-1|）-> n=(diff+1)/2
    odd_diffs = diff[odd_mask]
    odd_u_odd = u_odd[odd_mask]
    n_idx = np.where(odd_diffs >= 0,
                     (odd_diffs - 1) // 2,
                     (odd_diffs + 1) // 2)
    sum_Z = int(np.sum(np.where(odd_u_odd, n_idx, 0)))
    sum_W = int(np.sum(np.where(~odd_u_odd, n_idx, 0)))

    # 偶数差对 X_n / Y_n，n=diff/2
    even_mask = ~odd_mask
    even_diffs = diff[even_mask]
    even_u_odd = u_odd[even_mask]
    n_even = even_diffs // 2
    sum_X = int(np.sum(np.where(even_u_odd, n_even, 0)))
    sum_Y = int(np.sum(np.where(~even_u_odd, n_even, 0)))
    return D, sum_Z, sum_W, sum_X, sum_Y


def sample_pairs_analysis(arr: np.ndarray) -> SPAResult:
    """对 HxWxC uint8 图像执行 SPA，水平/垂直像素对合并统计。"""
    D_tot = sZ = sW = sX = sY = 0
    n_pairs = 0
    for c in range(arr.shape[2]):
        plane = arr[:, :, c]
        # 水平对：每行相邻列
        D, z, w, x, y = _collect(plane[:, :-1].reshape(-1), plane[:, 1:].reshape(-1))
        D_tot += D; sZ += z; sW += w; sX += x; sY += y
        n_pairs += plane[:, :-1].size
        # 垂直对：每列相邻行
        D, z, w, x, y = _collect(plane[:-1, :].reshape(-1), plane[1:, :].reshape(-1))
        D_tot += D; sZ += z; sW += w; sX += x; sY += y
        n_pairs += plane[:-1, :].size

    a = 2.0 * sW
    b = 2.0 * (sZ - sW) - D_tot
    c = -float(sZ)

    if abs(a) < 1e-12:
        # 退化为线性
        p = (-c / b) if abs(b) > 1e-12 else 0.0
        disc = b * b - 4 * a * c
        clipped = not (0.0 <= p <= 1.0)
        return SPAResult(float(np.clip(p, 0, 1)), n_pairs, a, b, c, disc, clipped)

    disc = b * b - 4.0 * a * c
    if disc < 0:
        # 数值上非正定，视为无嵌入
        return SPAResult(0.0, n_pairs, a, b, c, disc, True)
    sqrt_disc = np.sqrt(disc)
    r1 = (-b + sqrt_disc) / (2.0 * a)
    r2 = (-b - sqrt_disc) / (2.0 * a)

    in_range = [r for r in (r1, r2) if -1e-9 <= r <= 1.0 + 1e-9]
    if in_range:
        p = min(in_range, key=abs)
        clipped = not (0.0 <= p <= 1.0)
        p = float(np.clip(p, 0.0, 1.0))
    else:
        # 两个根都在区间外：取离 [0,1] 最近的并裁剪，标记不可靠
        p = float(np.clip(r1 if abs(r1) < abs(r2) else r2, 0.0, 1.0))
        clipped = True

    return SPAResult(rate=p, n_pairs=n_pairs, a=float(a), b=float(b), c=float(c),
                     discriminant=float(disc), clipped=clipped)
