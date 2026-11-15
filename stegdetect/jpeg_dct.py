"""基于 DCT 系数直方图的 JPEG 隐写检测（针对 JSteg 类 LSB 替换）。

三组互补指标
------------
1. 系数量级卡方：JSteg 把值在值对 (2k,2k+1)、(-2k-1,-2k-2) 内部互换，
   嵌入后每对桶频数趋于相等。对 AC 直方图做 POV 卡方检验（0/1 对不参与，
   因为 JSteg 跳过 0 和 1）。p 值越接近 1 越可疑。
2. h(1) 损耗（Westfeld 一阶统计）：嵌入把部分 ±2 拉低成 ±1，h(1)/h(2)
   比值随嵌入率升高。与裁剪重编码的“校准图”比较，可消除图像内容依赖。
3. 校准：对解码像素裁掉 4 行 4 列后用相同量化表重编码（Fridrich 校准思想），
   校准图近似“无嵌入基准”，比较两者直方图 L1 距离与 h(1) 漂移。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .jpeg_codec import JpegImage, decode_to_pixels, encode_jpeg, read_jpeg

MIN_EXPECTED = 5.0
CALIB_CROP = 4


@dataclass
class JpegDCTResult:
    chi_p: float              # 系数直方图卡方 p 值
    chi_rate: float           # 卡方配对估计的嵌入率（低嵌入率不敏感，仅旁证）
    h1_ratio: float           # 本图 h(1)/h(2)
    calib_h1_ratio: float     # 校准图 h(1)/h(2)
    h1_drift: float           # (本图 - 校准)/校准，JSteg 下为负且随嵌入率加深
    hist_distance: float      # 与校准图直方图归一化 L1 距离
    calib_rate: float         # 基于校准差分回归的嵌入率估计
    n_ac: int
    n_nonzero_ac: int         # JSteg 容量（非 0 的 AC）
    suspicion: float          # 融合可疑度 [0,1]


def _ac_histogram(ac: np.ndarray, lo: int = -16, hi: int = 16) -> np.ndarray:
    bins = np.arange(lo, hi + 2)
    hist, _ = np.histogram(ac, bins=bins)
    return hist.astype(np.float64), lo


def _pair_chi(ac: np.ndarray) -> tuple[float, float, float]:
    """对 JSteg 值对做卡方，返回 (p, 配对嵌入率估计, 自由度)。

    只取低幅值桶 |值| <= 5：高量化质量下高频系数会集中在少数几个值上
    （如 6/7 桶有数万样本），它们的基数会淹没低幅值桶的均衡化信号，
    而 JSteg 实际改动最密集的正是 ±2..±5 这些非零小系数。
    0/1 对跳过（经典 JSteg 不动 0；1 与 0 同对，统计上不具区分度）。
    """
    lo, hi = -5, 5
    hist, base = _ac_histogram(ac, lo, hi)  # bins [-5,-4,...,5,6)

    def cnt(v: int) -> float:
        return float(hist[v - base])

    # LSB 值对：(-5,-4),(-3,-2),(-1,0),(1,2),(3,4)；跳过含 0 的 (-1,0)
    pairs = [(-5, -4), (-3, -2), (1, 2), (3, 4)]
    statistic = 0.0
    dof = 0
    diff_energy = 0.0
    total_energy = 0.0
    for va, vb in pairs:
        na, nb = cnt(va), cnt(vb)
        expected = (na + nb) * 0.5
        if expected < MIN_EXPECTED:
            continue
        statistic += (na - expected) ** 2 / expected + (nb - expected) ** 2 / expected
        dof += 1
        diff_energy += (na - nb) ** 2
        total_energy += (na + nb) ** 2
    if dof < 2:
        return 0.0, 0.0, float(dof)
    p = float(chi2.sf(statistic, dof))
    # 配对不平衡随 (1-p_embed)^2 衰减
    ratio = diff_energy / max(total_energy, 1e-12)
    rate = float(np.clip(1.0 - np.sqrt(max(ratio, 0.0)), 0.0, 1.0))
    return p, rate, float(dof)


def _h1_ratio(ac: np.ndarray) -> float:
    h1 = int(np.count_nonzero(np.abs(ac) == 1))
    h2 = int(np.count_nonzero(np.abs(ac) == 2))
    return h1 / max(h2, 1)


def _calibrate(jpg: JpegImage, quality_estimate: int) -> tuple[np.ndarray, float]:
    """裁剪 4 行 4 列后重编码，返回校准图 AC 系数与 h1 比。"""
    rgb = decode_to_pixels(jpg)
    cropped = rgb[CALIB_CROP:, CALIB_CROP:, :]
    # 沿用原图量化表反推一个等效 quality（直接用默认 85 会换表，这里查表最近匹配）
    q = _nearest_quality(jpg.quant_tables[0])
    stream = encode_jpeg(cropped, quality=q, subsampling=jpg.subsampling)
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(stream)
        calib = read_jpeg(path)
    finally:
        os.unlink(path)
    from .jpeg_codec import all_ac_coeffs
    ac = all_ac_coeffs(calib)
    return ac, _h1_ratio(ac)


def _nearest_quality(lum_qt: np.ndarray) -> int:
    """找与当前亮度量化表最接近的标准 quality（用于校准重编码）。"""
    target = lum_qt.astype(np.float64)
    best_q, best_err = 85, np.inf
    from .jpeg_codec import scaled_quant_tables
    for q in range(50, 100):
        lum, _ = scaled_quant_tables(q)
        err = float(np.mean(np.abs(lum - target)))
        if err < best_err:
            best_err, best_q = err, q
    return best_q


def jpeg_dct_analysis(jpg: JpegImage) -> JpegDCTResult:
    from stegdetect.jpeg_codec import all_ac_coeffs, jsteg_capacity
    ac = all_ac_coeffs(jpg)
    chi_p, chi_rate, _dof = _pair_chi(ac)
    h1 = _h1_ratio(ac)
    calib_ac, calib_h1 = _calibrate(jpg, 85)
    h1_drift = (h1 - calib_h1) / max(calib_h1, 1e-9)

    # 归一化直方图 L1 距离（-12..12 主区间）
    lo, hi = -12, 12
    h_a, _ = _ac_histogram(ac, lo, hi)
    h_b, _ = _ac_histogram(calib_ac, lo, hi)
    hist_distance = float(np.sum(np.abs(h_a - h_b)) / max(np.sum(h_a), 1))

    # ---- 基于校准差分的嵌入率回归 ----
    # 经验标定（在 256px 多基底测试集上实测，见 tools/calibrate.py / README）：
    #   干净图 h1 漂移集中在 -0.06~+0.11（基线取 0.0）；
    #   完全嵌入时漂移约 -0.60。线性映射到 [0,1]，负向越深嵌入率越高。
    DRIFT_CLEAN = 0.0
    DRIFT_FULL = -0.60
    calib_rate = float(np.clip(
        (h1_drift - DRIFT_CLEAN) / (DRIFT_FULL - DRIFT_CLEAN), 0.0, 1.0))
    # 直方图距离：干净约 0.005~0.011，完全嵌入约 0.05~0.06
    DIST_CLEAN, DIST_FULL = 0.008, 0.055
    dist_rate = float(np.clip(
        (hist_distance - DIST_CLEAN) / (DIST_FULL - DIST_CLEAN), 0.0, 1.0))
    rate_fused = 0.55 * calib_rate + 0.45 * dist_rate

    # ---- 分量打分 ----
    # 校准信号为主；卡方 p 仅在高嵌入率（值对充分均衡）时才有响应，作确认
    s_drift = float(np.clip(calib_rate, 0, 1))
    s_dist = float(np.clip(dist_rate, 0, 1))
    s_chi = float(np.clip((chi_p - 0.5) / 0.5, 0, 1))  # p>=0.95 才接近满分
    suspicion = float(np.clip(
        0.5 * s_drift + 0.35 * s_dist + 0.15 * s_chi, 0, 1))

    # 小图保护：AC 系数样本量不足时校准统计涨落大，按样本量折减可疑度。
    # 标定数据集单图约 9 万以上 AC；低于 2 万时线性折减到 0.5 倍置信。
    n_ac = int(ac.size)
    if n_ac < 20000:
        confidence = float(np.clip(0.5 + 0.5 * n_ac / 20000, 0.5, 1.0))
        suspicion *= confidence
        rate_fused *= confidence

    return JpegDCTResult(
        chi_p=chi_p,
        chi_rate=chi_rate,
        h1_ratio=h1,
        calib_h1_ratio=calib_h1,
        h1_drift=float(h1_drift),
        hist_distance=hist_distance,
        calib_rate=rate_fused,
        n_ac=int(ac.size),
        n_nonzero_ac=jsteg_capacity(jpg),
        suspicion=suspicion,
    )
