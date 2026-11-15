"""卡方分析（Westfeld & Pfitzmann, 1999）。

原理
----
LSB 替换只会把值在值对 (2k, 2k+1) 内部互相转换。完全嵌入后，每对的两个
桶频数趋于相等。对观测频数与“配对均分”的期望频数做卡方拟合优度检验：

- p 值接近 1  -> 值对几乎完全均分 -> 高度疑似 LSB 嵌入
- p 值接近 0  -> 自然图像的正常波动

为识别“顺序嵌入”（消息只占图像前一段），再做累积滑窗：沿样本扫描方向
不断扩大前缀，记录每个前缀的 p 值曲线，首次越过 0.95 的位置即估计的
顺序嵌入长度比例。

另外给出一个不需要干净参考图的配对不平衡启发式估计（对全域均匀嵌入也有效）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

# 累积滑窗数量与判定阈值（经典 POView 参数）
N_WINDOWS = 20
P_TRIGGER = 0.95
MIN_EXPECTED = 5.0  # 卡方检验每桶期望样本下限，低于此则合并尾部桶


@dataclass
class ChiSquareResult:
    global_p: float          # 全图卡方 p 值，[0,1]，越大越可疑
    window_p: list[float]    # 各累积前缀的 p 值
    sequential_fraction: float  # 滑窗估计的顺序嵌入比例（检测不到为 0）
    pair_estimate: float     # 配对不平衡启发式嵌入率估计
    n_samples: int
    dof: int


def _hist_pvalue(hist: np.ndarray) -> tuple[float, int]:
    """对 256 级直方图做 POV 卡方检验，返回 (p 值, 自由度)。

    每个值对 (2k,2k+1) 贡献一个自由度。期望频数 <5 的低值对先逐条剔除，
    若剔除后把它们的残差合并为一个额外桶，则该桶再贡献 1 个自由度。
    """
    even = hist[0::2].astype(np.float64)
    odd = hist[1::2].astype(np.float64)
    expected = (even + odd) * 0.5
    obs_diff2 = (even - odd) ** 2

    keep = expected >= MIN_EXPECTED
    statistic = float(np.sum(
        np.where(keep, obs_diff2 / np.maximum(expected, 1e-12), 0.0)))
    dof = int(keep.sum())  # 每个保留值对 1 个自由度

    # 被剔除的小桶合并为一个桶（总期望>0 时），额外贡献 1 个自由度
    pooled_exp = float(expected[~keep].sum())
    if pooled_exp > 0:
        statistic += float(obs_diff2[~keep].sum()) / pooled_exp
        dof += 1
    # 卡方拟合优度自由度 = 桶数 - 1；值对场景下保留桶本身已各占 1 dof，
    # 合并桶与保留桶共同受总样本约束，最终 dof 取 max(桶数-1, 1)
    dof = max(dof - 1, 1)
    if statistic <= 0:
        return 0.0, dof
    return float(chi2.sf(statistic, dof)), dof


def _pair_imbalance_estimate(hist: np.ndarray) -> float:
    """配对不平衡启发式估计。

    嵌入使值对内部差异随 (1-p) 衰减，而相邻值对之间的跨对差异受影响较小，
    用两者能量比反推嵌入率（纯启发式，作为 RS/SPA 定量结果的旁证）。
    """
    even = hist[0::2].astype(np.float64)
    odd = hist[1::2].astype(np.float64)
    within = np.sum((even - odd) ** 2)
    cross = np.sum((odd[:-1] - even[1:]) ** 2)
    if cross <= 1e-9:
        return 0.0
    ratio = within / cross
    est = 1.0 - np.sqrt(max(ratio, 0.0))
    return float(np.clip(est, 0.0, 1.0))


def chi_square_analysis(arr: np.ndarray, n_windows: int = N_WINDOWS) -> ChiSquareResult:
    """对 HxWxC uint8 图像执行卡方分析。"""
    channels = [arr[:, :, c].reshape(-1) for c in range(arr.shape[2])]
    total = sum(ch.size for ch in channels)

    # 全图直方图（各通道相加）
    full_hist = np.zeros(256, dtype=np.int64)
    for ch in channels:
        full_hist += np.bincount(ch, minlength=256)
    global_p, dof = _hist_pvalue(full_hist)
    pair_est = _pair_imbalance_estimate(full_hist)

    # 累积滑窗：逐通道按扫描方向取前缀算 p 值，取各通道最大值。
    # 必须逐通道：顺序嵌入写满一个通道才进入下一个，跨通道拼合直方图或对
    # p 值取平均，都会让未嵌入通道（p≈0）把消息所在通道的高 p 稀释掉。
    window_p: list[float] = []
    trigger_fraction = 0.0
    for i in range(1, n_windows + 1):
        frac = i / n_windows
        p_values = []
        for ch in channels:
            cut = max(int(round(ch.size * frac)), 1)
            hist = np.bincount(ch[:cut], minlength=256)
            p, _ = _hist_pvalue(hist)
            p_values.append(p)
        p = float(np.max(p_values))
        window_p.append(p)
        if trigger_fraction == 0.0 and p >= P_TRIGGER:
            trigger_fraction = frac

    return ChiSquareResult(
        global_p=global_p,
        window_p=window_p,
        sequential_fraction=float(trigger_fraction),
        pair_estimate=pair_est,
        n_samples=int(total),
        dof=dof,
    )
