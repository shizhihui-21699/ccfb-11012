"""单图/批量检测流水线与多方法评分融合。"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .image_io import load_spatial, kind_of, SPATIAL_EXTS, JPEG_EXTS
from .chi_square import chi_square_analysis
from .rs_analysis import rs_analysis
from .sample_pairs import sample_pairs_analysis
from .jpeg_codec import read_jpeg, JpegError
from .jpeg_dct import jpeg_dct_analysis
from .calibration import calibrate

# 判定阈值（在带真值测试集上标定，报告中披露）。
# 分数体系下干净图可疑度普遍 <0.10，中高嵌入率载密图显著更高；
# 0.30 起判 suspicious，0.55 起判 likely_stego，可在零误报前提下检出
# 嵌入率 >=0.25 的样本。<=0.10 的低嵌入率与自然统计涨落重叠，不强行报警。
THRESHOLD_SUSPICIOUS = 0.30
THRESHOLD_HIGH = 0.55


@dataclass
class DetectionResult:
    file: str
    format: str               # png/bmp/jpeg
    width: int
    height: int
    channels: int
    capacity_bits: int
    # 各方法原始输出
    chi_global_p: Optional[float]
    chi_sequential_fraction: Optional[float]
    chi_pair_estimate: Optional[float]
    rs_rate: Optional[float]
    spa_rate: Optional[float]
    spa_clipped: Optional[bool]
    jpeg_chi_p: Optional[float]
    jpeg_chi_rate: Optional[float]
    jpeg_h1_drift: Optional[float]
    jpeg_hist_distance: Optional[float]
    # 融合结果
    suspicion: float
    estimated_rate: float
    estimated_payload_bits: int
    verdict: str              # clean / suspicious / likely_stego
    error: Optional[str] = None


def _verdict(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "likely_stego"
    if score >= THRESHOLD_SUSPICIOUS:
        return "suspicious"
    return "clean"


def _detect_spatial(path: str) -> DetectionResult:
    img = load_spatial(path)
    arr = img.array
    chi = chi_square_analysis(arr)
    rs = rs_analysis(arr)
    spa = sample_pairs_analysis(arr)

    # 经验校准：把各方法原始估计校正到嵌入率量纲
    spa_cal = calibrate("spa", spa.rate)
    chi_cal = calibrate("chi_pair", chi.pair_estimate)
    rs_cal = calibrate("rs", rs.rate)

    # 融合权重：SPA 与卡方为主力（在测试集上单调、干净误报低）；
    # RS 在强纹理/真实照片上有效，但在平滑合成图上二次模型会退化，权重最低。
    W_SPA, W_CHI, W_RS = 0.45, 0.40, 0.15
    # 顺序嵌入时滑窗检出是强信号，额外抬升可疑度
    seq_signal = chi.sequential_fraction
    estimated_rate = float(np.clip(
        W_SPA * spa_cal + W_CHI * chi_cal + W_RS * rs_cal, 0, 1))
    if seq_signal > 0:
        estimated_rate = float(np.clip(max(estimated_rate, seq_signal), 0, 1))

    # 可疑度：校准嵌入率是主信号（开方拉伸以提升中低嵌入率灵敏度），
    # 卡方 p 值与顺序滑窗作为存在性证据加权。
    s_chi_p = float(np.clip(chi.global_p, 0, 1))
    s_seq = float(np.clip(seq_signal, 0, 1))
    rate_component = float(np.sqrt(np.clip(estimated_rate, 0, 1)))
    suspicion = float(np.clip(
        0.70 * rate_component
        + 0.18 * s_chi_p
        + 0.12 * s_seq, 0, 1))

    payload = int(round(estimated_rate * img.capacity_bits))

    h, w, c = arr.shape
    return DetectionResult(
        file=path, format=img.kind, width=w, height=h, channels=c,
        capacity_bits=img.capacity_bits,
        chi_global_p=chi.global_p,
        chi_sequential_fraction=chi.sequential_fraction,
        chi_pair_estimate=chi.pair_estimate,
        rs_rate=rs.rate,
        spa_rate=spa.rate,
        spa_clipped=spa.clipped,
        jpeg_chi_p=None, jpeg_chi_rate=None,
        jpeg_h1_drift=None, jpeg_hist_distance=None,
        suspicion=suspicion,
        estimated_rate=estimated_rate,
        estimated_payload_bits=payload,
        verdict=_verdict(suspicion),
    )


def _detect_jpeg(path: str) -> DetectionResult:
    jpg = read_jpeg(path)
    dct = jpeg_dct_analysis(jpg)
    suspicion = dct.suspicion
    # 嵌入率以校准回归为主；卡方配对估计仅在高嵌入率可信
    rate = dct.calib_rate
    payload = int(round(rate * dct.n_nonzero_ac))
    return DetectionResult(
        file=path, format="jpeg", width=jpg.width, height=jpg.height,
        channels=3, capacity_bits=dct.n_nonzero_ac,
        chi_global_p=None, chi_sequential_fraction=None,
        chi_pair_estimate=None, rs_rate=None, spa_rate=None, spa_clipped=None,
        jpeg_chi_p=dct.chi_p,
        jpeg_chi_rate=dct.chi_rate,
        jpeg_h1_drift=dct.h1_drift,
        jpeg_hist_distance=dct.hist_distance,
        suspicion=float(suspicion),
        estimated_rate=float(np.clip(rate, 0, 1)),
        estimated_payload_bits=payload,
        verdict=_verdict(suspicion),
    )


def detect_file(path: str) -> DetectionResult:
    try:
        if kind_of(path) == "jpeg":
            return _detect_jpeg(path)
        return _detect_spatial(path)
    except JpegError as e:
        return _error_result(path, f"不支持的 JPEG: {e}")
    except Exception as e:  # 单图失败不应拖垮批量任务
        return _error_result(path, f"{type(e).__name__}: {e}")


def _error_result(path: str, msg: str) -> DetectionResult:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return DetectionResult(
        file=path, format=ext if ext else "?", width=0, height=0,
        channels=0, capacity_bits=0,
        chi_global_p=None, chi_sequential_fraction=None,
        chi_pair_estimate=None, rs_rate=None, spa_rate=None,
        spa_clipped=None, jpeg_chi_p=None, jpeg_chi_rate=None,
        jpeg_h1_drift=None, jpeg_hist_distance=None,
        suspicion=0.0, estimated_rate=0.0, estimated_payload_bits=0,
        verdict="error", error=msg,
    )


def detect_directory(root: str) -> list[DetectionResult]:
    from .image_io import iter_image_files
    results = []
    for path in iter_image_files(root):
        results.append(detect_file(path))
    return results


def result_to_dict(r: DetectionResult) -> dict:
    d = asdict(r)
    d["file"] = os.path.basename(r.file)
    return d
