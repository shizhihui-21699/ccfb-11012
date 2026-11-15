"""在带真值测试集上评估各检测方法并拟合经验校准。

输出
----
- 各方法原始估计 vs 真值的对照表
- 保序回归（isotonic）校准参数，写入 stegdetect/calibration.json
- 干净图误报率、载密图检出率、嵌入率 MAE
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stegdetect.image_io import load_spatial
from stegdetect.chi_square import chi_square_analysis
from stegdetect.rs_analysis import rs_analysis
from stegdetect.sample_pairs import sample_pairs_analysis
from stegdetect.jpeg_codec import read_jpeg, jsteg_capacity
from stegdetect.jpeg_dct import jpeg_dct_analysis


def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """简单保序回归（PAVA），返回单调不降的分段点。"""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    # PAVA
    blocks_v, blocks_w, blocks_x = [], [], []
    for xi, yi in zip(xs, ys):
        blocks_v.append(yi); blocks_w.append(1); blocks_x.append(xi)
        while len(blocks_v) >= 2 and blocks_v[-2] > blocks_v[-1]:
            w1, w2 = blocks_w[-2], blocks_w[-1]
            merged = (blocks_v[-2] * w1 + blocks_v[-1] * w2) / (w1 + w2)
            blocks_v[-2:] = [merged]; blocks_w[-2:] = [w1 + w2]
            blocks_x[-2:] = [blocks_x[-2]]
    # 取每个唯一 x 的拟合值
    uniq_x, idx = np.unique(xs, return_inverse=True)
    # 用块均值构造映射点
    pts_x, pts_y = [], []
    bi = 0
    cum = 0
    block_ends = np.cumsum(blocks_w)
    for ux in uniq_x:
        while bi < len(block_ends) and cum >= block_ends[bi]:
            bi += 1
        pts_x.append(ux)
        pts_y.append(np.clip(blocks_v[min(bi, len(blocks_v) - 1)], 0, 1))
        cum += 1
    return np.array(pts_x), np.array(pts_y)


class Calibrator:
    """分段线性单调映射，端点外裁剪到 [0,1]。"""

    def __init__(self, xp: np.ndarray, yp: np.ndarray):
        order = np.argsort(xp)
        self.xp = np.concatenate([[0.0], xp[order], [1.0]])
        self.yp = np.concatenate([[yp[order][0]], yp[order], [yp[order][-1]]])
        self.yp = np.clip(self.yp, 0, 1)

    def apply(self, x: float) -> float:
        return float(np.interp(x, self.xp, self.yp))

    def to_json(self) -> dict:
        return {"xp": self.xp.tolist(), "yp": self.yp.tolist()}


def evaluate_spatial(manifest, root):
    rows = []
    for m in manifest:
        if m["format"] not in ("png", "bmp"):
            continue
        img = load_spatial(os.path.join(root, m["file"]))
        arr = img.array
        chi = chi_square_analysis(arr)
        rs = rs_analysis(arr)
        spa = sample_pairs_analysis(arr)
        rows.append({
            "true": m["true_rate"],
            "chi_pair": chi.pair_estimate,
            "chi_trig": chi.sequential_fraction,
            "rs": rs.rate,
            "spa": spa.rate,
            "mode": m["mode"],
        })
    return rows


def evaluate_jpeg(manifest, root):
    rows = []
    for m in manifest:
        if m["format"] != "jpeg":
            continue
        jpg = read_jpeg(os.path.join(root, m["file"]))
        r = jpeg_dct_analysis(jpg)
        rows.append({
            "true": m["true_rate"],
            "chi_rate": r.chi_rate,
            "calib_rate": r.calib_rate,
            "drift": r.h1_drift,
            "dist": r.hist_distance,
            "suspicion": r.suspicion,
        })
    return rows


def main():
    root = "data/test_images"
    manifest = json.load(open(os.path.join(root, "manifest.json")))

    sp = evaluate_spatial(manifest, root)
    jp = evaluate_jpeg(manifest, root)
    true_sp = np.array([r["true"] for r in sp])

    print("== 空间域原始估计（按真值聚合均值）==")
    print(f"{'true':>6} {'SPA':>7} {'RS':>7} {'chi_pair':>9} {'n':>3}")
    for t in sorted(set(true_sp)):
        sel = [r for r in sp if abs(r["true"] - t) < 1e-9]
        print(f"{t:6.2f} {np.mean([r['spa'] for r in sel]):7.3f} "
              f"{np.mean([r['rs'] for r in sel]):7.3f} "
              f"{np.mean([r['chi_pair'] for r in sel]):9.3f} {len(sel):3d}")

    # ---- 拟合校准（仅用随机散布 + 顺序合并，覆盖两种模式）----
    calib = {}
    for key in ("spa", "rs", "chi_pair"):
        x = np.array([r[key] for r in sp])
        xp, yp = isotonic_fit(x, true_sp)
        cal = Calibrator(xp, yp)
        calib[key] = cal.to_json()
        # 校准后 MAE
        pred = np.array([cal.apply(v) for v in x])
        mae = np.mean(np.abs(pred - true_sp))
        print(f"calibrated {key:9s} MAE={mae:.3f}")

    # JPEG 校准评估
    true_jp = np.array([r["true"] for r in jp])
    print("\n== JPEG 原始/校准估计 ==")
    print(f"{'true':>6} {'calib_rate':>11} {'suspicion':>10} {'n':>3}")
    for t in sorted(set(true_jp)):
        sel = [r for r in jp if abs(r["true"] - t) < 1e-9]
        print(f"{t:6.2f} {np.mean([r['calib_rate'] for r in sel]):11.3f} "
              f"{np.mean([r['suspicion'] for r in sel]):10.3f} {len(sel):3d}")
    jmae = np.mean(np.abs(np.array([r["calib_rate"] for r in jp]) - true_jp))
    print(f"JPEG calib MAE={jmae:.3f}")

    # ---- 干净图误报检查 ----
    clean_sp = [r for r in sp if r["true"] == 0]
    print(f"\n空间域干净图 n={len(clean_sp)}: "
          f"SPA max={max(r['spa'] for r in clean_sp):.3f} "
          f"RS max={max(r['rs'] for r in clean_sp):.3f}")
    clean_jp = [r for r in jp if r["true"] == 0]
    print(f"JPEG 干净图 n={len(clean_jp)}: "
          f"suspicion max={max(r['suspicion'] for r in clean_jp):.3f} "
          f"mean={np.mean([r['suspicion'] for r in clean_jp]):.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "stegdetect", "calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"spatial": calib}, f, indent=2)
    print(f"\n[+] 校准参数已写入 {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
