"""批量生成带真值的隐写测试集。

布局（data/test_images/）
------------------------
clean/    干净 PNG/BMP/JPEG 对照图
stego_png/  不同嵌入率的 PNG 载密图（顺序 + 随机散布两种模式）
stego_bmp/  同上的 BMP 版本（每基底各取部分嵌入率，控制总量）
stego_jpg/  JSteg 载密 JPEG
manifest.json  每张图的真值（基底、格式、真实嵌入率、模式、嵌入位数、容量）

基底图为程序化合成的“类自然图像”（梯度/噪声/分形/斑块/条纹/混合），
固定随机种子保证可复现。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stegdetect.stego_embed import (  # noqa: E402
    embed_spatial_lsb, make_payload, embed_jsteg)
from stegdetect.jpeg_codec import encode_jpeg, read_jpeg  # noqa: E402

W, H = 256, 256
RATES = [0.05, 0.10, 0.25, 0.50, 0.90]
JPEG_RATES = [0.05, 0.15, 0.30, 0.60, 0.90]
JPEG_QUALITY = 85


def _smooth_field(rng: np.random.Generator, coarse: int,
                  sigma: float = 0.0) -> np.ndarray:
    """生成连续色调的平滑随机场（双三次上采样 + 可选高斯平滑）。

    用插值上采样而非块状复制，避免出现 LSB 恒定的平坦块——后者会让 RS
    判别函数大量平局、卡方直方图饱和，都不是真实照片的统计特征。
    """
    low = rng.normal(0, 1, (coarse, coarse, 3))
    zoom_factor = H / coarse
    field = zoom(low, (zoom_factor, zoom_factor, 1), order=3)
    # 裁/补到精确尺寸
    field = field[:H, :W, :]
    if field.shape[0] < H or field.shape[1] < W:
        pad = np.zeros((H, W, 3))
        pad[:field.shape[0], :field.shape[1]] = field
        field = pad
    if sigma > 0:
        for c in range(3):
            field[:, :, c] = gaussian_filter(field[:, :, c], sigma)
    return field


# ---------------------------------------------------------------------------
# 基底图合成（固定种子，连续色调、类照片统计，值分布平滑不饱和）
# ---------------------------------------------------------------------------
def _base_gradient(rng: np.random.Generator) -> np.ndarray:
    """大尺度平滑渐变 + 很弱的相关噪声。"""
    xs = np.linspace(0, 1, W)[None, :, None]
    ys = np.linspace(0, 1, H)[:, None, None]
    img = np.zeros((H, W, 3), dtype=np.float64)
    img[:, :, 0] = (xs * 180 + 35).squeeze()
    img[:, :, 1] = (ys * 160 + 45).squeeze()
    img[:, :, 2] = ((1 - xs) * 150 + (1 - ys) * 55).squeeze()
    img += 4 * _smooth_field(rng, 32, sigma=1.0)
    return np.clip(img, 8, 247).astype(np.uint8)


def _base_smooth_noise(rng: np.random.Generator) -> np.ndarray:
    """低频平滑色块场（插值上采样）+ 细颗粒相关噪声。"""
    img = 128 + 55 * _smooth_field(rng, 16, sigma=1.2)
    img += rng.normal(0, 2.5, img.shape)  # 弱细噪声，保留 LSB 结构
    return np.clip(img, 6, 250).astype(np.uint8)


def _base_fractal(rng: np.random.Generator) -> np.ndarray:
    """多尺度平滑随机场叠加（1/f 统计），模拟自然纹理。

    每个尺度都用插值上采样，保证像素平滑变化、相邻像素强相关。
    """
    img = np.zeros((H, W, 3), dtype=np.float64)
    for c in range(3):
        acc = np.zeros((H, W))
        amp, total_amp, coarse = 1.0, 0.0, 2
        while coarse <= H // 2:
            low = rng.normal(0, 1, (coarse, coarse))
            up = zoom(low, H / coarse, order=3)[:H, :W]
            acc += amp * up
            total_amp += amp
            amp *= 0.62
            coarse *= 2
        img[:, :, c] = 128 + 68 * acc / total_amp
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 6, 250).astype(np.uint8)


def _base_patches(rng: np.random.Generator) -> np.ndarray:
    """柔边斑块：随机椭圆/矩形叠加后整体高斯模糊，无硬边不饱和。"""
    img = np.zeros((H, W, 3), dtype=np.float64)
    base = 120 + 30 * _smooth_field(rng, 8, sigma=2.0)
    img += base
    layer = np.zeros((H, W, 3))
    for _ in range(25):
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        ry, rx = rng.integers(15, 60, 2)
        color = rng.integers(-50, 50, 3)
        yy, xx = np.mgrid[0:H, 0:W]
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        layer[mask] += color
    for c in range(3):
        layer[:, :, c] = gaussian_filter(layer[:, :, c], 6.0)
    img += layer + rng.normal(0, 2.5, img.shape)
    return np.clip(img, 6, 250).astype(np.uint8)


def _base_stripes(rng: np.random.Generator) -> np.ndarray:
    """平滑正弦条纹（软过渡，无硬阶跃）+ 弱噪声。"""
    f = rng.uniform(0.02, 0.06)
    phase = rng.uniform(0, 2 * np.pi, 3)
    xs = np.arange(W)
    img = np.zeros((H, W, 3))
    for c in range(3):
        img[:, :, c] = 128 + 55 * np.sin(2 * np.pi * f * xs + phase[c])[None, :]
    img += 18 * _smooth_field(rng, 24, sigma=1.0)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 6, 250).astype(np.uint8)


def _base_mixed(rng: np.random.Generator) -> np.ndarray:
    """非平稳混合：上半分形纹理 + 下半平滑渐变 + 柔边亮盘。"""
    top = _base_fractal(rng)
    grad = _base_gradient(rng)
    img = np.vstack([top[:H // 2], grad[H // 2:]]).astype(np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx, r = H // 2, W // 2, 55
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2).astype(float)
    disk = gaussian_filter(disk, 3.0)[:, :, None]
    img += 60 * disk
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 6, 250).astype(np.uint8)


BASE_BUILDERS = [
    ("gradient", _base_gradient),
    ("smooth_noise", _base_smooth_noise),
    ("fractal", _base_fractal),
    ("patches", _base_patches),
    ("stripes", _base_stripes),
    ("mixed", _base_mixed),
]


def build_bases(seed: int = 20261116) -> list[tuple[str, np.ndarray]]:
    bases = []
    for i, (name, fn) in enumerate(BASE_BUILDERS):
        rng = np.random.default_rng(seed + i * 101)
        bases.append((name, fn(rng)))
    return bases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成带真值的隐写测试集")
    ap.add_argument("--out", default="data/test_images")
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args(argv)

    root = args.out
    for sub in ("clean", "stego_png", "stego_bmp", "stego_jpg"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    bases = build_bases(args.seed)
    manifest = []
    payload_seed = args.seed + 7

    def record(**kw):
        manifest.append(kw)

    # ---------- 干净对照 ----------
    for name, arr in bases:
        png_path = os.path.join(root, "clean", f"{name}_clean.png")
        bmp_path = os.path.join(root, "clean", f"{name}_clean.bmp")
        Image.fromarray(arr).save(png_path)
        Image.fromarray(arr).save(bmp_path, format="BMP")
        record(file=os.path.relpath(png_path, root), base=name, format="png",
               true_rate=0.0, mode="none", payload_bits=0,
               capacity_bits=int(arr.size))
        record(file=os.path.relpath(bmp_path, root), base=name, format="bmp",
               true_rate=0.0, mode="none", payload_bits=0,
               capacity_bits=int(arr.size))

        jpg_stream = encode_jpeg(arr, quality=JPEG_QUALITY, subsampling="4:2:0")
        jpg_path = os.path.join(root, "clean", f"{name}_clean.jpg")
        with open(jpg_path, "wb") as f:
            f.write(jpg_stream)
        jpg = read_jpeg(jpg_path)
        from stegdetect.jpeg_codec import jsteg_capacity
        record(file=os.path.relpath(jpg_path, root), base=name, format="jpeg",
               true_rate=0.0, mode="none", payload_bits=0,
               capacity_bits=jsteg_capacity(jpg))

    # ---------- 空间域载密 PNG（全嵌入率 × 两模式） ----------
    for name, arr in bases:
        cap = arr.size
        for rate in RATES:
            n_bits = int(round(cap * rate))
            payload = make_payload(n_bits, seed=payload_seed)
            for mode in ("sequential", "random"):
                stego, written, _ = embed_spatial_lsb(
                    arr, payload, mode=mode, seed=payload_seed + 1)
                tag = "seq" if mode == "sequential" else "rnd"
                fname = f"{name}_r{int(rate * 100):02d}_{tag}.png"
                fpath = os.path.join(root, "stego_png", fname)
                Image.fromarray(stego).save(fpath)
                record(file=os.path.relpath(fpath, root), base=name, format="png",
                       true_rate=rate, mode=mode, payload_bits=written,
                       capacity_bits=cap)

    # ---------- BMP 载密（每基底取 3 个嵌入率，顺序模式，控制规模） ----------
    for name, arr in bases:
        cap = arr.size
        for rate in (0.10, 0.50, 0.90):
            n_bits = int(round(cap * rate))
            payload = make_payload(n_bits, seed=payload_seed + 2)
            stego, written, _ = embed_spatial_lsb(arr, payload, mode="sequential")
            fname = f"{name}_r{int(rate * 100):02d}_seq.bmp"
            fpath = os.path.join(root, "stego_bmp", fname)
            Image.fromarray(stego).save(fpath, format="BMP")
            record(file=os.path.relpath(fpath, root), base=name, format="bmp",
                   true_rate=rate, mode="sequential", payload_bits=written,
                   capacity_bits=cap)

    # ---------- JPEG JSteg 载密 ----------
    for name, arr in bases:
        clean_path = os.path.join(root, "clean", f"{name}_clean.jpg")
        jpg = read_jpeg(clean_path)
        cap = jsteg_capacity(jpg)
        for rate in JPEG_RATES:
            n_bits = int(round(cap * rate))
            payload = make_payload(n_bits, seed=payload_seed + 3)
            stream, written, _ = embed_jsteg(jpg, payload, mode="sequential")
            fname = f"{name}_r{int(rate * 100):02d}_seq.jpg"
            fpath = os.path.join(root, "stego_jpg", fname)
            with open(fpath, "wb") as f:
                f.write(stream)
            record(file=os.path.relpath(fpath, root), base=name, format="jpeg",
                   true_rate=rate, mode="jsteg_sequential",
                   payload_bits=written, capacity_bits=cap)

    manifest_path = os.path.join(root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_spatial = sum(1 for m in manifest if m["format"] in ("png", "bmp"))
    n_jpeg = sum(1 for m in manifest if m["format"] == "jpeg")
    print(f"[+] 测试集生成完毕: {root}")
    print(f"    图像总数 {len(manifest)}（空间域 {n_spatial}，JPEG {n_jpeg}）")
    print(f"    真值清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
