"""LSB 嵌入器（仅用于生成带真值的测试集与验证检测算法）。

- 空间域：PNG/BMP 像素 LSB 替换，支持顺序/伪随机散布
- JPEG：JSteg，对非 0 非 1 的 AC 系数做 LSB 替换（DC、0、1 不动）
"""
from __future__ import annotations

import numpy as np

from .jpeg_codec import JpegImage, encode_coeffs_jpeg


def embed_spatial_lsb(arr: np.ndarray, payload_bits: np.ndarray,
                      mode: str = "sequential",
                      seed: int = 12345) -> tuple[np.ndarray, int, int]:
    """在 HxWxC uint8 图像的 LSB 平面嵌入。

    返回 (载密数组, 实际写入位数, 容量位数)。payload 超出容量时抛错。
    """
    stego = arr.copy()
    flat = stego.reshape(-1)
    capacity = flat.size
    if payload_bits.size > capacity:
        raise ValueError(f"载荷 {payload_bits.size} bit 超过容量 {capacity} bit")

    if mode == "sequential":
        indices = np.arange(payload_bits.size)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.permutation(capacity)[:payload_bits.size]
    else:
        raise ValueError(f"未知嵌入模式 {mode}")

    # 向量化：清 LSB 再或入载荷
    flat[indices] = (flat[indices] & np.uint8(0xFE)) | payload_bits.astype(np.uint8)
    return stego, int(payload_bits.size), int(capacity)


def make_payload(n_bits: int, seed: int = 42) -> np.ndarray:
    """生成模拟秘密消息的伪随机比特流。"""
    return np.random.default_rng(seed).integers(0, 2, size=n_bits, dtype=np.uint8)


def embed_jsteg(jpg: JpegImage, payload_bits: np.ndarray,
                mode: str = "sequential", seed: int = 98765
                ) -> tuple[bytes, int, int]:
    """对 JpegImage 执行 JSteg 嵌入并重新编码为字节流。

    返回 (JPEG 字节, 写入位数, 容量)。
    """
    # 按编解码相同的遍历顺序收集可嵌入系数位置 (comp, by, bx, i, j)
    # 经典 JSteg：跳过 DC 与值为 0 的 AC，其余都可做 LSB 替换
    slots: list[tuple[int, int, int, int, int]] = []
    for ci, comp in enumerate(jpg.components):
        c = comp.blocks
        bh, bw = c.shape[:2]
        for by in range(bh):
            for bx in range(bw):
                for i in range(8):
                    for j in range(8):
                        if i == 0 and j == 0:
                            continue
                        v = c[by, bx, i, j]
                        if v != 0:
                            slots.append((ci, by, bx, i, j))
    capacity = len(slots)
    if payload_bits.size > capacity:
        raise ValueError(f"JSteg 载荷 {payload_bits.size} bit 超过容量 {capacity} bit")

    if mode == "sequential":
        chosen = slots[:payload_bits.size]
    else:
        rng = np.random.default_rng(seed)
        pick = rng.permutation(capacity)[:payload_bits.size]
        chosen = [slots[k] for k in sorted(pick)]

    new_coeffs = {comp.cid: comp.blocks.copy() for comp in jpg.components}
    cid_by_index = {i: comp.cid for i, comp in enumerate(jpg.components)}
    written = 0
    for bit, (ci, by, bx, i, j) in zip(payload_bits, chosen):
        cid = cid_by_index[ci]
        v = int(new_coeffs[cid][by, bx, i, j])
        # LSB 替换（绝对值 LSB，符号保持）：偶数 <-> 相邻奇数
        new_v = (v & ~1) | int(bit)
        new_coeffs[cid][by, bx, i, j] = new_v
        written += 1

    stream = encode_coeffs_jpeg(jpg, coeff_override=new_coeffs)
    return stream, written, capacity
