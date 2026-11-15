"""纯 Python baseline JPEG 编解码器。

为什么不用 Pillow：PIL 只提供像素级接口，拿不到量化后的整数 DCT 系数，
而 JSteg 类隐写与 DCT 直方图检测都必须在系数量级操作。这里实现一个
自给自足的 baseline（SOF0，顺序式，8 位样本，3 分量 YCbCr，采样 4:2:0/4:4:4）
编解码器，覆盖测试集生成与检测所需的全部能力；遇到 progressive/SOF2 等
非 baseline 码流会明确报错而不是静默给错结果。

系数约定：所有分析都基于“量化后的整数 DCT 系数”（即 JSteg 实际操作的数据）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dctn, idctn

# ---------------------------------------------------------------------------
# 常量：标准 zigzag、Annex K 量化表与 Huffman 表
# ---------------------------------------------------------------------------
ZIGZAG = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)

# ITU-T.81 Annex K 标准亮度/色度量化表
STD_LUM_QT = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)

STD_CHR_QT = np.array([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
], dtype=np.float64)


def quality_to_scale(quality: int) -> float:
    """libjpeg 风格的质量 -> 量化表缩放系数。"""
    q = int(np.clip(quality, 1, 100))
    if q < 50:
        return 5000.0 / q / 100.0
    return (200.0 - 2.0 * q) / 100.0


def scaled_quant_tables(quality: int) -> tuple[np.ndarray, np.ndarray]:
    s = quality_to_scale(quality)
    lum = np.clip(np.floor((STD_LUM_QT * s) + 0.5), 1, 255)
    chr_ = np.clip(np.floor((STD_CHR_QT * s) + 0.5), 1, 255)
    return lum.reshape(8, 8).astype(np.int32), chr_.reshape(8, 8).astype(np.int32)


# Annex K 标准 Huffman 表（bits[0..15] = 各长度码字数；values = 符号值）
STD_HUFF = {
    ("DC", 0): {
        # ITU-T.81 K.3.3.1 标准 DC 亮度 BITS（16 项，对应码长 1..16）
        "bits": [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    ("DC", 1): {
        # ITU-T.81 K.3.3.2 标准 DC 色度 BITS
        "bits": [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    ("AC", 0): {
        "bits": [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
        "values": [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
            0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
            0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
            0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
            0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
            0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
            0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
            0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
            0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
            0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
            0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
            0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
            0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
            0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
            0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa,
        ],
    },
    ("AC", 1): {
        "bits": [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
        "values": [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
            0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
            0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
            0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
            0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
            0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
            0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
            0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
            0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
            0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
            0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
            0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
            0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
            0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
            0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa,
        ],
    },
}


# ---------------------------------------------------------------------------
# Huffman 表结构
# ---------------------------------------------------------------------------
@dataclass
class HuffmanTable:
    bits: list[int]
    values: list[int]
    # 解码用：码长 -> 最小码字、首值偏移；符号 -> (码长, 码字)
    min_code: list[int] = field(default_factory=lambda: [-1] * 17)
    max_code: list[int] = field(default_factory=lambda: [-1] * 17)
    val_ptr: list[int] = field(default_factory=lambda: [0] * 17)
    enc_map: dict = field(default_factory=dict)

    def __post_init__(self):
        # ITU-T.81 Annex C.3 标准建表流程；bits 为 16 项，bits[l-1]=码长 l 的码字数
        code = 0
        k = 0
        for li in range(16):
            length = li + 1
            n = self.bits[li]
            if n:
                self.min_code[length] = code
                self.val_ptr[length] = k
                for _ in range(n):
                    self.enc_map[self.values[k]] = (length, code)
                    code += 1
                    k += 1
                self.max_code[length] = code - 1
            else:
                self.min_code[length] = 0
                self.val_ptr[length] = k
                self.max_code[length] = -1
            # 为下一码长做准备（最后一次移位无副作用，code 不再被读取）
            code <<= 1


def build_standard_huffman() -> dict[tuple[str, int], HuffmanTable]:
    return {key: HuffmanTable(**spec) for key, spec in STD_HUFF.items()}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Component:
    cid: int
    h_samp: int
    v_samp: int
    qt_id: int
    dc_ht_id: int
    ac_ht_id: int
    blocks: np.ndarray          # (bh, bw, 8, 8) 量化整数 DCT 系数（zigzag 已还原）
    stride_w: int               # 该分量像素宽（含填充）
    stride_h: int


@dataclass
class JpegImage:
    width: int
    height: int
    components: list[Component]
    quant_tables: dict          # id -> (8,8) 自然序量化表
    huff_tables: dict
    restart_interval: int
    subsampling: str            # "4:2:0" | "4:4:4"


class JpegError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 位流读写
# ---------------------------------------------------------------------------
class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def _fill_byte(self):
        if self.pos >= len(self.data):
            raise JpegError("熵码流意外结束")
        b = self.data[self.pos]
        self.pos += 1
        if b == 0xFF:
            marker = self.data[self.pos]
            self.pos += 1
            if marker != 0x00 and not (0xD0 <= marker <= 0xD7):
                raise JpegError(f"熵码流中出现标记 FF{marker:02X}")
        return b

    def read(self, n: int) -> int:
        # MSB 优先：缓冲高位是最早到达的比特
        while self.nbits < n:
            self.acc = (self.acc << 8) | self._fill_byte()
            self.nbits += 8
        self.nbits -= n
        val = (self.acc >> self.nbits) & ((1 << n) - 1)
        # 清掉已消费的高位，保持 acc 只保留未读比特
        self.acc &= (1 << self.nbits) - 1
        return val

    def receive_symbol(self, ht: HuffmanTable) -> int:
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.read(1)
            if code <= ht.max_code[length]:
                idx = ht.val_ptr[length] + (code - ht.min_code[length])
                return ht.values[idx]
        raise JpegError("Huffman 解码失败：码字未命中")

    def restart_align(self):
        self.acc = 0
        self.nbits = 0


def _extend(value: int, size: int) -> int:
    """JPEG 幅值扩展（差补码）。"""
    if size == 0:
        return 0
    if value < (1 << (size - 1)):
        return value - (1 << size) + 1
    return value


# ---------------------------------------------------------------------------
# 解码器
# ---------------------------------------------------------------------------
def _u16(data: bytes, i: int) -> tuple[int, int]:
    return (data[i] << 8) | data[i + 1], i + 2


def read_jpeg(path: str) -> JpegImage:
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        raise JpegError("不是 JPEG（缺 SOI）")

    pos = 2
    quant_tables: dict[int, np.ndarray] = {}
    huff_tables: dict[tuple[str, int], HuffmanTable] = {}
    frame = None
    comp_specs = None
    restart_interval = 0
    sos_info = None

    while pos < len(data):
        if data[pos] != 0xFF:
            raise JpegError(f"标记解析失败 @ {pos}")
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        marker = data[pos]
        pos += 1

        if marker == 0xD9:  # EOI
            break
        if 0xD0 <= marker <= 0xD7:  # RST
            continue
        if marker == 0xDA:  # SOS
            length, pos = _u16(data, pos)
            seg_end = pos + length - 2
            n_comp = data[pos]
            pos += 1
            sos = []
            for _ in range(n_comp):
                cid = data[pos]
                tbl = data[pos + 1]
                pos += 2
                sos.append((cid, tbl >> 4, tbl & 0x0F))
            # 跳过 Ss/Se/AhAl
            pos = seg_end
            sos_info = sos
            break

        length, pos = _u16(data, pos)
        seg_end = pos + length - 2

        if marker == 0xDB:  # DQT
            while pos < seg_end:
                pq_tq = data[pos]
                pq, tq = pq_tq >> 4, pq_tq & 0x0F
                pos += 1
                if pq != 0:
                    raise JpegError("仅支持 8 位量化表")
                flat = np.frombuffer(data, dtype=np.uint8, count=64, offset=pos).astype(np.int32)
                pos += 64
                natural = np.empty(64, dtype=np.int32)
                natural[ZIGZAG] = flat
                quant_tables[tq] = natural.reshape(8, 8)
        elif marker == 0xC4:  # DHT
            while pos < seg_end:
                tc_th = data[pos]
                tc, th = tc_th >> 4, tc_th & 0x0F
                pos += 1
                bits = list(data[pos:pos + 16])  # 码长 1..16 的码字数
                pos += 16
                n_values = sum(bits)
                values = list(data[pos:pos + n_values])
                pos += n_values
                kind = "DC" if tc == 0 else "AC"
                huff_tables[(kind, th)] = HuffmanTable(bits, values)
        elif marker == 0xC0:  # SOF0
            precision = data[pos]
            if precision != 8:
                raise JpegError(f"仅支持 8 位样本，收到 {precision} 位")
            height, pos = _u16(data, pos + 1)
            width, pos = _u16(data, pos)
            n_comp = data[pos]
            pos += 1
            if n_comp != 3:
                raise JpegError(f"仅支持 3 分量 YCbCr，收到 {n_comp} 分量")
            specs = []
            for _ in range(n_comp):
                cid = data[pos]
                samp = data[pos + 1]
                qt = data[pos + 2]
                pos += 3
                specs.append((cid, samp >> 4, samp & 0x0F, qt))
            frame = (width, height, specs)
        elif marker == 0xDD:  # DRI
            restart_interval, pos = _u16(data, pos)
        else:
            pos = seg_end  # 跳过不关心的段（APPn、COM、DQT 之外等）
        pos = seg_end

    if frame is None or sos_info is None:
        raise JpegError("缺少 SOF0 或 SOS，可能不是 baseline JPEG")
    if any(m == 0xC2 for m in ()):  # 占位：SOF2 在前面就会走 else，下面再校验
        pass

    width, height, specs = frame
    h_max = max(s[1] for s in specs)
    v_max = max(s[2] for s in specs)
    if (h_max, v_max) not in ((1, 1), (2, 2)):
        raise JpegError(f"不支持的采样因子组合 h_max={h_max} v_max={v_max}")
    subsampling = "4:2:0" if (h_max, v_max) == (2, 2) else "4:4:4"

    sos_map = {cid: (dc_id, ac_id) for cid, dc_id, ac_id in sos_info}
    total_mcu_x = math.ceil(width / 8 / h_max)
    total_mcu_y = math.ceil(height / 8 / v_max)
    components: list[Component] = []
    block_grids = {}
    for cid, hs, vs, qt_id in specs:
        dc_id, ac_id = sos_map[cid]
        # 每个 MCU 内该分量占 hs x vs 个块
        bw = total_mcu_x * hs
        bh = total_mcu_y * vs
        blocks = np.zeros((bh, bw, 8, 8), dtype=np.int32)
        block_grids[cid] = blocks
        components.append(Component(
            cid=cid, h_samp=hs, v_samp=vs, qt_id=qt_id,
            dc_ht_id=dc_id, ac_ht_id=ac_id, blocks=blocks,
            stride_w=bw * 8, stride_h=bh * 8,
        ))

    # 熵解码
    reader = _BitReader(data[pos:])
    prev_dc = {c.cid: 0 for c in components}
    mcu_count = 0
    for my in range(total_mcu_y):
        for mx in range(total_mcu_x):
            if restart_interval and mcu_count > 0 and mcu_count % restart_interval == 0:
                reader.restart_align()
                # 吃掉 RST 标记
                while reader.pos < len(reader.data) and reader.data[reader.pos] == 0xFF:
                    m2 = reader.data[reader.pos + 1]
                    if 0xD0 <= m2 <= 0xD7:
                        reader.pos += 2
                        break
                    break
                prev_dc = {c.cid: 0 for c in components}
            for comp in components:
                for vy in range(comp.v_samp):
                    for vx in range(comp.h_samp):
                        bx = mx * comp.h_samp + vx
                        by = my * comp.v_samp + vy
                        if bx >= comp.blocks.shape[1] or by >= comp.blocks.shape[0]:
                            continue
                        coeff = _decode_block(reader, comp, huff_tables, prev_dc)
                        comp.blocks[by, bx] = coeff
            mcu_count += 1

    return JpegImage(
        width=width, height=height, components=components,
        quant_tables=quant_tables, huff_tables=huff_tables,
        restart_interval=restart_interval, subsampling=subsampling,
    )


def _decode_block(reader: _BitReader, comp: Component,
                  huff_tables: dict, prev_dc: dict) -> np.ndarray:
    dc_ht = huff_tables[("DC", comp.dc_ht_id)]
    ac_ht = huff_tables[("AC", comp.ac_ht_id)]
    coeff_zig = np.zeros(64, dtype=np.int32)

    # DC
    sym = reader.receive_symbol(dc_ht)
    diff = _extend(reader.read(sym), sym)
    prev_dc[comp.cid] += diff
    coeff_zig[0] = prev_dc[comp.cid]

    # AC
    k = 1
    while k < 64:
        rs = reader.receive_symbol(ac_ht)
        run, size = rs >> 4, rs & 0x0F
        if rs == 0x00:  # EOB
            break
        if rs == 0xF0:  # ZRL
            k += 16
            continue
        k += run
        if k >= 64:
            break
        coeff_zig[k] = _extend(reader.read(size), size)
        k += 1

    natural = np.empty(64, dtype=np.int32)
    natural[ZIGZAG] = coeff_zig
    return natural.reshape(8, 8)


# ---------------------------------------------------------------------------
# 像素重建（供校准图重新编码使用）
# ---------------------------------------------------------------------------
def decode_to_pixels(jpg: JpegImage) -> np.ndarray:
    """量化系数 -> RGB uint8，返回 HxWx3（已裁掉填充）。"""
    planes = {}
    for comp in jpg.components:
        qt = jpg.quant_tables[comp.qt_id]
        bh, bw = comp.blocks.shape[:2]
        plane = np.zeros((bh * 8, bw * 8), dtype=np.float64)
        for by in range(bh):
            for bx in range(bw):
                deq = comp.blocks[by, bx].astype(np.float64) * qt
                block = idctn(deq, type=2, norm="ortho") + 128.0
                plane[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8] = block
        planes[comp.cid] = np.clip(np.round(plane), 0, 255).astype(np.uint8)

    y = planes[jpg.components[0].cid]
    cb_full, cr_full = _upsample(planes[jpg.components[1].cid],
                                  planes[jpg.components[2].cid],
                                  jpg, y.shape)
    rgb = _ycbcr_to_rgb(y, cb_full, cr_full)
    return rgb[:jpg.height, :jpg.width, :]


def _upsample(cb: np.ndarray, cr: np.ndarray, jpg: JpegImage,
              target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = target_shape
    if cb.shape == target_shape:
        return cb[:h, :w], cr[:h, :w]
    # 4:2:0 -> 最近邻上采样（编解码两端一致即可，校准只关心统计量）
    ys = (np.arange(h) * cb.shape[0] // h)
    xs = (np.arange(w) * cb.shape[1] // w)
    return cb[np.ix_(ys, xs)], cr[np.ix_(ys, xs)]


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    yf = y.astype(np.float64)
    pb = cb.astype(np.float64) - 128.0
    pr = cr.astype(np.float64) - 128.0
    r = yf + 1.402 * pr
    g = yf - 0.344136 * pb - 0.714136 * pr
    b = yf + 1.772 * pb
    return np.clip(np.round(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------
class _BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def write_bits(self, value: int, n: int):
        self.acc = (self.acc << n) | (value & ((1 << n) - 1))
        self.nbits += n
        while self.nbits >= 8:
            self.nbits -= 8
            byte = (self.acc >> self.nbits) & 0xFF
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0x00)  # 字节填充

    def flush(self):
        if self.nbits > 0:
            self.write_bits(0x7F, 8 - self.nbits)  # 填 1
        self.nbits = 0
        self.acc = 0


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, np.clip(cb, 0, 255), np.clip(cr, 0, 255)


def _subsample_420(plane: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """全分辨率像素面 -> 4:2:0 色度面（每维减半，尺寸为 8 的倍数）。

    bh/bw 为亮度块网格尺寸；色度块网格为其一半，色度像素面 = (bh/2*8, bw/2*8)。
    2x2 平均，边缘复制。
    """
    pad = plane.copy()
    if pad.shape[0] % 2:
        pad = np.vstack([pad, pad[-1:, :]])
    if pad.shape[1] % 2:
        pad = np.hstack([pad, pad[:, -1:]])
    small = (pad[0::2, 0::2] + pad[0::2, 1::2]
             + pad[1::2, 0::2] + pad[1::2, 1::2]) * 0.25
    target_h, target_w = bh * 4, bw * 4
    out = np.zeros((target_h, target_w), dtype=np.float64)
    th = min(target_h, small.shape[0])
    tw = min(target_w, small.shape[1])
    out[:th, :tw] = small[:th, :tw]
    if th < target_h:
        out[th:, :tw] = small[-1:, :tw]
    if tw < target_w:
        out[:, tw:] = out[:, tw - 1:tw]
    return out


def _pixels_to_coeffs(plane: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """像素面（尺寸为 8 的倍数）-> (bh,bw,8,8) 量化整数 DCT 系数。

    轴序：轴 0 为块行（垂直），轴 1 为块列（水平），与解码器一致。
    """
    h, w = plane.shape
    bh, bw = h // 8, w // 8
    blocks = np.zeros((bh, bw, 8, 8), dtype=np.int32)
    for by in range(bh):
        for bx in range(bw):
            block = plane[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8] - 128.0
            coeff = dctn(block, type=2, norm="ortho") / qt
            blocks[by, bx] = np.round(coeff).astype(np.int32)
    return blocks


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _dqt_segment(qt_id: int, qt: np.ndarray) -> bytes:
    # DQT 段按 zigzag 顺序书写：取自然序表中索引为 ZIGZAG[k] 的元素
    zig = qt.reshape(-1)[ZIGZAG].astype(np.uint8).tobytes()
    return _segment(0xDB, bytes([qt_id]) + zig)


def _dht_segment(tc: int, th: int, ht: HuffmanTable) -> bytes:
    body = bytes([(tc << 4) | th]) + bytes(ht.bits) + bytes(ht.values)
    return _segment(0xC4, body)


def _size_and_bits(value: int) -> tuple[int, int]:
    if value == 0:
        return 0, 0
    mag = abs(value)
    size = mag.bit_length()
    if value > 0:
        bits = value
    else:
        bits = value + (1 << size) - 1
    return size, bits


def _encode_block_coeffs(coeff: np.ndarray, writer: _BitWriter,
                         dc_ht: HuffmanTable, ac_ht: HuffmanTable,
                         prev_dc: dict, cid: int):
    # 自然序 -> zigzag 序列：zig[k] = natural[ZIGZAG[k]]
    natural = coeff.reshape(-1)
    zig = natural[ZIGZAG]
    # DC（差分；prev_dc 为跨块共享的可变状态）
    diff = int(zig[0]) - prev_dc[cid]
    prev_dc[cid] = int(zig[0])
    size, bits = _size_and_bits(diff)
    length, code = dc_ht.enc_map[size]
    writer.write_bits(code, length)
    if size:
        writer.write_bits(bits, size)
    # AC
    run = 0
    for k in range(1, 64):
        v = int(zig[k])
        if v == 0:
            run += 1
            continue
        while run > 15:
            length, code = ac_ht.enc_map[0xF0]
            writer.write_bits(code, length)
            run -= 16
        size, bits = _size_and_bits(v)
        length, code = ac_ht.enc_map[(run << 4) | size]
        writer.write_bits(code, length)
        writer.write_bits(bits, size)
        run = 0
    if run:  # EOB
        length, code = ac_ht.enc_map[0x00]
        writer.write_bits(code, length)


def encode_jpeg(rgb: np.ndarray, quality: int = 85,
                subsampling: str = "4:2:0") -> bytes:
    """RGB uint8 图像 -> baseline JPEG 字节流。"""
    h, w = rgb.shape[:2]
    lum_qt, chr_qt = scaled_quant_tables(quality)
    huff = build_standard_huffman()

    h_max = v_max = 2 if subsampling == "4:2:0" else 1
    mcu_x = math.ceil(w / 8 / h_max)
    mcu_y = math.ceil(h / 8 / v_max)
    # 分量块网格尺寸：4:2:0 时色度块网格为亮度的一半
    y_bw, y_bh = mcu_x * h_max, mcu_y * v_max
    c_bw, c_bh = mcu_x, mcu_y
    yf, cbf, crf = _rgb_to_ycbcr(rgb)
    # 填充到块边界（行列都用边缘复制）
    def pad_to(plane: np.ndarray, bh: int, bw: int) -> np.ndarray:
        target_h, target_w = bh * 8, bw * 8
        out = np.zeros((target_h, target_w), dtype=np.float64)
        out[:plane.shape[0], :plane.shape[1]] = plane
        if plane.shape[0] < target_h:
            out[plane.shape[0]:, :plane.shape[1]] = plane[-1:, :]
        if plane.shape[1] < target_w:
            out[:, plane.shape[1]:] = out[:, plane.shape[1] - 1:plane.shape[1]]
        return out

    y_pad = pad_to(yf, y_bh, y_bw)
    if subsampling == "4:2:0":
        cb_small = _subsample_420(cbf, y_bh, y_bw)
        cr_small = _subsample_420(crf, y_bh, y_bw)
    else:
        cb_small, cr_small = cbf, crf
    cb_pad = pad_to(cb_small, c_bh, c_bw)
    cr_pad = pad_to(cr_small, c_bh, c_bw)

    y_coeff = _pixels_to_coeffs(y_pad, lum_qt)
    cb_coeff = _pixels_to_coeffs(cb_pad, chr_qt)
    cr_coeff = _pixels_to_coeffs(cr_pad, chr_qt)

    comp_coeffs = [(1, y_coeff, 0, 0, 0, h_max, v_max),
                   (2, cb_coeff, 1, 1, 1, 1, 1),
                   (3, cr_coeff, 1, 1, 1, 1, 1)]
    return _encode_coeffs(w, h, comp_coeffs, {0: lum_qt, 1: chr_qt}, huff,
                          subsampling)


def _encode_coeffs(width: int, height: int, comp_coeffs, quant_tables: dict,
                   huff: dict, subsampling: str) -> bytes:
    out = bytearray(b"\xff\xd8")  # SOI
    # APP0 JFIF
    app0 = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    out += _segment(0xE0, app0)
    # DQT
    for qt_id, qt in quant_tables.items():
        out += _dqt_segment(qt_id, qt)
    # DHT：四张标准表
    out += _dht_segment(0, 0, huff[("DC", 0)])
    out += _dht_segment(1, 0, huff[("AC", 0)])
    out += _dht_segment(0, 1, huff[("DC", 1)])
    out += _dht_segment(1, 1, huff[("AC", 1)])
    # SOF0
    sof = bytearray()
    sof += bytes([8])
    sof += height.to_bytes(2, "big") + width.to_bytes(2, "big")
    sof += bytes([len(comp_coeffs)])
    for cid, _coeff, qt_id, _dc, _ac, hs, vs in comp_coeffs:
        sof += bytes([cid, (hs << 4) | vs, qt_id])
    out += _segment(0xC0, bytes(sof))
    # SOS
    sos = bytearray([len(comp_coeffs)])
    for cid, _coeff, _qt, dc_id, ac_id, _hs, _vs in comp_coeffs:
        sos += bytes([cid, (dc_id << 4) | ac_id])
    sos += bytes([0x00, 0x3F, 0x00])  # Ss=0 Se=63 Ah/Al=0
    out += _segment(0xDA, bytes(sos))

    # 熵数据
    writer = _BitWriter()
    prev_dc = {cid: 0 for cid, *_ in comp_coeffs}
    coeff_map = {cid: coeff for cid, coeff, *_ in comp_coeffs}
    h_max = max(t[5] for t in comp_coeffs)
    v_max = max(t[6] for t in comp_coeffs)
    mcu_x = math.ceil(width / 8 / h_max)
    mcu_y = math.ceil(height / 8 / v_max)
    for my in range(mcu_y):
        for mx in range(mcu_x):
            for cid, coeff, _qt, dc_id, ac_id, hs, vs in comp_coeffs:
                dc_ht = huff[("DC", dc_id)]
                ac_ht = huff[("AC", ac_id)]
                for vy in range(vs):
                    for vx in range(hs):
                        bx = mx * hs + vx
                        by = my * vs + vy
                        if bx >= coeff.shape[1] or by >= coeff.shape[0]:
                            continue
                        _encode_block_coeffs(
                            coeff[by, bx], writer, dc_ht, ac_ht, prev_dc, cid)
    writer.flush()
    out += writer.out
    out += b"\xff\xd9"  # EOI
    return bytes(out)


def encode_coeffs_jpeg(jpg: JpegImage, coeff_override: dict | None = None) -> bytes:
    """用给定（或 JpegImage 自带的）量化系数重新编码为 JPEG 字节流。

    coeff_override: {分量 cid: (bh,bw,8,8) int32 系数}，用于隐写后回写。
    """
    comp_coeffs = []
    for comp in jpg.components:
        coeff = coeff_override[comp.cid] if coeff_override else comp.blocks
        comp_coeffs.append((comp.cid, coeff, comp.qt_id, comp.dc_ht_id,
                            comp.ac_ht_id, comp.h_samp, comp.v_samp))
    return _encode_coeffs(jpg.width, jpg.height, comp_coeffs,
                          jpg.quant_tables, jpg.huff_tables, jpg.subsampling)


def jsteg_capacity(jpg: JpegImage) -> int:
    """JSteg 可嵌入位数：所有非 0 的 AC 系数（DC 不动）。

    经典 JSteg 只跳过 0 系数（避免把 0 变成 1 改变稀疏性），其余 AC
    系数（含 ±1）都在 LSB 值对内替换。
    """
    total = 0
    for comp in jpg.components:
        c = comp.blocks
        ac = c[:, :, :, :].copy()
        ac[:, :, 0, 0] = 0  # 排除 DC
        total += int(np.count_nonzero(ac != 0))
    return total


def all_ac_coeffs(jpg: JpegImage) -> np.ndarray:
    """展平的 AC 系数（含 0），用于直方图检测。"""
    vals = []
    for comp in jpg.components:
        c = comp.blocks
        mask = np.ones((8, 8), dtype=bool)
        mask[0, 0] = False
        vals.append(c[:, :, mask].reshape(-1))
    return np.concatenate(vals) if vals else np.array([], dtype=np.int32)
