"""LSB 隐写检测工具包。

模块划分
--------
- image_io     : 图像读取与格式识别
- chi_square   : Westfeld-Pfitzmann 卡方分析（含滑窗）
- rs_analysis  : Fridrich RS 分析
- sample_pairs : Dumitrescu 样本对分析 (SPA)
- jpeg_codec   : 纯 Python baseline JPEG 编解码（DCT 系数级访问）
- jpeg_dct     : 基于 DCT 系数直方图的 JSteg 检测
- detector     : 单图/批量检测流水线与评分融合
- report       : CSV / HTML 报告与热力图
"""

__version__ = "1.0.0"
