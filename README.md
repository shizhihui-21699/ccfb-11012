# LSB / JSteg 隐写检测工具集

针对 **PNG / BMP 的 LSB 空间域隐写**与 **JPEG 的 JSteg（DCT 系数量级 LSB 替换）隐写**
的批量检测工具。实现卡方分析、RS 分析、样本对分析（SPA）三种空间域方法，以及基于
DCT 系数直方图的 JPEG 检测，输出每张图的可疑度、估计嵌入率与嵌入量，并生成带热力图的
HTML 报告。

> 用途：防御性安全 / 数据外泄排查。配套的嵌入器（`stego_embed.py`）仅用于构造带真值的
> 测试集与验证检测算法。

## 目录结构

```
stegdetect/
  image_io.py       图像读取与格式识别（PNG/BMP 走 Pillow，JPEG 走自研编解码器）
  chi_square.py     Westfeld-Pfitzmann 卡方分析（全图 + 累积滑窗 + 配对估计）
  rs_analysis.py    Fridrich RS 分析（混合翻转网格数值求根）
  sample_pairs.py   Dumitrescu 样本对分析（SPA，二次方程）
  jpeg_codec.py     纯 Python baseline JPEG 编解码器（量化 DCT 系数量级访问）
  jpeg_dct.py       DCT 系数直方图检测（卡方 + h(1) 漂移 + 裁剪校准）
  stego_embed.py    LSB / JSteg 嵌入器（测试集生成用）
  calibration.py    经验校准映射加载
  calibration.json  在测试集上拟合的保序回归校准参数
  detector.py       单图/批量流水线与多方法评分融合
  report.py         CSV + 带热力图的自包含 HTML 报告
  cli.py            命令行入口
tools/
  generate_testset.py  批量生成带真值测试图（干净对照 + 多嵌入率载密图）
  calibrate.py         在真值集上评估各方法并拟合校准
  evaluate.py          检测结果 vs 真值的混淆矩阵与指标
```

## 快速开始

```bash
# 依赖安装（项目内虚拟环境，不污染系统 Python）
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy pillow matplotlib

# 1. 生成带真值测试集（126 张：18 干净 + 78 空间域载密 + 30 JPEG 载密）
.venv/bin/python tools/generate_testset.py

# 2. 批量检测目录并出报告
.venv/bin/python -m stegdetect.cli detect data/test_images --out reports/run1

# 单张图
.venv/bin/python -m stegdetect.cli detect path/to/image.png --no-report
```

报告产物：`report.html`（自包含，含 5 张热力图/分布图）、`detection_results.csv`（逐图明细）、
`assets/*.png`（图表）。

## 方法与参数

### 1. 卡方分析（`chi_square.py`）

LSB 替换只在值对 (2k, 2k+1) 内部互换，完全嵌入后每对两个桶的频数趋于相等。对观测直方图与
"配对均分" 期望做卡方拟合优度检验，p 值越接近 1 越可疑。

- **全图 p 值**：256 级直方图，期望频数 <5 的值对剔除并把残差合并为一个桶，自由度 = 桶数−1。
- **累积滑窗（20 段）**：逐通道取扫描前缀（顺序嵌入写满一个通道才进下一个，必须逐通道、
  取各通道最大 p，否则未嵌入通道会稀释信号），首次 p≥0.95 的位置估计顺序嵌入长度。
- **配对不平衡估计**：用值对内部能量与跨对能量之比反推嵌入率，对全域均匀（随机散布）嵌入有效。

适用边界：卡方要求图像 LSB 原本具有结构（连续色调照片的相邻像素强相关）。对 LSB 近似随机的
纯白噪声、或大面积纯色饱和图，p 值不敏感——这是方法本身的特性，非实现缺陷。

### 2. RS 分析（`rs_analysis.py`）

对水平 + 垂直 4 像素组、掩码 m=(0,1,1,0)，判别函数 f(G)=Σ|x_{i+1}−x_i|。定义两种翻转
F1（LSB 翻转）与 F−1（反向翻转），f 减小为奇异组。在嵌入率网格 t∈[0,1] 上对原图与其全嵌入
图分别施加混合翻转 F(t)（以概率 t 取 F1，否则 F−1），实测 d(t)、d′(t) 两条曲线，数值求
d(t)=d′(t) 的最小非负根 z，再由 z=1−(1−p)² 反解 p=1−√(1−z)。平局（f 不变）不计入。

适用边界：RS 的二次曲线模型在过度平滑、F−1 奇异组比例接近 0 的图上会退化。本实现保留完整
理论框架与曲线（报告可可视化），但在融合中权重最低，主力交给 SPA 与卡方。

### 3. 样本对分析 SPA（`sample_pairs.py`）

取水平 + 垂直相邻像素对，按差值奇偶与首像素奇偶分类为 D / X_n / Y_n / Z_n / W_n 多集合，
对 LSB 替换建立迹条件，导出关于嵌入率 p 的二次方程

```
a = 2 Σ W_n        b = 2(Σ Z_n − Σ W_n) − D        c = −Σ Z_n
```

取 [0,1] 内最接近 0 的根；根被裁剪时标记 `spa_clipped` 表示低置信。负差对的 n 符号按
diff=2n±1 分别处理（这是容易出错的点，已在受控数据上验证）。

### 4. JPEG DCT 直方图检测（`jpeg_dct.py`）

Pillow 不暴露量化 DCT 系数，因此 `jpeg_codec.py` 实现了自给自足的 baseline JPEG 编解码器
（SOF0 / 8 位 / YCbCr / 4:2:0 与 4:4:4，标准 Annex K 量化表与 Huffman 表），产出的 JPEG
已通过 Pillow 标准解码器交叉验证（36/36 可正常读取）。JSteg 跳过 0 系数与 DC，对其余 AC
系数做 LSB 替换。三组互补指标：

- **低幅值值对卡方**：只取 |系数|≤5 的桶（高频量化表会产生数万量级的 6/7 饱和桶，纳入会
  淹没低幅值桶的均衡信号），跳过含 0 的对。
- **h(1) 漂移**：JSteg 把部分 ±2 拉成 ±1，h(1)/h(2) 随嵌入率下降。
- **裁剪校准（Fridrich 校准思想）**：解码后裁掉 4 行 4 列、用相同量化表重编码得到近似
  "无嵌入基准"，比较本图与校准图的 h1 漂移和直方图归一化 L1 距离，消除图像内容依赖。

标定常数（在 256px 多基底测试集实测）：干净漂移 ≈0、完全嵌入 ≈−0.60；直方图距离干净
≈0.008、完全嵌入 ≈0.055。小图（AC<2 万）按样本量折减置信度。

## 评分融合与判定阈值

空间域融合权重：SPA 0.45、卡方配对估计 0.40、RS 0.15（各方法先经过 `calibration.json`
保序回归校准），顺序滑窗检出时取 max。可疑度 = 0.70·√(校准嵌入率) + 0.18·卡方p + 0.12·滑窗。

判定阈值（在测试集上数据驱动标定，以零误报为前提）：

| 可疑度 | 判定 |
|---|---|
| < 0.30 | clean |
| 0.30 – 0.55 | suspicious |
| ≥ 0.55 | likely_stego |

JPEG 阈值同表，融合权重 h1 漂移 0.5 / 直方图距离 0.35 / 卡方 p 0.15。

## 测试集与实测性能

测试图为程序化合成的类自然图像（平滑插值随机场、多尺度分形、柔边斑块、正弦条纹、非平稳
混合），固定随机种子可复现。嵌入率：空间域 {5%,10%,25%,50%,90%}（顺序 + 随机散布两种模式），
JPEG {5%,15%,30%,60%,90%}，每种另含干净对照。

在 126 张测试集上的实测结果（`tools/evaluate.py`）：

| 格式 | 精确率 | 召回率 | 嵌入率 MAE | 干净图误报 |
|---|---|---|---|---|
| PNG | 1.00 | 0.35 | 0.154 | 0/6 |
| BMP | 1.00 | 0.67 | 0.175 | 0/6 |
| JPEG | 1.00 | 0.60 | 0.095 | 0/6 |

**能力边界（如实说明）**：

- 所有格式**零误报**，嵌入率 ≥0.5 的载密图基本全检出，JPEG ≥0.30 全检出。
- 嵌入率 ≤0.1 的低载荷与自然统计涨落重叠，刻意不强行报警（宁可漏报不误报）。
- 空间域召回受合成基底 LSB 结构偏弱限制；在真实连续色调照片上卡方与 RS 的灵敏度通常更高。
- 检测针对 LSB 替换 / JSteg；对 LSB matching（±1 随机增减）、F5、OutGuess 等需另配方法。

## 复现实验

```bash
.venv/bin/python tools/generate_testset.py     # 生成 data/test_images + manifest.json
.venv/bin/python tools/calibrate.py            # 评估原始方法并刷新 calibration.json
.venv/bin/python tools/evaluate.py             # 真值对比、混淆矩阵、MAE
.venv/bin/python -m stegdetect.cli detect data/test_images --out reports/final
```
