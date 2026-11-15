"""检测报告输出：CSV + 带热力图的自包含 HTML。"""
from __future__ import annotations

import csv
import html
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .detector import DetectionResult, THRESHOLD_SUSPICIOUS, THRESHOLD_HIGH

# 图表内部用英文标签，避免无中文字体时渲染成方块；HTML 正文保持中文
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

_HEAT_CMAP = "magma"


def write_csv(results: list[DetectionResult], path: str):
    if not results:
        return
    fields = list(results[0].__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in results:
            writer.writerow([getattr(r, k) if getattr(r, k) is not None else ""
                             for k in fields])


def _heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
             title: str, out_path: str, fmt: str = "{:.2f}",
             vmin: float = 0.0, vmax: float = 1.0, cmap: str = _HEAT_CMAP):
    n_rows, n_cols = matrix.shape
    fig_h = max(2.2, 0.42 * n_rows + 1.4)
    fig_w = max(6.0, 1.35 * n_cols + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "-"
                color = "black"
            else:
                txt = fmt.format(v)
                color = "white" if v > (vmin + vmax) * 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)
    ax.set_title(title, fontsize=11, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _distribution_plot(results: list[DetectionResult], out_path: str):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    scores = np.array([r.suspicion for r in results if r.verdict != "error"])
    ax.hist(scores, bins=np.linspace(0, 1, 21), color="#6a51a3",
            edgecolor="white", alpha=0.9)
    ax.axvline(THRESHOLD_SUSPICIOUS, color="#e6550d", ls="--", lw=1.4,
               label=f"suspicious >= {THRESHOLD_SUSPICIOUS}")
    ax.axvline(THRESHOLD_HIGH, color="#a50f15", ls="--", lw=1.4,
               label=f"likely_stego >= {THRESHOLD_HIGH}")
    ax.set_xlabel("Suspicion score")
    ax.set_ylabel("Image count")
    ax.set_title("Suspicion score distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_report(results: list[DetectionResult], out_dir: str,
                 title: str = "LSB 隐写检测报告") -> str:
    os.makedirs(out_dir, exist_ok=True)
    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    csv_path = os.path.join(out_dir, "detection_results.csv")
    write_csv(results, csv_path)

    valid = [r for r in results if r.verdict != "error"]
    spatial = [r for r in valid if r.format in ("png", "bmp")]
    jpegs = [r for r in valid if r.format == "jpeg"]
    charts: list[str] = []

    # ---- 空间域方法可疑度热力图 ----
    if spatial:
        labels = [os.path.basename(r.file) for r in spatial]
        score_mat = np.array([
            [_f(r.chi_global_p),
             _f(r.rs_rate) * 1.25,
             _f(r.spa_rate) * 1.25,
             r.suspicion]
            for r in spatial
        ]).clip(0, 1)
        p1 = os.path.join(assets, "heatmap_spatial_scores.png")
        _heatmap(score_mat, labels,
                 ["chi-square p", "RS score", "SPA score", "FUSED"],
                 "Spatial images - method suspicion scores", p1)
        charts.append(("assets/heatmap_spatial_scores.png",
                       "空间域各方法可疑度热力图（颜色越亮越可疑，FUSED 为融合分）"))

        rate_mat = np.array([
            [_f(r.chi_pair_estimate), _f(r.rs_rate), _f(r.spa_rate),
             r.estimated_rate]
            for r in spatial
        ])
        p2 = os.path.join(assets, "heatmap_spatial_rates.png")
        _heatmap(rate_mat, labels,
                 ["chi pair-est", "RS rate", "SPA rate", "FUSED rate"],
                 "Spatial images - estimated embedding rate", p2)
        charts.append(("assets/heatmap_spatial_rates.png",
                       "空间域各方法嵌入率估计热力图（0~1，占 LSB 容量比例）"))

    # ---- JPEG DCT 热力图 ----
    if jpegs:
        labels = [os.path.basename(r.file) for r in jpegs]
        score_mat = np.array([
            [_f(r.jpeg_chi_p),
             np.clip(_f(r.jpeg_h1_drift) / 0.25, 0, 1),
             np.clip(_f(r.jpeg_hist_distance) / 0.08, 0, 1),
             r.suspicion]
            for r in jpegs
        ])
        p3 = os.path.join(assets, "heatmap_jpeg_scores.png")
        _heatmap(score_mat, labels,
                 ["DCT chi-square p", "h(1) drift", "hist L1", "FUSED"],
                 "JPEG images - DCT histogram detection scores", p3)
        charts.append(("assets/heatmap_jpeg_scores.png",
                       "JPEG DCT 直方图检测热力图（卡方 p 值 / h(1) 漂移 / 校准直方图距离 / 融合分）"))

    # ---- 总体可疑度大图 ----
    all_labels = [os.path.basename(r.file) for r in valid]
    fused = np.array([r.suspicion for r in valid]).reshape(-1, 1)
    p4 = os.path.join(assets, "heatmap_fused.png")
    _heatmap(fused, all_labels, ["FUSED suspicion"],
             "All images - fused suspicion score", p4)
    charts.append(("assets/heatmap_fused.png", "全部图像融合可疑度热力图"))

    p5 = os.path.join(assets, "score_distribution.png")
    _distribution_plot(valid, p5)
    charts.append(("assets/score_distribution.png", "可疑度分布直方图与判定阈值"))

    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(title, results, charts, csv_path))
    return html_path


def _f(v) -> float:
    return float(v) if v is not None else np.nan


def _render_html(title: str, results: list[DetectionResult],
                 charts: list[tuple[str, str]], csv_name: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(results)
    n_clean = sum(1 for r in results if r.verdict == "clean")
    n_susp = sum(1 for r in results if r.verdict == "suspicious")
    n_stego = sum(1 for r in results if r.verdict == "likely_stego")
    n_err = sum(1 for r in results if r.verdict == "error")

    rows = []
    for r in sorted(results, key=lambda x: -x.suspicion):
        cls = {"clean": "v-clean", "suspicious": "v-susp",
               "likely_stego": "v-stego", "error": "v-err"}[r.verdict]
        def pct(v):
            return "—" if v is None else f"{v * 100:.1f}%"
        def num(v):
            return "—" if v is None else f"{v:.3f}"
        rows.append(f"""
        <tr class="{cls}">
          <td class="fname">{html.escape(os.path.basename(r.file))}</td>
          <td>{r.format.upper()}</td>
          <td>{r.width}×{r.height}</td>
          <td><b>{r.suspicion:.3f}</b></td>
          <td><span class="badge {cls}">{r.verdict}</span></td>
          <td>{pct(r.estimated_rate)}</td>
          <td>{r.estimated_payload_bits:,}</td>
          <td>{num(r.chi_global_p)}</td>
          <td>{pct(r.chi_sequential_fraction)}</td>
          <td>{num(r.rs_rate)}</td>
          <td>{num(r.spa_rate)}{"*" if r.spa_clipped else ""}</td>
          <td>{num(r.jpeg_chi_p)}</td>
          <td>{num(r.jpeg_h1_drift)}</td>
          <td class="err">{html.escape(r.error) if r.error else ""}</td>
        </tr>""")

    chart_html = "\n".join(
        f'<figure><img src="{src}" alt="{html.escape(cap)}">'
        f"<figcaption>{html.escape(cap)}</figcaption></figure>"
        for src, cap in charts)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ --bg:#f7f7fb; --card:#fff; --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 22px 60px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:34px 0 12px; border-left:4px solid #6a51a3;
        padding-left:10px; }}
  .meta {{ color:var(--muted); font-size:13px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:14px 20px; min-width:130px; }}
  .card .k {{ color:var(--muted); font-size:12px; }}
  .card .v {{ font-size:26px; font-weight:700; }}
  figure {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
            padding:14px; margin:16px 0; }}
  figure img {{ width:100%; height:auto; border-radius:6px; }}
  figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
  table {{ border-collapse:collapse; width:100%; background:var(--card);
           border:1px solid var(--line); border-radius:10px; overflow:hidden;
           font-size:12.5px; }}
  th, td {{ padding:7px 9px; border-bottom:1px solid var(--line);
            text-align:right; white-space:nowrap; }}
  th {{ background:#f0eef7; font-weight:600; position:sticky; top:0; }}
  td.fname, th:first-child {{ text-align:left; white-space:normal; min-width:180px; }}
  tr.v-stego {{ background:#fde8e8; }}
  tr.v-susp {{ background:#fff4e0; }}
  tr.v-clean {{ background:#f3faf4; }}
  .badge {{ padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
  .badge.v-stego {{ background:#c0392b; color:#fff; }}
  .badge.v-susp {{ background:#e67e22; color:#fff; }}
  .badge.v-clean {{ background:#27ae60; color:#fff; }}
  .badge.v-err {{ background:#7f8c8d; color:#fff; }}
  td.err {{ color:#c0392b; white-space:normal; }}
  .note {{ color:var(--muted); font-size:12.5px; }}
  code {{ background:#eee; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <div class="meta">生成时间：{now} ｜ 明细 CSV：<code>{html.escape(os.path.basename(csv_name))}</code></div>

  <div class="cards">
    <div class="card"><div class="k">检测图像</div><div class="v">{n}</div></div>
    <div class="card"><div class="k">干净 clean</div><div class="v" style="color:#27ae60">{n_clean}</div></div>
    <div class="card"><div class="k">可疑 suspicious</div><div class="v" style="color:#e67e22">{n_susp}</div></div>
    <div class="card"><div class="k">高度疑似 likely_stego</div><div class="v" style="color:#c0392b">{n_stego}</div></div>
    <div class="card"><div class="k">解析失败</div><div class="v">{n_err}</div></div>
  </div>

  <h2>热力图</h2>
  {chart_html}

  <h2>逐图明细（按可疑度降序）</h2>
  <table>
    <thead><tr>
      <th>文件</th><th>格式</th><th>尺寸</th><th>可疑度</th><th>判定</th>
      <th>估计嵌入率</th><th>估计嵌入量(bit)</th>
      <th>卡方 p</th><th>滑窗检出</th><th>RS 率</th><th>SPA 率</th>
      <th>JPEG 卡方 p</th><th>h(1)漂移</th><th>错误</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p class="note">* SPA 率带星号表示二次方程根被裁剪到 [0,1]，该估计置信度较低。
  判定阈值：可疑度 ≥ {THRESHOLD_SUSPICIOUS} 为 suspicious，≥ {THRESHOLD_HIGH} 为 likely_stego。
  方法细节与参数见 README.md。</p>
</div>
</body>
</html>"""
