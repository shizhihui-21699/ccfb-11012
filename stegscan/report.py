"""Reporting: CSV + a self-contained HTML report with heatmaps.

The report is meant to be read by an analyst triaging a directory, so it
leads with the ranked list of suspicious files and then shows the evidence
behind each verdict.  Three figures are produced:

1. **Suspiciousness / rate heatmap** -- one row per file, one column per
   detector plus the consensus.  Colour encodes the estimated embedding
   rate, so a file that only one detector flagged stands out as a single
   hot cell, while a genuinely embedded file is hot across the row.
2. **Estimate vs truth** -- only when a labelled manifest is supplied
   (i.e. a test run): a scatter plot of estimated against true rate, with
   the identity line, so calibration error is visible rather than asserted.
3. **Detection curve** -- fraction of files flagged vs true embedding
   rate, again only with labels.

Everything is embedded as base64 so the HTML file can be mailed or opened
offline without breaking.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SPATIAL_METHODS = ("chi2", "spa", "rs")
METHOD_ORDER = ("chi2", "spa", "rs", "dct")
METHOD_LABELS = {
    "chi2": "chi-square",
    "spa": "sample pairs",
    "rs": "RS analysis",
    "dct": "DCT histogram",
}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

_RATE_CMAP = LinearSegmentedColormap.from_list(
    "steg", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"], N=256
)
_SUSP_CMAP = LinearSegmentedColormap.from_list(
    "susp", ["#f0f0f0", "#fee08b", "#fdae61", "#f46d43", "#a50026"], N=256
)


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def figure_rate_heatmap(verdicts, max_rows: int = 60) -> str:
    """Rows = files (most suspicious first), columns = detectors + consensus."""
    files = [v for v in verdicts if not v.error]
    if not files:
        return ""
    files = sorted(files, key=lambda v: -_num(v.suspiciousness))[:max_rows]

    cols = [m for m in METHOD_ORDER if any(m in v.methods for v in files)]
    cols = cols + ["consensus"]
    mat = np.full((len(files), len(cols)), np.nan)
    for i, v in enumerate(files):
        for j, c in enumerate(cols):
            if c == "consensus":
                mat[i, j] = v.rate
            elif c in v.methods:
                mat[i, j] = v.methods[c].rate

    fig_h = max(2.2, 0.22 * len(files) + 1.4)
    fig_w = max(4.6, 1.05 * len(cols) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=_RATE_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_facecolor("#eceff3")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([("consensus" if c == "consensus" else METHOD_LABELS[c])
                        for c in cols], fontsize=9)
    ax.set_yticks(range(len(files)))
    labels = []
    for v in files:
        name = v.path.name
        labels.append(name if len(name) <= 44 else name[:41] + "...")
    ax.set_yticklabels(labels, fontsize=7)

    # annotate each cell: value where measurable, "—" where the detector
    # declined to estimate (grey background), so an absent estimate is never
    # confused with an estimate of zero.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if val > 0.55 else "#20232a")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#8a939e")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("estimated embedding rate (fraction of LSB capacity)", fontsize=8)
    ax.set_title("Per-detector embedding-rate estimates", fontsize=11, loc="left")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(files), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    return _png_b64(fig)


def figure_score_distribution(verdicts) -> str:
    """Histogram of suspiciousness, plus the clean/stego split when labelled.

    A one-column heatmap of scores is unreadable once many files saturate at
    100 (which they do for any sizeable payload), so the score summary is a
    distribution instead: it answers "how many files sit in each band",
    which is what a triage reader actually wants from this figure.
    """
    files = [v for v in verdicts if not v.error]
    if not files:
        return ""
    scores = np.array([_num(v.suspiciousness) for v in files])
    bins = np.arange(0, 101, 10)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.hist(scores, bins=bins, color="#6baed6", edgecolor="white", linewidth=0.8)
    ax.axvline(50, color="#a50026", linestyle="--", linewidth=1.2)
    ax.text(51, ax.get_ylim()[1] * 0.92, "flag threshold", fontsize=8,
            color="#a50026")
    ax.set_xlabel("suspiciousness score", fontsize=9)
    ax.set_ylabel("files", fontsize=9)
    ax.set_title(f"Score distribution (n={len(files)})", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)
    return _png_b64(fig)


def figure_estimate_vs_truth(verdicts, truth: dict[str, float]) -> str:
    """Scatter of estimated vs true rate for a labelled test corpus."""
    pts = []
    for v in verdicts:
        if v.error:
            continue
        key = v.path.name
        if key in truth and np.isfinite(v.rate):
            pts.append((truth[key], v.rate))
    if len(pts) < 4:
        return ""
    t = np.array([p[0] for p in pts])
    e = np.array([p[1] for p in pts])

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="perfect")
    ax.scatter(t, e, s=14, alpha=0.65, color="#2171b5", edgecolors="none")
    ax.set_xlabel("true embedding rate", fontsize=9)
    ax.set_ylabel("estimated embedding rate", fontsize=9)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    mae = float(np.mean(np.abs(e - t)))
    ax.set_title(f"Estimate vs truth (MAE {mae:.3f}, n={len(t)})", fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    return _png_b64(fig)


def figure_detection_curve(verdicts, truth: dict[str, float],
                           threshold: float = 50.0) -> str:
    groups: dict[float, list[bool]] = {}
    for v in verdicts:
        if v.error:
            continue
        key = v.path.name
        if key not in truth:
            continue
        groups.setdefault(round(truth[key], 3), []).append(
            _num(v.suspiciousness) >= threshold)
    if len(groups) < 3:
        return ""
    xs = sorted(groups)
    ys = [100.0 * sum(groups[r]) / len(groups[r]) for r in xs]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.step(xs, ys, where="mid", color="#a50026", linewidth=1.6)
    ax.scatter(xs, ys, s=26, color="#a50026", zorder=3)
    ax.set_xlabel("true embedding rate", fontsize=9)
    ax.set_ylabel(f"% flagged (score >= {threshold:g})", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title("Detection rate by payload size", fontsize=10, loc="left")
    return _png_b64(fig)


def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def write_csv(verdicts, path: Path) -> None:
    cols = [
        "path", "format", "width", "height", "channels", "bit_depth",
        "file_size_bytes", "verdict", "suspiciousness",
        "est_rate", "est_payload_bytes", "rate_spread",
    ]
    for m in METHOD_ORDER:
        cols += [f"{m}_rate", f"{m}_raw", f"{m}_n_obs", f"{m}_p_uniform",
                 f"{m}_channel_spread", f"{m}_note"]
    cols += ["error"]

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for v in verdicts:
            row = [
                str(v.path), v.fmt, v.width, v.height, v.channels, v.bit_depth,
                v.file_size, v.verdict,
                f"{_num(v.suspiciousness):.1f}",
                _fmt(v.rate), _fmt(v.est_payload_bytes), _fmt(v.rate_spread),
            ]
            for m in METHOD_ORDER:
                mv = v.methods.get(m)
                if mv is None:
                    row += ["", "", "", "", "", ""]
                else:
                    row += [_fmt(mv.rate), _fmt(mv.raw_rate), mv.n_obs,
                            _fmt(mv.p_uniform), _fmt(mv.channel_spread), mv.note]
            row.append(v.error)
            w.writerow(row)


def _fmt(x) -> str:
    """Four decimals for a rate; empty string when not measurable."""
    return f"{x:.4f}" if isinstance(x, (int, float)) and np.isfinite(x) else ""


def _fmt_bytes(x) -> str:
    """Payload size, scaled to a readable unit."""
    if not isinstance(x, (int, float)) or not np.isfinite(x):
        return "—"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f} MB"
    if x >= 1_000:
        return f"{x / 1_000:.1f} kB"
    return f"{x:,.0f} B"


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def _rows_html(verdicts) -> str:
    out = []
    for v in sorted(verdicts, key=lambda v: -_num(v.suspiciousness)):
        cls = "row-high" if _num(v.suspiciousness) >= 85 else (
            "row-mid" if _num(v.suspiciousness) >= 50 else (
                "row-weak" if _num(v.suspiciousness) >= 20 else "row-clean"))
        method_cells = []
        for m in METHOD_ORDER:
            mv = v.methods.get(m)
            if mv is None:
                method_cells.append('<td class="num muted">—</td>')
                continue
            note = f' title="{html.escape(mv.note)}"' if mv.note else ""
            method_cells.append(
                f'<td class="num"{note}>{_fmt(mv.rate) or "—"}'
                f'<span class="raw">raw {_fmt(mv.raw_rate) or "—"} · n={mv.n_obs}</span></td>'
            )
        out.append(f"""
      <tr class="{cls}">
        <td class="path" title="{html.escape(str(v.path))}">{html.escape(v.path.name)}</td>
        <td class="fmt">{v.fmt} {v.width}&times;{v.height}</td>
        <td class="num score">{_num(v.suspiciousness):.0f}</td>
        <td class="num">{_fmt(v.rate) or "—"}</td>
        <td class="num">{_fmt_bytes(v.est_payload_bytes)}</td>
        {''.join(method_cells)}
        <td class="verdict">{html.escape(v.verdict)}{('<br><span class="err">' + html.escape(v.error) + '</span>') if v.error else ''}</td>
      </tr>""")
    return "".join(out)


def write_html(verdicts, path: Path, *, title: str, root: str,
               truth: dict[str, float] | None = None,
               figures: dict[str, str] | None = None,
               calibrate: bool = True) -> None:
    figures = figures or {}
    files = [v for v in verdicts if not v.error]
    errors = [v for v in verdicts if v.error]
    flagged = [v for v in files if _num(v.suspiciousness) >= 50]
    strong = [v for v in files if _num(v.suspiciousness) >= 85]
    weak = [v for v in files if 20 <= _num(v.suspiciousness) < 50]

    mae = mae_txt = ""
    if truth:
        diffs = [abs(v.rate - truth[v.path.name]) for v in files
                 if v.path.name in truth and np.isfinite(v.rate)]
        if diffs:
            mae = f"{float(np.mean(diffs)):.3f}"

    def fig_block(key: str, caption: str) -> str:
        b64 = figures.get(key)
        if not b64:
            return ""
        return f"""
    <figure>
      <img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">
      <figcaption>{caption}</figcaption>
    </figure>"""

    bench_html = ""
    if truth:
        bench_html = f"""
  <section>
    <h2>Calibration (labelled test run)</h2>
    <p>This scan had ground truth available for {len(truth)} files, so the
    estimate can be checked against the true payload size. Mean absolute
    error: <strong>{mae}</strong> of capacity.</p>
    <div class="figs">
      {fig_block("truth", "Estimated vs true embedding rate. Points on the dashed line are exact.")}
      {fig_block("detection", "Share of files flagged, grouped by true payload size.")}
    </div>
  </section>"""

    detect_note = "" if calibrate else (
        '<p class="warn">Calibration curves were <strong>disabled</strong> for this run: '
        'the reported rates are raw estimator output.</p>')

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
<style>
  :root {{
    color-scheme: light;
    --bg: #ffffff; --fg: #1b1f24; --muted: #6b7280; --line: #e5e7eb;
    --card: #f8fafc; --accent: #2171b5; --hot: #a50026; --warn: #b45309;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --bg: #0f1419; --fg: #e6e9ee; --muted: #9aa4b2; --line: #263041;
      --card: #161d26; --accent: #74b3ea; --hot: #ff6b6b; --warn: #fbbf24;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --bg: #0f1419; --fg: #e6e9ee; --muted: #9aa4b2; --line: #263041;
    --card: #161d26; --accent: #74b3ea; --hot: #ff6b6b; --warn: #fbbf24;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding-block: 28px;
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }}
  main {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 1.55rem; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.1rem; margin: 34px 0 10px; }}
  .sub {{ color: var(--muted); margin: 0 0 22px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 14px; min-width: 0;
  }}
  .card .k {{ font-size: 1.5rem; font-weight: 650; font-variant-numeric: tabular-nums; }}
  .card .l {{ color: var(--muted); font-size: 0.82rem; }}
  .card.hot .k {{ color: var(--hot); }}
  .warn {{ color: var(--warn); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  thead th {{
    position: sticky; top: 0; background: var(--bg); text-align: left;
    border-bottom: 1px solid var(--line); padding: 7px 7px; white-space: nowrap;
    font-weight: 600; font-size: 12px; color: var(--muted); z-index: 2;
  }}
  tbody td {{ border-bottom: 1px solid var(--line); padding: 6px 7px; vertical-align: top; }}
  td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.fmt {{ white-space: nowrap; color: var(--muted); font-size: 12px; }}
  td.path {{ max-width: 240px; min-width: 150px; overflow: hidden;
             text-overflow: ellipsis; white-space: nowrap; }}
  td .raw {{ display: block; color: var(--muted); font-size: 10px; white-space: nowrap; }}
  td.muted {{ color: var(--muted); }}
  td.score {{ font-weight: 650; }}
  td.verdict {{ font-size: 11px; white-space: nowrap; }}
  .err {{ color: var(--hot); }}
  tr.row-high td.score {{ color: var(--hot); }}
  tr.row-mid td.score {{ color: #d9480f; }}
  tr.row-weak td.score {{ color: var(--muted); }}
  .tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }}
  figure {{ margin: 0 0 18px; }}
  figure img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
  figcaption {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
  .figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .figs > figure img {{ width: 100%; }}
  ul.notes {{ margin: 8px 0 0 18px; padding: 0; }}
  li {{ margin-bottom: 6px; }}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="sub">Directory <code>{html.escape(root)}</code> · {len(verdicts)} files ·
  scan {html.escape(_now())}</p>
  {detect_note}

  <section class="cards">
    <div class="card {('hot' if strong else '')}"><div class="k">{len(strong)}</div><div class="l">strong evidence (score ≥ 85)</div></div>
    <div class="card {('hot' if flagged else '')}"><div class="k">{len(flagged)}</div><div class="l">flagged (score ≥ 50)</div></div>
    <div class="card"><div class="k">{len(weak)}</div><div class="l">weak / inconclusive (20–49)</div></div>
    <div class="card"><div class="k">{len(files) - len(flagged)}</div><div class="l">not flagged</div></div>
    <div class="card"><div class="k">{len(errors)}</div><div class="l">unreadable</div></div>
  </section>

  <section>
    <h2>Rate heatmap</h2>
    {fig_block("rate", "Estimated embedding rate per detector. Hot cells mean a detector believes that fraction of the LSB plane is payload. Compare across a row: agreement is evidence, one hot cell is usually a model artefact.")}
    {fig_block("susp", "Distribution of the overall suspiciousness score. The score is derived from the consensus rate and attenuated when the detectors disagree; the dashed line is the reporting threshold.")}
  </section>
{bench_html}
  <section>
    <h2>Findings</h2>
    <p class="sub">Sorted by suspiciousness. Estimated payload = consensus rate ×
    pixels × channels ÷ 8. “raw” is the estimator output before the
    calibration map.</p>
    <div class="tablewrap">
      <table>
        <thead>
          <tr>
            <th>file</th><th>format</th><th>score</th><th>est. rate</th>
            <th>est. bytes</th>
            <th>chi-square</th><th>sample pairs</th><th>RS</th><th>DCT</th>
            <th>verdict</th>
          </tr>
        </thead>
        <tbody>{_rows_html(verdicts)}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>How to read this</h2>
    <ul class="notes">
      <li><strong>Rate</strong> is payload bits divided by available LSB capacity
        (pixels × analysed channels), so 0.25 means a quarter of every channel's
        LSB plane carries payload.</li>
      <li><strong>Detection floor.</strong> Below roughly 10 % of capacity the
        structural detectors lose power: the payload changes too few pixel pairs
        to outvote the image's own statistics. A short message is therefore
        genuinely hard to prove; “weak” is the honest answer, not a threshold
        to tune away.</li>
      <li><strong>JPEG</strong> is analysed in the quantised DCT coefficients
        (JSteg-style LSB replacement), calibrated by recompressing a 4-pixel
        crop so the 8×8 grid shifts and the cover statistics come back. A
        spatial-domain payload in a JPEG would be destroyed by re-encoding and
        is a different threat model.</li>
      <li><strong>Clean controls matter.</strong> {len(files) - len(flagged)} of
        {len(files)} files here were not flagged; a low false-positive rate on
        unmodified images is what makes the flagged ones worth acting on.</li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------
# entry point
# -------------------------------------------------------------------------------------------------


def build_report(verdicts, out_dir: str | Path, *, root: str = "",
                 title: str = "Steganalysis report",
                 truth: dict[str, float] | None = None,
                 calibrate: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "rate": figure_rate_heatmap(verdicts),
        "susp": figure_score_distribution(verdicts),
    }
    if truth:
        figures["truth"] = figure_estimate_vs_truth(verdicts, truth)
        figures["detection"] = figure_detection_curve(verdicts, truth)

    csv_path = out_dir / "findings.csv"
    write_csv(verdicts, csv_path)

    html_path = out_dir / "report.html"
    write_html(verdicts, html_path, title=title, root=root,
               truth=truth, figures=figures, calibrate=calibrate)

    summary = {
        "files": len(verdicts),
        "flagged": sum(1 for v in verdicts if not v.error and _num(v.suspiciousness) >= 50),
        "errors": sum(1 for v in verdicts if v.error),
        "mae": None,
    }
    if truth:
        d = [abs(v.rate - truth[v.path.name]) for v in verdicts
             if not v.error and v.path.name in truth and np.isfinite(v.rate)]
        if d:
            summary["mae"] = float(np.mean(d))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"html": html_path, "csv": csv_path,
            "summary": out_dir / "summary.json"}
