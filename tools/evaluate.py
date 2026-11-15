"""全量检测结果 vs 真值的混淆矩阵与指标。"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stegdetect.detector import detect_directory

root = "data/test_images"
manifest = {m["file"]: m for m in json.load(open(os.path.join(root, "manifest.json")))}
results = detect_directory(root)

# 按格式分组评估
for fmt in ("png", "bmp", "jpeg"):
    sub = [r for r in results if r.format == fmt and r.verdict != "error"]
    if not sub:
        continue
    print(f"\n===== {fmt.upper()} (n={len(sub)}) =====")
    print(f"{'true':>5} {'est_rate':>9} {'susp':>6} {'verdict':>13}  file")
    mae = []
    fp = fn = tp = tn = 0
    for r in sorted(sub, key=lambda x: (manifest[os.path.relpath(x.file, root)]['true_rate'], x.file)):
        key = os.path.relpath(r.file, root)
        true = manifest[key]["true_rate"]
        is_stego = true > 0
        flagged = r.verdict in ("suspicious", "likely_stego")
        if is_stego and flagged: tp += 1
        elif is_stego and not flagged: fn += 1
        elif not is_stego and flagged: fp += 1
        else: tn += 1
        mae.append(abs(r.estimated_rate - true))
    # 打印每档聚合
    for t in sorted(set(manifest[os.path.relpath(r.file, root)]['true_rate'] for r in sub)):
        sel = [r for r in sub if abs(manifest[os.path.relpath(r.file, root)]['true_rate'] - t) < 1e-9]
        susp = np.mean([r.suspicion for r in sel])
        rate = np.mean([r.estimated_rate for r in sel])
        v = [r.verdict for r in sel]
        print(f"{t:5.2f} {rate:9.3f} {susp:6.3f} "
              f"stego={v.count('likely_stego')+v.count('suspicious')}/{len(sel)}")
    print(f"MAE(嵌入率)={np.mean(mae):.3f}  检出 TP={tp} 漏报 FN={fn} 误报 FP={fp} 正确干净 TN={tn}")
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"precision={prec:.3f} recall={rec:.3f}")
