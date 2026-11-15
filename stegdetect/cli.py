"""命令行入口：批量检测目录并生成报告。

用法
----
python -m stegdetect.cli detect <图片目录> [--out 报告目录] [--title 标题]
python -m stegdetect.cli detect <单张图片>
"""
from __future__ import annotations

import argparse
import os
import sys

from .detector import detect_file, detect_directory
from .report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stegdetect.cli",
        description="PNG/BMP LSB 与 JPEG JSteg 隐写检测（卡方/RS/SPA/DCT 直方图）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="检测单张图片或整个目录")
    d.add_argument("target", help="图片文件或目录路径")
    d.add_argument("--out", default="reports/latest", help="报告输出目录")
    d.add_argument("--title", default="LSB 隐写检测报告")
    d.add_argument("--no-report", action="store_true", help="只打印终端摘要，不写 HTML")
    args = parser.parse_args(argv)

    if os.path.isfile(args.target):
        results = [detect_file(args.target)]
    else:
        print(f"[*] 扫描目录: {args.target}")
        results = detect_directory(args.target)

    if not results:
        print("[!] 未发现受支持的图片（.png/.bmp/.jpg/.jpeg）")
        return 1

    # 终端摘要
    print(f"[*] 共 {len(results)} 张图像")
    header = f"{'可疑度':>6} {'判定':>13} {'估计率':>7} {'嵌入量(bit)':>12}  文件"
    print(header)
    print("-" * len(header) + "-" * 30)
    for r in sorted(results, key=lambda x: -x.suspicion):
        name = os.path.basename(r.file)
        if r.verdict == "error":
            print(f"{'ERR':>6} {'error':>13} {'-':>7} {'-':>12}  {name}  [{r.error}]")
            continue
        print(f"{r.suspicion:6.3f} {r.verdict:>13} {r.estimated_rate * 100:6.1f}% "
              f"{r.estimated_payload_bits:12d}  {name}")

    if not args.no_report:
        html_path = build_report(results, args.out, title=args.title)
        print(f"\n[+] 报告已生成: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
