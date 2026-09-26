"""Plot measured frame completions in full one-second wall-clock windows."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    frames_path = args.report.with_suffix(".frames.csv")
    with frames_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["elapsed_seconds"]) for row in rows])
    assert len(times) == report["rendered_frames"]
    assert np.isfinite(times).all() and np.all(np.diff(times) > 0)
    complete_seconds = int(np.floor(times[-1]))
    edges = np.arange(complete_seconds + 1, dtype=float)
    fps, _ = np.histogram(times[times < complete_seconds], bins=edges)
    output = args.report.with_suffix(".fps.png")
    buckets_path = args.report.with_suffix(".fps.csv")
    if output.exists() or buckets_path.exists():
        raise FileExistsError("Plot output already exists")
    with buckets_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("start_seconds", "end_seconds", "frames_per_second"))
        writer.writerows(zip(edges[:-1], edges[1:], fps.tolist()))
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
    fig.patch.set_facecolor("#f6f8fc")
    ax.set_facecolor("white")
    x = edges[:-1] + 0.5
    ax.plot(x, fps, color="#2563eb", linewidth=1.8, label="实测 FPS（每 1 秒统计）")
    ax.axhline(report["rendered_fps"], color="#0f766e", linewidth=1.4,
               linestyle=":", label=f"全程平均 {report['rendered_fps']:.2f} FPS")
    render_cap = report.get("render_fps_cap", 120)
    ax.axhline(render_cap, color="#94a3b8", linewidth=1.2, linestyle="--", label=f"渲染上限 {render_cap} FPS")
    ax.set(xlim=(0, complete_seconds), ylim=(0, max(130, float(fps.max()) * 1.12)),
           xlabel="现实经过时间（秒）", ylabel="帧率（FPS）")
    ax.set_xticks(np.arange(0, complete_seconds + 1, 15))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.legend(loc="lower right" if not report.get("ai_enabled", True) else "upper right",
              frameon=False, fontsize=9)
    fig.suptitle("Barrage 窗口实测：帧率随时间变化", x=0.08, ha="left", fontsize=18, fontweight="bold")
    controller_label = (f"v42 最优权重 · {report.get('ai_device', 'cpu').upper()} / {report['torch_threads']} 线程"
                        if report.get("ai_enabled", True) else "无模型 · 免疫伤害 · 飞机静止")
    ax.set_title(f"{controller_label} · {report['bullets']} 发子弹 · 10% 瞄准弹 · seed {report['seed']}",
                 loc="left", fontsize=10, color="#475569", pad=15)
    ai = report["ai_decision"]
    latency_label = (f"决策延迟：平均 {ai['mean_ms']:.2f} ms / P99 {ai['p99_ms']:.2f} ms / 最大 {ai['max_ms']:.2f} ms"
                     f"     超过 {1000 / report['decision_hz_target']:.2f} ms：{report['decision_over_budget_count']}/{ai['count']}"
                     if ai["count"] else "决策延迟、决策超时：不适用（未加载模型，未执行 AI 决策）")
    fig.text(0.08, 0.055, latency_label,
             fontsize=10, color="#334155")
    fig.text(0.08, 0.017, "口径：pygame.display.flip() 返回后的帧完成次数；排除末尾不足 1 秒区间。未测量显示器实际出光延迟。",
             fontsize=9, color="#64748b")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.19)
    fig.savefig(output)
    plt.close(fig)
    print(json.dumps({"plot": str(output.resolve()), "one_second_fps_min": int(fps.min()),
                      "one_second_fps_max": int(fps.max()), "windows": len(fps)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
