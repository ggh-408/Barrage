"""Plot recorded collision centers in the 820 x 820 game coordinate system."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source: Path, output: Path, title: str) -> None:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "legend.frameon": False})
    # The plotting area itself is exactly 820 x 820 pixels at the saved DPI.
    fig = plt.figure(figsize=(11.6, 10.2), dpi=100, facecolor="white")
    ax = fig.add_axes([.09, .105, 820/1160, 820/1020])
    ax.set(xlim=(0, 820), ylim=(820, 0), xlabel="X / 像素", ylabel="Y / 像素")
    ax.set_aspect("equal")
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 820])
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 820])
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.set_facecolor("#f7f9fc")
    if rows:
        x = [float(r["plane_center_x"]) for r in rows]
        y = [float(r["plane_center_y"]) for r in rows]
        t = [float(r["survival_seconds"]) for r in rows]
        assert all(0 <= a <= 820 for a in x+y)
        if rows[0].get("record_group"):
            for group, marker, label in (("current", "o", "当前修复版"),
                                          ("historical", "^", "历史版本")):
                indices = [i for i, r in enumerate(rows) if r["record_group"] == group]
                points = ax.scatter([x[i] for i in indices], [y[i] for i in indices],
                                    c=[t[i] for i in indices], cmap="viridis", vmin=0, vmax=120,
                                    marker=marker, s=85, edgecolors="#24324a" if group == "current" else "white",
                                    linewidths=1, zorder=3, label=f"{label}（{len(indices)}）")
            ax.legend(loc="lower right", frameon=False, fontsize=10)
        else:
            points = ax.scatter(x, y, c=t, cmap="viridis", vmin=0, vmax=120,
                                s=80, edgecolors="white", linewidths=1, zorder=3)
        occupied = []
        for i, (px, py, seconds) in enumerate(zip(x, y, t), 1):
            label = f"{rows[i-1].get('point_label', i)} · {seconds:.2f}s"
            width, height = len(label)*5.6, 16
            candidates = []
            for radius in (18, 30, 45, 60, 80, 105, 135, 165):
                for angle in range(0, 360, 30):
                    lx = px + radius*math.cos(math.radians(angle))
                    ly = py + radius*math.sin(math.radians(angle))
                    box = (lx-width/2, ly-height/2, lx+width/2, ly+height/2)
                    if box[0] < 3 or box[1] < 3 or box[2] > 817 or box[3] > 817:
                        continue
                    overlaps = sum(box[0] < b[2] and box[2] > b[0] and
                                   box[1] < b[3] and box[3] > b[1] for b in occupied)
                    covered = sum(box[0]-5 < a < box[2]+5 and box[1]-5 < b < box[3]+5
                                  for a, b in zip(x, y))
                    candidates.append((overlaps*10000 + covered*1000 + radius, lx, ly, box))
            _, lx, ly, box = min(candidates)
            occupied.append(box)
            ax.annotate(label, (px, py), xytext=(lx, ly), ha="center", va="center",
                        fontsize=8, color="#17243b",
                        arrowprops=dict(arrowstyle="-", color="#8190a4", lw=.6), zorder=4)
        cax = fig.add_axes([.84, .36, .022, .42])
        fig.colorbar(points, cax=cax, label="死亡时间 / 秒")
    else:
        ax.text(410, 410, "所选种子在本次回放中无死亡", ha="center")
    fig.suptitle(title, fontsize=15, y=.965)
    fig.text(.09, .04, f"死亡点：{len(rows)}  |  坐标为飞机中心  |  原点位于左上角，Y 轴向下", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="死亡位置与死亡时间")
    args = parser.parse_args()
    plot(args.source, args.output, args.title)
