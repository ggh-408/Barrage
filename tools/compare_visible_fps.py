"""Compare two matched 100-FPS visible tests using the existing one-second bins.

Writes one before/after overlay PNG, a concise comparison JSON and the plotted
one-second samples CSV. An optional stress report is summarized only in JSON.
All artifacts are published exclusively; existing files are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.report_helpers import _atomic_publish, _write_json


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    frames_path = path.with_suffix(".frames.csv")
    with frames_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["elapsed_seconds"]) for row in rows], dtype=np.float64)
    if not len(times) or len(times) != report["rendered_frames"]:
        raise ValueError(f"Frame count mismatch in {frames_path}")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0) or times[0] < 0.0:
        raise ValueError(f"Frame completion times must be finite and increasing: {frames_path}")
    complete_seconds = int(np.floor(times[-1]))
    if complete_seconds < 1:
        raise ValueError(f"At least one complete second is required: {frames_path}")
    # Identical binning to plot_visible_fps.py: frame completions in [n,n+1),
    # discarding the incomplete trailing second from the per-second curve.
    edges = np.arange(complete_seconds + 1, dtype=float)
    fps, _ = np.histogram(times[times < complete_seconds], bins=edges)
    return {
        "path": path.resolve(), "report": report, "edges": edges,
        "fps": fps, "complete_seconds": complete_seconds,
    }


def _check_comparable(before: dict[str, Any], after: dict[str, Any]) -> None:
    for name, data in (("before", before), ("after", after)):
        if data["report"].get("render_fps_cap") != 100:
            raise ValueError(f"{name} report must explicitly record render_fps_cap=100")
    for field in (
        "seed", "checkpoint_sha256", "bullets", "targeted_bullet_probability",
        "damage_immunity", "physics_fps", "ai_enabled",
    ):
        if field not in before["report"] or field not in after["report"]:
            raise ValueError(f"Both reports must contain comparison setting {field}")
        if before["report"][field] != after["report"][field]:
            raise ValueError(f"Reports use different values for {field}")


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    report = data["report"]
    decision = report["ai_decision"]
    interval = report["frame_interval"]
    return {
        "report": str(data["path"]),
        "duration_seconds": float(report["duration_seconds"]),
        "rendered_frames": int(report["rendered_frames"]),
        "average_fps": float(report["rendered_fps"]),
        "one_second_fps_min": int(data["fps"].min()),
        "one_second_fps_max": int(data["fps"].max()),
        "complete_one_second_windows": data["complete_seconds"],
        "decision_count": int(decision["count"]),
        "decision_mean_ms": float(decision["mean_ms"]),
        "decision_p99_ms": float(decision["p99_ms"]),
        "decision_max_ms": float(decision["max_ms"]),
        "decision_over_budget_count": int(report["decision_over_budget_count"]),
        "decision_budget_ms": float(report["decision_budget_ms"]),
        "frame_interval_mean_ms": float(interval["mean_ms"]),
        "frame_interval_p99_ms": float(interval["p99_ms"]),
        "frame_interval_max_ms": float(interval["max_ms"]),
        "physics_catchup_frames": int(report["physics_catchup_frames"]),
        "physics_max_steps_per_frame": int(report["physics_max_steps_per_frame"]),
        "alive_physics_steps": report.get("alive_physics_steps"),
        "survival_game_seconds": report.get("survival_game_seconds"),
        "final_score": report.get("final_score"),
        "alive_at_end": report.get("alive_at_end"),
        "torch_threads": report.get("torch_threads"),
        "runtime_stages": report.get("runtime_stages", {}),
    }


def _publish_csv(path: Path, before: dict[str, Any], after: dict[str, Any], seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.writer(stream)
            writer.writerow(("start_seconds", "end_seconds", "before_fps", "after_fps"))
            for second in range(seconds):
                writer.writerow((second, second + 1, int(before["fps"][second]), int(after["fps"][second])))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    _atomic_publish(temporary, path)


def _draw_plot(path: Path, before: dict[str, Any], after: dict[str, Any], seconds: int) -> None:
    first = before["report"]
    second = after["report"]
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
    fig.patch.set_facecolor("#f6f8fc")
    ax.set_facecolor("white")
    x = np.arange(seconds, dtype=float) + 0.5
    ax.plot(x, before["fps"][:seconds], color="#64748b", linewidth=1.8,
            label=f"优化前（平均 {first['rendered_fps']:.2f} FPS）")
    ax.plot(x, after["fps"][:seconds], color="#2563eb", linewidth=1.8,
            label=f"优化后（平均 {second['rendered_fps']:.2f} FPS）")
    ax.axhline(first["rendered_fps"], color="#64748b", linewidth=1.4, linestyle=":")
    ax.axhline(second["rendered_fps"], color="#0f766e", linewidth=1.4, linestyle=":")
    ax.axhline(100, color="#94a3b8", linewidth=1.2, linestyle="--", label="渲染上限 100 FPS")
    maximum = max(float(before["fps"][:seconds].max()), float(after["fps"][:seconds].max()))
    ax.set(xlim=(0, seconds), ylim=(0, max(130, maximum * 1.12)),
           xlabel="现实经过时间（秒）", ylabel="帧率（FPS）")
    ax.set_xticks(np.arange(0, seconds + 1, 15))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Barrage 窗口实测：100 FPS 上限优化前后对比",
                 x=0.08, ha="left", fontsize=18, fontweight="bold")
    before_threads = first.get("torch_threads")
    after_threads = second.get("torch_threads")
    thread_label = (f"CPU / {before_threads} 线程" if before_threads == after_threads
                    else f"CPU / 线程 {before_threads} → {after_threads}")
    checkpoint_label = Path(second["checkpoint"]).parent.name.replace("visual_set_", "")
    ax.set_title(f"{checkpoint_label} 最优权重 · {thread_label} · {second['bullets']} 发子弹 · 10% 瞄准弹 · seed {second['seed']}",
                 loc="left", fontsize=10, color="#475569", pad=15)
    first_ai, second_ai = first["ai_decision"], second["ai_decision"]
    fig.text(0.08, 0.055,
             f"决策平均延迟 {first_ai['mean_ms']:.2f} → {second_ai['mean_ms']:.2f} ms"
             f"     P99 {first_ai['p99_ms']:.2f} → {second_ai['p99_ms']:.2f} ms"
             f"     最大 {first_ai['max_ms']:.2f} → {second_ai['max_ms']:.2f} ms",
             fontsize=10, color="#334155")
    fig.text(0.08, 0.017,
             "口径：pygame.display.flip() 返回后的帧完成次数；每 1 秒统计，排除不足 1 秒区间。未测量显示器实际出光延迟。",
             fontsize=9, color="#64748b")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.19)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(temporary, format="png")
        _atomic_publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--stress", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Overlay PNG; JSON and CSV use the same stem")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix.lower() != ".png":
        parser.error("output must have a .png extension")
    comparison_path = output.with_suffix(".json")
    buckets_path = output.with_suffix(".csv")
    for path in (output, comparison_path, buckets_path):
        if path.exists():
            parser.error(f"Refusing to overwrite existing artifact: {path}")
    before = _load_report(args.before)
    after = _load_report(args.after)
    _check_comparable(before, after)
    stress = _load_report(args.stress) if args.stress is not None else None
    if stress is not None:
        _check_comparable(after, stress)
    seconds = min(before["complete_seconds"], after["complete_seconds"])
    first, second = _summary(before), _summary(after)
    comparison: dict[str, Any] = {
        "kind": "visible_100fps_before_after_comparison",
        "plot": str(output), "one_second_samples": str(buckets_path),
        "plot_complete_seconds": seconds,
        "settings": {key: before["report"][key] for key in (
            "render_fps_cap", "seed", "checkpoint_sha256", "bullets",
            "targeted_bullet_probability", "damage_immunity", "physics_fps", "ai_enabled",
        )},
        "before": first, "after": second,
        "change": {
            "average_fps_delta": second["average_fps"] - first["average_fps"],
            "average_fps_gain_percent": (second["average_fps"] / first["average_fps"] - 1.0) * 100.0,
            "decision_mean_ms_delta": second["decision_mean_ms"] - first["decision_mean_ms"],
            "decision_mean_reduction_percent": (
                (1.0 - second["decision_mean_ms"] / first["decision_mean_ms"]) * 100.0
                if first["decision_mean_ms"] > 0.0 else None
            ),
            "decision_p99_ms_delta": second["decision_p99_ms"] - first["decision_p99_ms"],
            "frame_interval_p99_ms_delta": second["frame_interval_p99_ms"] - first["frame_interval_p99_ms"],
            "frame_interval_max_ms_delta": second["frame_interval_max_ms"] - first["frame_interval_max_ms"],
        },
        "notes": [
            "The overlay uses the shared complete duration; each report summary uses its full measured duration.",
            "Average FPS is the existing report value: all completed frames divided by wall duration.",
            "One-second bins exactly match tools/plot_visible_fps.py and exclude the final incomplete second.",
            "Timing diagnostics with damage immunity do not measure survival acceptance.",
            "FPS records display.flip completion rather than monitor photon output.",
        ],
    }
    if stress is not None:
        comparison["stress"] = _summary(stress)
    _draw_plot(output, before, after, seconds)
    _publish_csv(buckets_path, before, after, seconds)
    _write_json(comparison_path, comparison)
    print(json.dumps({"plot": str(output), "comparison": str(comparison_path),
                      "one_second_samples": str(buckets_path), "change": comparison["change"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
