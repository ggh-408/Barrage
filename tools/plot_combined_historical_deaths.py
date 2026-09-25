"""Plot the 39 recorded large-test deaths alongside four current-controller deaths."""
import csv
import json
from pathlib import Path
from plot_death_positions import plot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/large_test_deaths_20260919"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    pairs = [
        ("pixel_guard_refined_3000_20260906_140829_564217", "refined_300_failure_tuning_20260907/baseline.json"),
        ("pixel_guard_refined_2000_20260907_125214_881749", "current_tail_audit_20260908/reproduction.json"),
        ("pixel_guard_refined_2000_20260908_120152_858438", "teacher_direction_audit_20260908/reproduction.json"),
    ]
    combined = []
    for i, r in enumerate(rows(OUT / "collision_positions.csv"), 1):
        combined.append(dict(point_label=f"C{i}", record_group="current", seed=int(r["seed"]),
            survival_seconds=float(r["survival_seconds"]), plane_center_x=float(r["plane_center_x"]),
            plane_center_y=float(r["plane_center_y"]), source=r["position_source"], original_test="current repaired controller"))
    for test, capture in pairs:
        source = ROOT / "diagnostics" / test / "evaluation/evaluation_episodes.csv"
        failures = [r for r in rows(source) if r["termination_reason"] == "collision"]
        capture_path = ROOT / "diagnostics" / capture
        saved = json.loads(capture_path.read_text())
        by_seed = {int(r["seed"]):r for r in saved["failures"]}
        for r in failures:
            seed = int(r["seed"])
            original_time = float(r["model_survival_seconds"])
            point = by_seed[seed]
            assert abs(float(point["survival_seconds"])-original_time) < 1e-4, (seed, original_time, point["survival_seconds"])
            assert float(point["safety_threshold"]) == .18
            combined.append(dict(point_label=f"H{len(combined)-3}", record_group="historical", seed=seed,
                survival_seconds=float(point["survival_seconds"]), plane_center_x=float(point["plane_center"][0]),
                plane_center_y=float(point["plane_center"][1]), source=str(capture_path.relative_to(ROOT)),
                original_test=str(source.relative_to(ROOT))))
    assert len(combined) == 43 and len({r["seed"] for r in combined}) == 43
    output = OUT / "combined_43_deaths.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined[0]))
        w.writeheader()
        w.writerows(combined)
    plot(output, OUT / "combined_43_deaths.png", "300 发死亡位置与时间 · 历史 39 点 + 当前修复版 4 点")
    (OUT / "combined_43_notes.md").write_text("# 合并死亡地图\n\nH1–H39 为三次大批测当时的历史死亡点；C1–C4 为当前保留修复版已核实的死亡点。全部使用同一权重、300 发设置，控制逻辑版本有差异。历史 39 个种子在后续修复回归中达到截止时间，历史位置不代表当前模型仍在此死亡。\n\n39 个历史坐标均已从原有失败复现记录取得，每个种子的死亡时间与原批测 CSV 比较误差小于 0.0001 秒。此前遗漏的 13 个位置保存在 teacher_direction_audit_20260908/reproduction.json，该目录包含图像策略失败复现与离线教师诊断；本图仅使用图像策略失败复现。无需重跑。\n\n坐标为飞机中心，原点左上，Y 向下。颜色及标签表示死亡秒数，三角形为历史版本，圆形为当前修复版；图例无边框。编号、种子、时间、坐标及来源见 combined_43_deaths.csv。\n", encoding="utf-8")
    print("Verified and plotted: 39 historical + 4 current = 43 death positions")


if __name__ == "__main__":
    main()
