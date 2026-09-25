"""Audit saved deaths against the retained production controller and 300-bullet task."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from plot_death_positions import plot

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "diagnostics/continuation_repair_20260909"
OUTPUT = ROOT / "diagnostics/current_verified_deaths_20260919"
CHECKPOINT = ROOT / "diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    current = ROOT / "tools/pixel_guard_continuation.py"
    reference = REPAIR / "forward_controller.py"
    assert current.read_text(encoding="utf-8") == reference.read_text(encoding="utf-8")
    assert CHECKPOINT.is_file()
    decision = read(REPAIR / "acceptance_decision.json")
    assert decision["retained_current_controller"] is True
    assert decision["status"] == "not_accepted"
    assert not OUTPUT.exists(), f"Output already exists: {OUTPUT}"

    records = {}
    current_results = {}
    for test in ("forward22", "regression67", "fixed200"):
        folder = REPAIR / test
        config = read(folder / "evaluation_config.json")
        assert Path(config["checkpoint"]).resolve() == CHECKPOINT.resolve()
        assert config["bullet_count"] == 300
        assert config["targeted_bullet_probability"] == .1
        assert config["rendered_rgb"] is True
        assert config["episode_limit_seconds"] == 120
        with (folder / "evaluation_episodes.csv").open(newline="", encoding="utf-8-sig") as stream:
            for r in csv.DictReader(stream):
                seed = int(r["seed"])
                current_results[seed] = dict(time=float(r["model_survival_seconds"]),
                    reason=r["termination_reason"], source=str((folder / "evaluation_episodes.csv").relative_to(ROOT)))
                if r["termination_reason"] == "collision":
                    records[seed] = dict(seed=seed, survival_seconds=float(r["model_survival_seconds"]),
                                         evidence_source=current_results[seed]["source"])

    sources = [ROOT / "diagnostics/continuation_holdout_audit_20260909/reproduction.json",
               ROOT / "diagnostics/continuation_certificate_failure_20260909/forward_two.json"]
    expected = {"search_depth": 18, "beam_width": 8, "trigger_safe_actions": 2,
                "wall_reserve_pixels": 48., "wall_cost_weight": 1., "velocity_error_floor": 2.,
                "compiled_beam": True, "search_workers": 9, "preserve_plans": True, "interval_consensus": True}
    for path in sources:
        data = read(path)
        assert data["bullet_count"] == 300 and data["targeted_bullet_probability"] == .1
        assert data["safety_threshold"] == .18 and data["evaluation_episode_limit_seconds"] == 120
        assert data["config"] == expected
        assert data["guard"]["variant"] == "receding_continuation"
        assert data["guard"]["safety_certificate"] == "full independent per-bullet pixel-offset rectangles over the complete horizon"
        for failure in data["failures"]:
            seed, seconds = int(failure["seed"]), float(failure["survival_seconds"])
            row = records.setdefault(seed, dict(seed=seed, survival_seconds=seconds,
                                                evidence_source=str(path.relative_to(ROOT))))
            assert abs(row["survival_seconds"] - seconds) < 1e-5
            row.update(survival_seconds=seconds, plane_center_x=float(failure["plane_center"][0]),
                       plane_center_y=float(failure["plane_center"][1]),
                       position_source=str(path.relative_to(ROOT)), bullet_count=300, safety_threshold=.18)

    missing = [seed for seed, r in records.items() if "plane_center_x" not in r]
    assert not missing, f"Replays required for missing coordinates: {missing}"
    rows = sorted(records.values(), key=lambda r: r["survival_seconds"])
    for i, row in enumerate(rows, 1):
        row["point_id"] = i
    exclusions = []
    old = read(ROOT / "diagnostics/historical_deaths_20260919/seed_sources.json")["seeds"]
    for item in old:
        seed = item["seed"]
        if seed in records:
            continue
        if seed in current_results and current_results[seed]["reason"] == "time_limit":
            reason = "Reached limit in retained repaired production controller; historical death not retained"
            evidence = current_results[seed]["source"]
        elif seed in (655066913, 890779774):
            reason = "Failure belongs to rejected experimental viability controller"
            evidence = "diagnostics/continuation_repair_20260909/acceptance_decision.json"
        else:
            reason = "No verified death record for the retained repaired controller at 300 bullets"
            evidence = " | ".join(item["sources"])
        exclusions.append(dict(seed=seed, reason=reason, evidence=evidence))

    OUTPUT.mkdir()
    for name, data in (("collision_positions.csv", rows), ("excluded_seeds.csv", exclusions)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    plot(OUTPUT / "collision_positions.csv", OUTPUT / "death_map.png",
         "当前保留修复版 · 300 发 · 已核实死亡位置与时间")
    summary = dict(checkpoint=str(CHECKPOINT), safety_threshold=.18, bullet_count=300,
                   targeted_bullet_probability=.1, evaluation_episode_limit_seconds=120,
                   controller=str(current), matching_saved_controller=str(reference),
                   controller_content_equal=True, historical_records_used=True,
                   recorded_deaths=len(rows), missing_positions=len(missing), replays_performed=0,
                   new_formal_evaluation=False,
                   limitation="Historical results with a documented retained-controller lineage; no new closed-loop evaluation")
    (OUTPUT / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = "\n".join(f"| {r['point_id']} | {r['seed']} | {r['survival_seconds']:.3f} | {r['plane_center_x']:.3f} | {r['plane_center_y']:.3f} |" for r in rows)
    report = f"""# 当前保留修复版死亡记录核对

本报告替代此前混合历史配置的死亡地图。只纳入已核实同一部署权重、300 发及当前保留控制器的历史死亡记录。

权重：`diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt`；安全阈值 0.18；300 发；逐弹瞄准概率 0.10；120 秒截止；输入采用 RGB 图像路径。

当前 `tools/pixel_guard_continuation.py` 与 `diagnostics/continuation_repair_20260909/forward_controller.py` 直接文本比较一致。`acceptance_decision.json` 和 `report.md` 记录主路径保留此基础修复版，历史依据保留与路线优先候选未部署。9 月 13 日的 `hotspot_deployed_20260913/report.md` 记录后续检测及追踪优化在采样输入上保持输出一致；这项证据限于已有测试。

`forward22`、`regression67`、`fixed200` 的 checkpoint 路径、300 发、0.10 概率、RGB 输入和截止时间已逐项检查。固定 200 局中的死亡种子 1577244970，位置来自同版 `continuation_holdout_audit_20260909/reproduction.json`。另外两个种子来自 `continuation_certificate_failure_20260909/forward_two.json`，主报告明确记录其在保留基础版本上重测且死亡时间一致。两份位置文件的 guard 变体、完整前瞻检查说明和全部控制参数均已核对。

| 图中编号 | 种子 | 死亡时间 / s | X / px | Y / px |
|---|---|---|---|---|
{table}

三条合格死亡记录均有位置，无需补跑。未开展新的正式评估，也未把其他候选的死亡点用于本图。

旧图中 2128943899、2144082357、2109317496、2036837908 等早期控制器死亡点，在保留版 `forward22` 中达到截止时间，已排除。655066913、890779774 属于未采用的 viability 候选失败，已排除。其余旧阈值、旧保护器、弹数或修复版本无法验证的记录均不纳入；逐种子处置见 `excluded_seeds.csv`。缺少旧配置坐标本身不构成本次补跑理由；补跑条件为已确认属于当前保留修复版的死亡记录缺少坐标。

地图坐标区为 820×820 像素，原点位于左上角，Y 向下；点为飞机中心，标签为编号和死亡时间。明细含原始数据来源。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"death_count": len(rows), "replays": 0, "rows": rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
