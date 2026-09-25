"""Merge verified historical results and the missing current-controller replays."""
import csv
import json
from pathlib import Path
from plot_death_positions import plot
from audit_large_test_deaths import read_csv, save_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/large_test_deaths_20260919"


def main():
    summary = json.loads((OUT / "inventory_summary.json").read_text())
    old = json.loads((OUT / "historical_seed_sources.json").read_text())
    results = {int(r["seed"]): r for r in read_csv(OUT / "verified_current_results.csv")}
    fresh = read_csv(OUT / "replay/episodes.csv")
    assert {int(r["seed"]) for r in fresh} == set(summary["replay_seeds"])
    for r in fresh:
        seed = int(r["seed"])
        assert seed not in results
        results[seed] = dict(seed=seed, seconds=float(r["survival_seconds"]),
                             reason=r["termination_reason"], source="diagnostics/large_test_deaths_20260919/replay/episodes.csv")
    assert set(results) == {int(s) for s in old}
    deaths = []
    for filename, source in ((ROOT / "diagnostics/current_verified_deaths_20260919/collision_positions.csv", "historical"),
                             (OUT / "replay/collision_positions.csv", "new replay")):
        for r in read_csv(filename):
            seed = int(r["seed"])
            assert results[seed]["reason"] == "collision"
            assert abs(float(results[seed]["seconds"])-float(r["survival_seconds"])) < 1e-4
            deaths.append(dict(seed=seed, survival_seconds=float(r["survival_seconds"]),
                               plane_center_x=float(r["plane_center_x"]), plane_center_y=float(r["plane_center_y"]),
                               position_source=r.get("position_source", str(filename.relative_to(ROOT))),
                               record_type=source))
    assert len(deaths) == len({r["seed"] for r in deaths})
    assert len(deaths) == sum(r["reason"] == "collision" for r in results.values())
    deaths.sort(key=lambda r:r["survival_seconds"])
    for i, r in enumerate(deaths, 1):
        r["point_id"] = i
    save_csv(OUT / "collision_positions.csv", deaths)
    reconciliation = []
    for seed, result in sorted(results.items()):
        reconciliation.append(dict(seed=seed, historical_death_sources=" | ".join(x["source"] for x in old[str(seed)]),
                                   historical_death_times=" | ".join(str(x["seconds"]) for x in old[str(seed)]),
                                   current_controller_reason=result["reason"],
                                   current_controller_seconds=result["seconds"], current_result_source=result["source"]))
    save_csv(OUT / "seed_reconciliation.csv", reconciliation)
    plot(OUT / "collision_positions.csv", OUT / "death_map.png",
         "当前保留修复版 · 300 发 · 全部历史死亡种子核对")
    summary.update(current_verified_deaths=len(deaths), replayed_seeds=len(fresh),
                   new_replay_deaths=sum(r["termination_reason"] == "collision" for r in fresh),
                   scope="Diagnostic reconciliation of historical failures, not a new full 7000-episode evaluation")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = "\n".join(f"| {r['point_id']} | {r['seed']} | {r['survival_seconds']:.3f} | {r['plane_center_x']:.3f} | {r['plane_center_y']:.3f} |" for r in deaths)
    text = f"""# 数千局历史测试死亡种子完整核对

上一轮“三个死亡种子”的表述缺少完整批测核对依据。本次重新盘点同一部署 checkpoint、300 发的 23 份有逐局记录的测试，三次大批测共覆盖 7000 个不同种子。

| 历史批测 | 当时保护器 | 当时死亡数 |
|---|---|---|
| 2026-09-06，3000 局 | refined | 19 |
| 2026-09-07，2000 局 | refined | 7 |
| 2026-09-08，2000 局 | receding，完整前瞻基础修复前 | 13 |

三批合计 39 个不同死亡种子。这 39 个种子在后续当前保留修复版 `forward22` / `regression67` 回归中均达到 120 秒，逐项对应见 `seed_reconciliation.csv`。历史死亡数与当前修复版死亡数需分别解释。

加入其他同一 checkpoint、300 发小批测后，共有 46 个历史死亡种子。43 个有已核实的当前保留版结果；缺少明确当前版结果的 655066913、890779774、2100645359 本次已按原项目并行 rollout 补跑。新回放仍使用 threshold_0.18.pt、300 发、逐弹 0.10 瞄准概率、RGB 输入、120 秒截止以及当前 receding 控制器。未修改策略或评价语义；真实坐标只用于碰撞后的诊断输出。

当前版本合计核实 {len(deaths)} 个死亡点，其中新补跑死亡 {summary['new_replay_deaths']} 个，其余直接复用已核实的当前版本记录。全体 46 个种子均有当前版本结果，死亡记录均有坐标。

| 图中编号 | 种子 | 死亡秒数 | X / px | Y / px |
|---|---|---|---|---|
{table}

图仅显示当前保留修复版下的死亡点。当前控制器与保存的 forward_controller.py 直接文本比较一致；完整控制器取舍依据见 `../continuation_repair_20260909/report.md`，前次三点来源核对见 `../current_verified_deaths_20260919/report.md`。

本次只补齐历史死亡种子的当前结果，未用当前版本重新执行全部 7000 个种子，不能据此宣称当前模型在完整 7000 局中仅死亡 {len(deaths)} 次；旧版本成功种子可能在当前版本产生回退，需完整重测才能确认。
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"historical_large_test_deaths":39, "all_historical_death_seeds":46,
                      "new_replay_deaths":summary["new_replay_deaths"], "current_deaths":deaths}), flush=True)


if __name__ == "__main__":
    main()
