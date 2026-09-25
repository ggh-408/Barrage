"""Reconcile all 300-bullet tests of the deployed checkpoint, including large tests."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/large_test_deaths_20260919"
CHECKPOINT = ROOT / "diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt"
REPAIR = ROOT / "diagnostics/continuation_repair_20260909"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    tests, historical = [], {}
    large_seeds, large_deaths = set(), set()
    for folder in (ROOT / "diagnostics", ROOT / "runs"):
        for path in sorted(folder.rglob("evaluation_config.json")):
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
            if cfg.get("bullet_count") != 300 or Path(cfg.get("checkpoint", "")).resolve() != CHECKPOINT.resolve():
                continue
            source = path.parent / "evaluation_episodes.csv"
            if not source.exists():
                source = path.parent / "evaluation_episodes.partial.csv"
            if not source.exists():
                continue
            rows = read_csv(source)
            failed = [r for r in rows if r["termination_reason"] == "collision"]
            manifest = source.parent.parent / "experiment_manifest.json"
            variant = "see recorded controller lineage"
            if manifest.exists():
                variant = json.loads(manifest.read_text()).get("variant", variant)
            tests.append(dict(source=str(source.relative_to(ROOT)), episodes=len(rows), deaths=len(failed),
                              checkpoint=str(CHECKPOINT.relative_to(ROOT)), bullets=300, variant=variant))
            for r in failed:
                historical.setdefault(int(r["seed"]), []).append(dict(
                    source=str(source.relative_to(ROOT)), seconds=float(r["model_survival_seconds"])))
            if len(rows) >= 1000:
                large_seeds.update(int(r["seed"]) for r in rows)
                large_deaths.update(int(r["seed"]) for r in failed)

    assert (ROOT / "tools/pixel_guard_continuation.py").read_text() == (REPAIR / "forward_controller.py").read_text()
    current = {}
    for test in ("forward22", "regression67", "fixed200"):
        source = REPAIR / test / "evaluation_episodes.csv"
        for r in read_csv(source):
            seed = int(r["seed"])
            if seed in historical:
                current[seed] = dict(seed=seed, seconds=float(r["model_survival_seconds"]),
                                     reason=r["termination_reason"], source=str(source.relative_to(ROOT)))
    for r in read_csv(ROOT / "diagnostics/current_verified_deaths_20260919/collision_positions.csv"):
        seed = int(r["seed"])
        current[seed] = dict(seed=seed, seconds=float(r["survival_seconds"]), reason="collision",
                             source=r["position_source"])
    pending = sorted(set(historical) - set(current))
    save_csv(OUT / "test_inventory.csv", tests)
    save_csv(OUT / "verified_current_results.csv", sorted(current.values(), key=lambda r:r["seed"]))
    (OUT / "replay_seeds.json").write_text(json.dumps(pending), encoding="utf-8")
    (OUT / "historical_seed_sources.json").write_text(json.dumps(historical, indent=2), encoding="utf-8")
    summary = dict(test_count=len(tests), historical_death_seeds=len(historical),
                   large_test_unique_seeds=len(large_seeds), large_test_unique_deaths=len(large_deaths),
                   large_test_deaths_verified_surviving=sum(current.get(s, {}).get("reason")=="time_limit" for s in large_deaths),
                   current_records=len(current), replay_seeds=pending,
                   large_tests=[t for t in tests if t["episodes"]>=1000])
    (OUT / "inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
