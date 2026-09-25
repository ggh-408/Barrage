"""Extract existing death positions without rerunning or changing historical data."""
import csv
import json
from pathlib import Path
from plot_death_positions import plot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/historical_deaths_20260919"


def main():
    points = {}
    def add(seed, seconds, x, y, threshold, bullets, path):
        key = (int(seed), float(seconds), float(x), float(y))
        if key not in points:
            points[key] = dict(seed=key[0], survival_seconds=key[1], plane_center_x=key[2],
                               plane_center_y=key[3], safety_threshold=threshold,
                               bullet_count=bullets, sources=[])
        points[key]["sources"].append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / "diagnostics").rglob("*.json")):
        if "historical_deaths_" in str(path) or any(s in str(path).lower() for s in ("teacher", "oracle")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("failures"), list):
            continue
        for r in data["failures"]:
            if isinstance(r, dict) and "plane_center" in r and "seed" in r:
                add(r["seed"], r["survival_seconds"], *r["plane_center"],
                    r.get("safety_threshold", data.get("safety_threshold", data.get("threshold", ""))),
                    data.get("bullet_count", data.get("bullets", "")), path)
    for path in (ROOT / "diagnostics").rglob("collision_positions.csv"):
        if "historical_deaths_" in str(path):
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                add(r["seed"], r["survival_seconds"], r["plane_center_x"], r["plane_center_y"],
                    r.get("safety_threshold", ""), "", path)
    rows = sorted(points.values(), key=lambda r: (r["seed"], r["survival_seconds"]))
    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r["point_id"] = i
        r["sources"] = " | ".join(r["sources"])
    fields = ["point_id", "seed", "survival_seconds", "plane_center_x", "plane_center_y",
              "safety_threshold", "bullet_count", "sources"]
    with (OUT / "recorded_deaths.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    plot(OUT / "recorded_deaths.csv", OUT / "death_map.png", "历史测试死亡位置与时间 · 已保存记录")
    seeds = {r["seed"] for r in rows}
    known = json.loads((OUT / "seed_sources.json").read_text())
    missing = [r for r in known["seeds"] if r["seed"] not in seeds]
    (OUT / "missing_positions.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    summary = dict(death_records=len(rows), seeds_with_positions=len(seeds),
                   historical_collision_seeds=len(known["seeds"]), seeds_without_positions=len(missing),
                   minimum_death_seconds=min(r["survival_seconds"] for r in rows),
                   maximum_death_seconds=max(r["survival_seconds"] for r in rows),
                   scope="Historical recorded collisions; model thresholds and guard variants differ; no new replay")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
