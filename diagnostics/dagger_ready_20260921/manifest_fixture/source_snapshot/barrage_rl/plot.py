"""Generate atomic training/evaluation dashboards for Barrage agents."""

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Some managed Windows profiles make Matplotlib's default user cache read-only.
# A process-local temp cache keeps plotting deterministic and avoids lock-file
# warnings without leaving generated files in the repository.
_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "barrage-matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))


def _atomic_save_figure(figure, output_path: Path, dpi: int = 200) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    figure.savefig(temporary, dpi=dpi)
    os.replace(temporary, output_path)


def _read_metrics(metrics_path: Path) -> Dict[str, List[float]]:
    """Read numeric columns from a metrics CSV, tolerating legacy/missing values."""
    if not metrics_path.exists():
        return {}

    columns: Dict[str, List[float]] = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return columns
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in columns:
                try:
                    columns[name].append(float(row.get(name, "nan")))
                except (TypeError, ValueError):
                    columns[name].append(float("nan"))
    return columns


def _ensure_success_at_limit_ci95_low(data: Dict[str, List[float]]) -> None:
    """Reconstruct the generic success-rate lower bound for legacy histories."""
    if "success_at_limit_ci95_low" in data:
        return
    rates = data["success_at_limit"]
    episode_counts = data.get("episodes")
    if episode_counts is None:
        data["success_at_limit_ci95_low"] = list(rates)
        return

    from .metrics import wilson_lower_bound

    lower_bounds = []
    for rate, count in zip(rates, episode_counts):
        episode_count = max(0, int(round(count)))
        success_count = int(round(rate * episode_count))
        lower_bounds.append(wilson_lower_bound(success_count, episode_count))
    data["success_at_limit_ci95_low"] = lower_bounds


def _dagger_best_index(
    data: Dict[str, List[float]],
    selection_mode: str,
    selection_iqm_floor: float,
) -> int:
    """Return the plotted best round using the same rule as checkpoint promotion."""
    iqm = np.asarray(data["model_iqm"], dtype=float)
    if "model_cvar5" in data:
        cvar5_values = data["model_cvar5"]
    elif "model_p10" in data:
        cvar5_values = data["model_p10"]
    else:
        cvar5_values = data["model_p1"]
    cvar5 = np.asarray(cvar5_values, dtype=float)
    if selection_mode in {"success_at_limit", "iqm_then_success_then_mean"}:
        success = np.asarray(data["success_at_limit"], dtype=float)
        return max(range(len(iqm)), key=lambda index: float(success[index]))
    if selection_mode == "success_then_cvar5":
        success = np.asarray(data["success_at_limit"], dtype=float)
        rmst = np.asarray(data.get("model_rmst", data["model_mean"]), dtype=float)
        return max(
            range(len(iqm)),
            key=lambda index: (
                float(success[index]),
                float(cvar5[index]),
                float(rmst[index]),
            ),
        )
    if selection_mode == "risk_constrained_cvar5":
        return max(
            range(len(iqm)),
            key=lambda index: (
                float(iqm[index] >= selection_iqm_floor),
                float(cvar5[index])
                if iqm[index] >= selection_iqm_floor
                else float(iqm[index]),
                float(iqm[index])
                if iqm[index] >= selection_iqm_floor
                else float(cvar5[index]),
            ),
        )
    return int(np.nanargmax(iqm))


def _dagger_best_metric(
    data: Dict[str, List[float]],
    selection_mode: str,
    selection_iqm_floor: float,
    best_index: int,
) -> str:
    """Return the metric that actually promotes the selected checkpoint."""
    best_iqm = float(data["model_iqm"][best_index])
    iqm_constraint_satisfied = best_iqm >= selection_iqm_floor
    if selection_mode in {"success_at_limit", "iqm_then_success_then_mean"}:
        return "success_at_limit"
    if selection_mode == "success_then_cvar5":
        return "success_at_limit"
    if selection_mode == "risk_constrained_cvar5":
        return "model_cvar5" if iqm_constraint_satisfied else "model_iqm"
    return "model_iqm"


def _set_middle_80_ylim(axis, *series: np.ndarray) -> None:
    """Place the finite data range in the middle 80% of the y-axis."""
    finite_parts = [
        np.asarray(values, dtype=float).ravel()
        for values in series
        if np.asarray(values).size
    ]
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    if span == 0.0:
        padding = max(abs(lower) * 0.10, 0.5)
    else:
        # A padding of one eighth of the data span on both sides makes the
        # data span occupy exactly 80% of the full axis height.
        padding = span / 8.0
    axis.set_ylim(lower - padding, upper + padding)


def _set_reliability_ylim(axis, *series: np.ndarray) -> None:
    """Scale the reliability panel while keeping its percentage ceiling at 100."""
    _set_middle_80_ylim(axis, *series)
    lower, _ = axis.get_ylim()
    axis.set_ylim(lower, 100.0)


def save_round_summary_plot(
    summary_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
) -> Path:
    """Save the cross-round DAgger dashboard using the actual selection rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read_metrics(summary_path)
    required = {
        "round", "model_mean", "model_median", "model_iqm", "model_p1",
        "success_at_limit", "wall_episode_fraction",
        "median_min_wall_distance",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing round summary columns: {', '.join(missing)}")

    def values(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)

    # Preserve plot compatibility with historical DAgger summaries that may
    # not contain the current lower-tail fields.
    if "model_cvar5" not in data:
        data["model_cvar5"] = data.get("model_p10", data["model_p1"])
    if "model_p5" not in data:
        data["model_p5"] = data.get("model_p10", data["model_p1"])
    _ensure_success_at_limit_ci95_low(data)
    if "failure_before_1s" not in data:
        data["failure_before_1s"] = np.zeros_like(values("model_p1"))
    if "failure_before_10s" not in data:
        data["failure_before_10s"] = np.zeros_like(values("model_p1"))
    config = {}
    if config_path is not None and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    selection_mode = str(config.get("selection_mode", "model_iqm"))
    selection_iqm_floor = float(
        config.get("evaluation_episode_limit_seconds", 120.0)
        if selection_mode in {"success_at_limit", "iqm_then_success_then_mean"}
        else config.get("selection_iqm_floor_seconds", 110.0)
    )

    rounds = values("round")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.subplots_adjust(
        left=0.07, right=0.93, bottom=0.10, top=0.88, wspace=0.30, hspace=0.34
    )
    version_match = re.search(
        r"visual_set_v(\d+)", str(config.get("output_dir", "")).lower()
    )
    model_version = int(version_match.group(1)) if version_match else 9
    figure.suptitle(
        f"Visual Set v{model_version} - DAgger Training Summary",
        y=0.955, fontsize=18, weight="bold",
    )

    best_index = _dagger_best_index(data, selection_mode, selection_iqm_floor)
    best_metric = _dagger_best_metric(
        data, selection_mode, selection_iqm_floor, best_index
    )
    best_round = int(rounds[best_index])
    if selection_mode == "risk_constrained_cvar5" and best_metric == "model_cvar5":
        best_label = f"Best: round {best_round} (CVaR5; IQM constraint met)"
    elif selection_mode in {"success_at_limit", "iqm_then_success_then_mean"}:
        best_label = f"Best: round {best_round} (success at limit)"
    elif selection_mode == "success_then_cvar5":
        best_label = f"Best: round {best_round} (success/CVaR5/RMST)"
    else:
        best_label = f"Best: round {best_round}"

    performance = axes[0, 0]
    if selection_mode == "risk_constrained_cvar5":
        iqm_label = f"IQM (>={selection_iqm_floor:g}s constraint)"
    elif (
        selection_mode in {"success_at_limit", "iqm_then_success_then_mean"}
        and best_metric == "success_at_limit"
    ):
        iqm_label = "IQM (at episode limit; reported)"
    elif selection_mode in {"success_then_cvar5", "iqm_then_success_then_mean"}:
        iqm_label = "IQM (reported)"
    else:
        iqm_label = "IQM (selection metric)"
    for name, label, marker in (
        ("model_mean", "Mean", "o"),
        ("model_median", "Median", "s"),
        ("model_iqm", iqm_label, "D"),
    ):
        performance.plot(rounds, values(name), marker=marker, linewidth=2.2, label=label)
    if "model_mean_ci95_low" in data and "model_mean_ci95_high" in data:
        performance.fill_between(
            rounds,
            values("model_mean_ci95_low"),
            values("model_mean_ci95_high"),
            alpha=0.14,
            label="Mean 95% CI",
        )
    if best_metric == "model_iqm":
        performance.scatter(
            rounds[best_index], values("model_iqm")[best_index],
            s=180, facecolors="none", edgecolors="#d62728", linewidths=2.5,
            label=best_label, zorder=5,
        )
    performance.set_title("Survival performance")
    performance.set_ylabel("Survival seconds")
    performance_series = [
        values("model_mean"), values("model_median"), values("model_iqm")
    ]
    if "model_mean_ci95_low" in data and "model_mean_ci95_high" in data:
        performance_series.extend(
            [values("model_mean_ci95_low"), values("model_mean_ci95_high")]
        )
    _set_middle_80_ylim(performance, *performance_series)
    performance_legend = performance.legend(fontsize=9, frameon=False)
    performance_legend.set_zorder(20)

    reliability = axes[0, 1]
    success_label = (
        "Reached episode limit (selection metric)"
        if best_metric == "success_at_limit"
        else "Reached episode limit"
    )
    reliability.plot(
        rounds, 100.0 * values("success_at_limit"), color="#2ca02c",
        marker="o", linewidth=2.5, label=success_label, zorder=3,
    )
    reliability.plot(
        rounds, 100.0 * values("success_at_limit_ci95_low"), color="#1b6f3a",
        marker="s", linestyle="--", linewidth=2.2,
        label="95% CI lower bound", zorder=3,
    )
    if best_metric == "success_at_limit":
        reliability.scatter(
            rounds[best_index], 100.0 * values("success_at_limit")[best_index],
            s=180, facecolors="none", edgecolors="#d62728", linewidths=2.5,
            label=best_label, zorder=5,
        )
    reliability.set_title("Reliability and difficult-case floor")
    reliability.set_ylabel("Reached episode limit (%)", color="#2ca02c")
    reliability.tick_params(axis="y", labelcolor="#2ca02c")
    _set_reliability_ylim(
        reliability,
        100.0 * values("success_at_limit"),
        100.0 * values("success_at_limit_ci95_low"),
    )
    reliability.set_axisbelow(True)
    reliability_legend = reliability.legend(loc="upper left", frameon=False)
    reliability_legend.set_zorder(20)

    rollout = axes[1, 0]
    for name, label, marker, color in (
        ("model_p1", "P1", "o", "#d62728"),
        ("model_p5", "P5", "s", "#ff7f0e"),
        ("model_cvar5", "CVaR5", "D", "#2ca02c"),
    ):
        rollout.plot(
            rounds, values(name), marker=marker, linewidth=2.2, label=label,
            color=color, zorder=3,
        )
    if best_metric == "model_cvar5":
        rollout.scatter(
            rounds[best_index], values("model_cvar5")[best_index],
            s=180, facecolors="none", edgecolors="#d62728", linewidths=2.5,
            label=best_label, zorder=5,
        )
    rollout.set_title("Post-round rollout statistics")
    rollout.set_ylabel("Survival seconds")
    _set_middle_80_ylim(
        rollout, values("model_p1"), values("model_p5"), values("model_cvar5")
    )
    rollout.set_axisbelow(True)
    rollout_legend = rollout.legend(loc="upper right", frameon=False)
    rollout_legend.set_zorder(20)

    safety = axes[1, 1]
    safety.plot(
        rounds, 100.0 * values("wall_episode_fraction"), color="#e45756",
        marker="s", linewidth=2.1, label="Episodes entering <40 px (%)",
    )
    safety.set_title("Wall safety")
    safety.set_ylabel("Episode share (%)")
    _set_middle_80_ylim(safety, 100.0 * values("wall_episode_fraction"))
    wall_axis = safety.twinx()
    wall_axis.plot(
        rounds, values("median_min_wall_distance"), color="#9467bd",
        marker="D", linestyle="--", linewidth=2.2,
        label="Median minimum wall distance",
    )
    wall_axis.set_ylabel("Distance (pixels)", color="#9467bd")
    wall_axis.tick_params(axis="y", labelcolor="#9467bd")
    _set_middle_80_ylim(wall_axis, values("median_min_wall_distance"))
    left_handles, left_labels = safety.get_legend_handles_labels()
    right_handles, right_labels = wall_axis.get_legend_handles_labels()
    safety_legend = wall_axis.legend(
        left_handles + right_handles, left_labels + right_labels,
        fontsize=8, loc="upper right", frameon=False,
    )
    safety_legend.set_zorder(20)

    for axis in axes.ravel():
        axis.set_xlabel("DAgger round")
        axis.set_xticks(rounds)
        axis.set_axisbelow(True)
        axis.grid(True, alpha=0.22)
    figure.text(
        0.5, 0.025,
        "Evaluation: {episodes} episodes/checkpoint | {bullets} bullets | "
        "size {core_size} | speed {core_speed:g} | target p={target_p:.2f} | "
        "fixed held-out seeds".format(
            episodes=int(data.get("episodes", [config.get("evaluation_episodes", 0)])[-1]),
            bullets=int(data.get("bullets", [config.get("bullet_count", 0)])[-1]),
            core_size=int(data.get(
                "bullet_size_min", [config.get("bullet_size", 0)]
            )[-1]),
            core_speed=float(data.get(
                "bullet_speed_min", [config.get("bullet_speed", 0)]
            )[-1]),
            target_p=float(data.get(
                "targeted_bullet_probability",
                [config.get("targeted_bullet_probability", 0)],
            )[-1]),
        ),
        ha="center", fontsize=9, color="#555555",
    )
    _atomic_save_figure(figure, output_path)
    plt.close(figure)
    return output_path
