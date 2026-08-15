"""Generate atomic training/evaluation dashboards for Barrage agents."""

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


# Keep performance/curriculum signals on the first row and optimization/runtime
# signals on the second row. Missing columns are skipped for legacy CSV files.
RESULT_COLUMNS: Sequence[str] = (
    "train_loss",
    "validation_loss",
    "train_accuracy",
    "validation_accuracy",
    "mean_return_100",
    "mean_survival_seconds_100",
    "mean_score_100",
    "mastery_ratio",
    "bullet_count",
    "noop_baseline_seconds",
    "steps_per_second",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "entropy_coefficient",
    "ppo_minibatches",
    "replay_samples",
    "elapsed_seconds",
)

X_COLUMNS: Sequence[str] = ("global_step", "update", "epoch")
IDENTIFIER_COLUMNS = {"round", *X_COLUMNS}


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


def _read_evaluation_episodes(path: Optional[Path]) -> Dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    rows = _read_metrics(path)
    return {
        name: np.asarray(rows[name], dtype=float)
        for name in ("model_survival_seconds",)
        if name in rows
    }


def save_results_plot(
    metrics_path: Path,
    output_path: Path,
    evaluation_episodes_path: Optional[Path] = None,
    evaluation_summary_path: Optional[Path] = None,
) -> Path:
    """Save all key PPO training metrics as one Ultralytics-style PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read_metrics(metrics_path)
    evaluation = _read_evaluation_episodes(evaluation_episodes_path)
    evaluation_summary = _read_metrics(evaluation_summary_path) if evaluation_summary_path else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_name = next((name for name in X_COLUMNS if name in data), None)
    if x_name is not None:
        x = np.asarray(data[x_name], dtype=float)
    else:
        # Some trainers only emit metric values. Use a one-based row index so
        # those CSV files still produce real curves instead of empty axes.
        row_count = max((len(values) for values in data.values()), default=0)
        x = np.arange(1, row_count + 1, dtype=float)
    columns = [name for name in RESULT_COLUMNS if name in data and name != x_name]
    if not columns:
        columns = [name for name in data if name not in IDENTIFIER_COLUMNS]

    evaluation_plot_count = 2 if evaluation else 0
    plot_count = max(len(columns) + evaluation_plot_count, 2)
    column_count = (plot_count + 1) // 2
    figure, axes = plt.subplots(
        2,
        column_count,
        figsize=(plot_count + 2, 6),
        tight_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for index, name in enumerate(columns):
        values = np.asarray(data[name], dtype=float)
        count = min(x.size, values.size)
        axis = flat_axes[index]
        valid = np.isfinite(x[:count]) & np.isfinite(values[:count])
        axis.plot(
            x[:count][valid],
            values[:count][valid],
            marker=".",
            label=metrics_path.stem,
            linewidth=2,
            markersize=8,
        )
        axis.set_title(name, fontsize=12)

    next_axis = len(columns)
    if evaluation:
        model_times = evaluation["model_survival_seconds"]
        axis = flat_axes[next_axis]
        axis.boxplot(
            (model_times,),
            tick_labels=("model",),
            showmeans=True,
        )
        axis.set_title("episode survival distribution", fontsize=12)
        axis.set_ylabel("seconds")
        next_axis += 1

        axis = flat_axes[next_axis]
        mean = float(model_times.mean())
        if evaluation_summary:
            low = evaluation_summary["model_mean_ci95_low"][0]
            high = evaluation_summary["model_mean_ci95_high"][0]
            errors = np.asarray([[mean - low], [high - mean]])
            axis.errorbar(
                (0,), (mean,), yerr=errors, fmt="o", capsize=6, linewidth=2
            )
        else:
            axis.plot((0,), (mean,), "o")
        axis.set_xticks((0,), ("model",))
        axis.set_title("mean survival (95% CI)", fontsize=12)
        axis.set_ylabel("seconds")
        next_axis += 1

    for axis in flat_axes[next_axis:]:
        axis.set_visible(False)
    _atomic_save_figure(figure, output_path)
    plt.close(figure)
    return output_path


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
        "round", "model_mean", "model_median", "model_iqm", "model_p10",
        "success_at_limit", "wall_episode_fraction",
        "median_min_wall_distance",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing round summary columns: {', '.join(missing)}")

    def values(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)

    # Preserve plot compatibility with historical DAgger summaries. Current
    # runs provide the true 120-second and lower-tail fields.
    if "success_at_120" not in data:
        data["success_at_120"] = data["success_at_limit"]
    if "success_at_120_ci95_low" not in data:
        data["success_at_120_ci95_low"] = data["success_at_limit"]
    if "model_cvar5" not in data:
        data["model_cvar5"] = data["model_p10"]
    if "model_p5" not in data:
        data["model_p5"] = data["model_p10"]
    if "failure_before_1s" not in data:
        data["failure_before_1s"] = np.zeros_like(values("model_p10"))
    if "failure_before_10s" not in data:
        data["failure_before_10s"] = np.zeros_like(values("model_p10"))
    config = {}
    if config_path is not None and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    rounds = values("round")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.subplots_adjust(
        left=0.07, right=0.93, bottom=0.10, top=0.88, wspace=0.30, hspace=0.34
    )
    figure.suptitle(
        "Visual Set v9 - DAgger Training Summary",
        y=0.955, fontsize=18, weight="bold",
    )

    performance = axes[0, 0]
    for name, label, marker in (
        ("model_mean", "Mean", "o"),
        ("model_median", "Median", "s"),
        ("model_iqm", "IQM (selection metric)", "D"),
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
    selection_order = np.lexsort(
        (
            values("model_iqm"), values("model_p10"), values("model_p5"),
            values("model_cvar5"), values("success_at_120_ci95_low"),
        )
    )
    best_index = int(selection_order[-1])
    performance.scatter(
        rounds[best_index], values("model_iqm")[best_index],
        s=180, facecolors="none", edgecolors="#d62728", linewidths=2.5,
        label=f"Best: round {int(rounds[best_index])}", zorder=5,
    )
    performance.set_title("Survival performance")
    performance.set_ylabel("Survival seconds")
    performance.legend(fontsize=9)

    reliability = axes[0, 1]
    reliability.plot(
        rounds, 100.0 * values("success_at_limit"), color="#2ca02c",
        marker="o", linewidth=2.5, label="Reached episode limit",
    )
    reliability.set_title("Reliability and difficult-case floor")
    reliability.set_ylabel("Reached episode limit (%)", color="#2ca02c")
    reliability.tick_params(axis="y", labelcolor="#2ca02c")
    reliability.set_ylim(bottom=0)
    p10_axis = reliability.twinx()
    p10_axis.plot(
        rounds, values("model_p10"), color="#d62728", marker="s", linewidth=2.2,
        label="P10 survival",
    )
    p10_axis.set_ylabel("P10 survival seconds", color="#d62728")
    p10_axis.tick_params(axis="y", labelcolor="#d62728")
    p10_axis.set_ylim(bottom=0)
    left_handles, left_labels = reliability.get_legend_handles_labels()
    right_handles, right_labels = p10_axis.get_legend_handles_labels()
    reliability.legend(
        left_handles + right_handles, left_labels + right_labels,
        loc="upper left",
    )

    rollout = axes[1, 0]
    for name, label, marker in (
        ("model_mean", "Mean", "o"),
        ("model_median", "Median", "s"),
        ("model_iqm", "IQM", "D"),
        ("model_p10", "P10", "^"),
    ):
        rollout.plot(
            rounds, values(name), marker=marker, linewidth=2.2, label=label,
        )
    rollout.set_title("Post-round rollout statistics")
    rollout.set_ylabel("Survival seconds")
    rollout.set_ylim(bottom=0)
    rollout.legend(loc="upper right")

    safety = axes[1, 1]
    safety.plot(
        rounds, 100.0 * values("wall_episode_fraction"), color="#e45756",
        marker="s", linewidth=2.1, label="Episodes entering <40 px (%)",
    )
    safety.set_title("Wall safety")
    safety.set_ylabel("Episode share (%)")
    safety.set_ylim(bottom=0)
    wall_axis = safety.twinx()
    wall_axis.plot(
        rounds, values("median_min_wall_distance"), color="#9467bd",
        marker="D", linestyle="--", linewidth=2.2,
        label="Median minimum wall distance",
    )
    wall_axis.set_ylabel("Distance (pixels)", color="#9467bd")
    wall_axis.tick_params(axis="y", labelcolor="#9467bd")
    left_handles, left_labels = safety.get_legend_handles_labels()
    right_handles, right_labels = wall_axis.get_legend_handles_labels()
    safety.legend(
        left_handles + right_handles, left_labels + right_labels,
        fontsize=8, loc="upper right",
    )

    for axis in axes.ravel():
        axis.set_xlabel("DAgger round")
        axis.set_xticks(rounds)
        axis.grid(True, alpha=0.22)
    figure.text(
        0.5, 0.025,
        "Evaluation: {episodes} episodes/checkpoint | {bullets} bullets | "
        "size {core_size} | speed {core_speed:g} | target p={target_p:.2f} | "
        "fixed held-out seeds".format(
            episodes=int(data.get("episodes", [config.get("evaluation_episodes", 0)])[-1]),
            bullets=int(config.get("bullet_count", 0)),
            core_size=int(config.get("core_bullet_size", 0)),
            core_speed=float(config.get("core_bullet_speed", 0)),
            target_p=float(config.get("targeted_bullet_probability", 0)),
        ),
        ha="center", fontsize=9, color="#555555",
    )
    _atomic_save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def save_qdagger_results_plot(
    history_path: Path, config_path: Path, output_path: Path
) -> Path:
    """Save a compact QDagger training/evaluation dashboard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read_metrics(history_path)
    required = {
        "global_step", "model_mean", "model_median", "model_iqm", "model_p10",
        "success_at_limit", "wall_episode_fraction",
        "median_min_wall_distance", "model_mean_ci95_low", "model_mean_ci95_high",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing QDagger history columns: {', '.join(missing)}")
    if not config_path.exists():
        raise ValueError(f"Missing QDagger config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def values(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)

    # Older v8 histories remain plottable; current fields are reconstructed only
    # where the legacy CSV contains enough information.
    if "success_at_120" not in data:
        data["success_at_120"] = data["success_at_limit"]
    if "success_at_120_ci95_low" not in data:
        data["success_at_120_ci95_low"] = data["success_at_limit"]
    if "model_cvar5" not in data:
        data["model_cvar5"] = data["model_p10"]
    if "model_p5" not in data:
        data["model_p5"] = data["model_p10"]
    if "failure_before_1s" not in data:
        data["failure_before_1s"] = np.zeros_like(values("model_p10"))
    if "failure_before_10s" not in data:
        data["failure_before_10s"] = np.zeros_like(values("model_p10"))

    steps = values("global_step") / 1_000_000.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.subplots_adjust(
        left=0.07, right=0.93, bottom=0.10, top=0.88, wspace=0.30, hspace=0.34
    )
    checkpoint_name = str(config.get("checkpoint", "")).lower()
    version_match = re.search(r"visual_set_v(\d+)", checkpoint_name)
    model_version = int(version_match.group(1)) if version_match else 9
    figure.suptitle(
        f"Visual Set v{model_version} - QDagger Training Summary", y=0.955,
        fontsize=18, weight="bold",
    )

    performance = axes[0, 0]
    for name, label, marker in (
        ("model_mean", "Mean", "o"),
        ("model_median", "Median", "s"),
        ("model_iqm", "IQM (selection metric)", "D"),
    ):
        performance.plot(steps, values(name), marker=marker, linewidth=2.2, label=label)
    performance.fill_between(
        steps, values("model_mean_ci95_low"), values("model_mean_ci95_high"),
        alpha=0.14, label="Mean 95% CI",
    )
    selection_order = np.lexsort(
        (
            values("model_iqm"), values("model_p10"), values("model_p5"),
            values("model_cvar5"),
            values("success_at_120_ci95_low"),
        )
    )
    best_index = int(selection_order[-1])
    performance.scatter(
        steps[best_index], values("model_iqm")[best_index],
        s=180, facecolors="none", edgecolors="#d62728", linewidths=2.5,
        label=f"Best: {steps[best_index]:.1f}M steps", zorder=5,
    )
    performance.set_title("Survival performance")
    performance.set_ylabel("Survival seconds")
    performance.set_ylim(top=1.10 * np.nanmax(np.concatenate((
        values("model_mean"),
        values("model_median"),
        values("model_iqm"),
        values("model_mean_ci95_high"),
    ))))
    performance.legend(fontsize=9)

    reliability = axes[0, 1]
    reliability.plot(
        steps, 100.0 * values("success_at_limit"), color="#2ca02c",
        marker="o", linewidth=2.5, label="Reached episode limit",
    )
    reliability.set_title("Reliability and difficult-case floor")
    reliability.set_ylabel("Reached episode limit (%)", color="#2ca02c")
    reliability.tick_params(axis="y", labelcolor="#2ca02c")
    reliability.set_ylim(
        bottom=0,
        top=1.10 * np.nanmax(100.0 * values("success_at_limit")),
    )
    p10_axis = reliability.twinx()
    p10_axis.plot(
        steps, values("model_p10"), color="#d62728", marker="s",
        linewidth=2.2, label="P10 survival",
    )
    p10_axis.set_ylabel("P10 survival seconds", color="#d62728")
    p10_axis.tick_params(axis="y", labelcolor="#d62728")
    p10_axis.set_ylim(bottom=0, top=1.10 * np.nanmax(values("model_p10")))
    left_handles, left_labels = reliability.get_legend_handles_labels()
    right_handles, right_labels = p10_axis.get_legend_handles_labels()
    reliability.legend(
        left_handles + right_handles, left_labels + right_labels,
        loc="upper left",
    )

    rollout = axes[1, 0]
    for name, label, marker in (
        ("model_mean", "Mean", "o"),
        ("model_median", "Median", "s"),
        ("model_iqm", "IQM", "D"),
        ("model_p10", "P10", "^"),
    ):
        rollout.plot(
            steps, values(name), marker=marker, linewidth=2.2, label=label,
        )
    rollout.set_title("Post-checkpoint rollout statistics")
    rollout.set_ylabel("Survival seconds")
    rollout.set_ylim(bottom=0, top=1.10 * np.nanmax(np.concatenate((
        values("model_mean"),
        values("model_median"),
        values("model_iqm"),
        values("model_p10"),
    ))))
    rollout.legend(loc="upper right")

    safety = axes[1, 1]
    safety.plot(
        steps, 100.0 * values("wall_episode_fraction"), color="#e45756",
        marker="s", linewidth=2.1, label="Episodes entering <40 px (%)",
    )
    safety.set_title("Wall safety")
    safety.set_ylabel("Episode share (%)")
    safety.set_ylim(
        bottom=0,
        top=1.10 * np.nanmax(100.0 * values("wall_episode_fraction")),
    )
    wall_axis = safety.twinx()
    wall_axis.plot(
        steps, values("median_min_wall_distance"), color="#9467bd",
        marker="D", linestyle="--", linewidth=2.2,
        label="Median minimum wall distance",
    )
    wall_axis.set_ylabel("Distance (pixels)", color="#9467bd")
    wall_axis.tick_params(axis="y", labelcolor="#9467bd")
    wall_axis.set_ylim(
        top=1.10 * np.nanmax(values("median_min_wall_distance"))
    )
    left_handles, left_labels = safety.get_legend_handles_labels()
    right_handles, right_labels = wall_axis.get_legend_handles_labels()
    safety.legend(
        left_handles + right_handles, left_labels + right_labels,
        fontsize=8, loc="upper right",
    )

    for axis in axes.ravel():
        axis.set_xlabel("Training steps (millions)")
        axis.set_xticks(steps)
        axis.grid(True, alpha=0.22)
    figure.text(
        0.5, 0.025,
        "Evaluation: {episodes} episodes/checkpoint | {bullets} bullets | "
        "size {core_size} | speed {core_speed:g} | target p={target_p:.2f} | "
        "fixed held-out seeds".format(
            episodes=int(config.get("evaluation_episodes", 0)),
            bullets=int(config.get("bullet_count", 0)),
            core_size=int(config.get("core_bullet_size", 0)),
            core_speed=float(config.get("core_bullet_speed", 0)),
            target_p=float(config.get("targeted_bullet_probability", 0)),
        ),
        ha="center", fontsize=9, color="#555555",
    )
    _atomic_save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def save_score_curve(metrics_path: Path, output_path: Path) -> Path:
    """Backward-compatible alias for callers using the previous API."""
    return save_results_plot(metrics_path, output_path)
