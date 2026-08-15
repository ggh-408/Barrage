"""Plot per-episode survival frequencies for one QDagger evaluation checkpoint."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "barrage-matplotlib")
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_survival_times(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as file:
        values = [
            float(row["model_survival_seconds"])
            for row in csv.DictReader(file)
        ]
    if not values:
        raise ValueError(f"No evaluation episodes found in {path}")
    return np.asarray(values, dtype=np.float64)


def _interquartile_mean(values: np.ndarray) -> float:
    ordered = np.sort(values)
    lower = int(np.floor(len(ordered) * 0.25))
    upper = int(np.ceil(len(ordered) * 0.75))
    return float(ordered[lower:upper].mean())


def plot_survival_frequency(
    episodes_path: Path,
    output_path: Path,
    bin_seconds: float = 5.0,
) -> Path:
    survival = _load_survival_times(episodes_path)
    horizon = float(np.max(survival))
    edges = np.arange(0.0, horizon + bin_seconds, bin_seconds)
    if edges[-1] <= horizon:
        edges = np.append(edges, edges[-1] + bin_seconds)
    counts, edges = np.histogram(survival, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    frequencies = 100.0 * counts / len(survival)

    metrics = {
        "Mean": float(np.mean(survival)),
        "Median": float(np.median(survival)),
        "IQM": _interquartile_mean(survival),
        "P10": float(np.percentile(survival, 10.0)),
    }
    colors = {
        "Mean": "#1f77b4",
        "Median": "#ff7f0e",
        "IQM": "#2ca02c",
        "P10": "#d62728",
    }

    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    figure.subplots_adjust(
        left=0.09, right=0.97, bottom=0.09, top=0.91, hspace=0.28
    )
    figure.suptitle(
        "QDagger step 400,004 — Survival-time frequencies",
        fontsize=18,
        weight="bold",
    )

    distribution = axes[0]
    distribution.plot(
        centers,
        frequencies,
        color="#355f8a",
        marker="o",
        markersize=5,
        linewidth=2.2,
        label=f"5-second bins (n={len(survival)})",
    )
    distribution.fill_between(
        centers, frequencies, color="#6baed6", alpha=0.22
    )
    distribution.set_title("Survival performance — time-frequency distribution")
    distribution.set_ylabel("Episode frequency (%)")
    distribution.set_ylim(bottom=0, top=max(1.0, 1.10 * frequencies.max()))

    rollout = axes[1]
    rollout.plot(
        centers,
        frequencies,
        color="#355f8a",
        marker="o",
        markersize=5,
        linewidth=2.2,
        label=f"5-second bins (n={len(survival)})",
    )
    rollout.fill_between(
        centers, frequencies, color="#6baed6", alpha=0.22
    )
    rollout.set_title(
        "Post-checkpoint rollout statistics — time-frequency distribution"
    )
    rollout.set_xlabel("Survival time (seconds)")
    rollout.set_ylabel("Episode frequency (%)")
    rollout.set_ylim(bottom=0, top=max(1.0, 1.10 * frequencies.max()))

    for axis in axes:
        for name, value in metrics.items():
            axis.axvline(
                value,
                color=colors[name],
                linestyle="--",
                linewidth=1.7,
                alpha=0.9,
                label=f"{name}: {value:.2f}s",
            )
        axis.set_xlim(0, horizon + 0.5 * bin_seconds)
        axis.grid(True, alpha=0.22)
        axis.legend(fontsize=9, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    figure.savefig(temporary, dpi=200)
    plt.close(figure)
    os.replace(temporary, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--bin-seconds", type=float, default=5.0)
    args = parser.parse_args()
    result = plot_survival_frequency(
        args.episodes_csv, args.output_png, args.bin_seconds
    )
    print(result.resolve())


if __name__ == "__main__":
    main()
