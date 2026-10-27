from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {
    "q_learning": "#2a78d6",
    "sarsa": "#eb6834",
    "dqn": "#1baf7a",
    "fixed_tuned": SECONDARY,
    "fixed_default": MUTED,
}
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA", "dqn": "DQN"}

Row = Mapping[str, float | str]


def new_axes(width: float, height: float, title: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = figure.subplots()
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=AXIS, labelcolor=SECONDARY, labelsize=9)
    axes.set_axisbelow(True)
    axes.set_title(title, loc="left", color=INK, fontsize=11, pad=12)
    return figure, axes


def rolling_mean(values: Sequence[float], window: int) -> NDArray[np.float64]:
    window = max(1, min(window, len(values)))
    return np.convolve(np.asarray(values, dtype=np.float64), np.ones(window) / window, "valid")


def plot_learning_curves(
    curves: Mapping[str, Sequence[float]],
    reference: tuple[str, float],
    path: Path,
    window: int = 20,
) -> None:
    figure, axes = new_axes(7, 4, "Average wait per vehicle during training")
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    for name, waits in curves.items():
        episodes = np.arange(len(waits))
        smooth = rolling_mean(waits, window)
        axes.plot(episodes, waits, color=COLORS[name], linewidth=0.8, alpha=0.25)
        axes.plot(
            episodes[len(episodes) - len(smooth) :],
            smooth,
            color=COLORS[name],
            linewidth=2,
            label=f"{LABELS[name]}, {window}-episode mean",
        )
    label, value = reference
    axes.axhline(value, color=SECONDARY, linewidth=1)
    axes.annotate(
        label,
        xy=(1, value),
        xycoords=axes.get_yaxis_transform(),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=SECONDARY,
        fontsize=9,
    )
    axes.set_xlabel("Training episode", color=SECONDARY)
    axes.set_ylabel("Seconds", color=SECONDARY)
    axes.set_ylim(bottom=0)
    axes.legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="upper right")
    figure.savefig(path)


def plot_comparison(rows: Sequence[Row], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["avg_wait_mean"]), reverse=True)
    means = [float(row["avg_wait_mean"]) for row in ordered]
    stds = [float(row["avg_wait_std"]) for row in ordered]
    positions = np.arange(len(ordered))
    figure, axes = new_axes(7, 0.55 * len(ordered) + 1.4, "Average wait per vehicle")
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.barh(
        positions,
        means,
        xerr=stds,
        height=0.6,
        color=[COLORS[str(row["controller"])] for row in ordered],
        error_kw={"ecolor": SECONDARY, "elinewidth": 1, "capsize": 0},
    )
    for position, mean, std in zip(positions, means, stds, strict=True):
        axes.text(
            mean + std + 0.3, float(position), f"{mean:.1f} s", va="center", color=INK, fontsize=9
        )
    axes.set_yticks(positions, [str(row["description"]) for row in ordered])
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.set_xlim(0, max(m + s for m, s in zip(means, stds, strict=True)) * 1.18)
    seeds = int(float(ordered[0]["episodes"]))
    axes.set_xlabel(f"Seconds, mean ± std over {seeds} held-out seeds", color=SECONDARY)
    figure.savefig(path)
