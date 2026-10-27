from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from traffic_rl.envs.base_env import GREEN_APPROACHES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GO = "#0ca30c"
STOP = "#d03b3b"

COLORS = {
    "q_learning": "#2a78d6",
    "sarsa": "#eb6834",
    "dqn": "#1baf7a",
    "fixed_tuned": SECONDARY,
    "fixed_default": MUTED,
}
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA", "dqn": "DQN"}
PHASE_NAMES = ("north-south", "east-west")

ARM_DIRECTIONS = np.array([(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)])
LANE_OFFSETS = np.array([(-0.65, 0.0), (0.65, 0.0), (0.0, 0.65), (0.0, -0.65)])
ROAD_HALF_WIDTH = 1.3
SCENE_EXTENT = 10.0
CAR_START = 1.9
CAR_SPACING = 0.62
MAX_CARS_DRAWN = 13

Row = Mapping[str, float | str]


class Snapshot(NamedTuple):
    sim_time: int
    phase: int
    queues: tuple[int, ...]


def style_axes(axes: Axes, title: str) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=AXIS, labelcolor=SECONDARY, labelsize=9)
    axes.set_axisbelow(True)
    axes.set_title(title, loc="left", color=INK, fontsize=11, pad=12)


def new_axes(width: float, height: float, title: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = figure.subplots()
    style_axes(axes, title)
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


def intersection_panel(axes: Axes, label: str) -> Callable[[Snapshot], None]:
    axes.set_xlim(-SCENE_EXTENT, SCENE_EXTENT)
    axes.set_ylim(-SCENE_EXTENT, SCENE_EXTENT)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title(label, loc="left", color=INK, fontsize=11)
    long, wide = 2 * SCENE_EXTENT, 2 * ROAD_HALF_WIDTH
    for origin, width, height in (
        ((-SCENE_EXTENT, -ROAD_HALF_WIDTH), long, wide),
        ((-ROAD_HALF_WIDTH, -SCENE_EXTENT), wide, long),
    ):
        axes.add_patch(Rectangle(origin, width, height, facecolor=GRID, linewidth=0))
    cars = axes.scatter([], [], s=26, marker="s", color=SECONDARY, linewidths=0)
    light_positions = LANE_OFFSETS + 1.45 * ARM_DIRECTIONS
    lights = axes.scatter(
        light_positions[:, 0],
        light_positions[:, 1],
        s=55,
        color=STOP,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
    )
    counts = [
        axes.text(x, y, "", ha="center", va="center", color=SECONDARY, fontsize=9)
        for x, y in -2.8 * LANE_OFFSETS + 8.8 * ARM_DIRECTIONS
    ]
    corner = -SCENE_EXTENT + 0.2
    status = axes.text(
        corner, corner, "", va="bottom", color=SECONDARY, fontsize=9, linespacing=1.5
    )

    def update(snapshot: Snapshot) -> None:
        positions = [
            LANE_OFFSETS[arm] + (CAR_START + slot * CAR_SPACING) * ARM_DIRECTIONS[arm]
            for arm, queue in enumerate(snapshot.queues)
            for slot in range(min(queue, MAX_CARS_DRAWN))
        ]
        cars.set_offsets(np.array(positions).reshape(-1, 2))
        lights.set_facecolor([GO if green else STOP for green in GREEN_APPROACHES[snapshot.phase]])
        for text, queue in zip(counts, snapshot.queues, strict=True):
            text.set_text(str(queue))
        waiting = sum(snapshot.queues)
        status.set_text(f"Green: {PHASE_NAMES[snapshot.phase]}\nWaiting now: {waiting}")

    return update


def render_demo_gif(
    traces: Mapping[str, tuple[str, Sequence[Snapshot]]],
    windows: Sequence[tuple[float, str]],
    path: Path,
    caption: str,
    seconds_per_frame: int = 15,
    fps: int = 12,
    hold_seconds: int = 2,
) -> None:
    figure = Figure(figsize=(8, 5.8), dpi=80, facecolor=SURFACE)
    grid = figure.add_gridspec(
        2,
        len(traces),
        height_ratios=(3, 1.5),
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.1,
        hspace=0.32,
        wspace=0.08,
    )
    panels = {
        key: intersection_panel(figure.add_subplot(grid[0, column]), label)
        for column, (key, (label, _)) in enumerate(traces.items())
    }
    times = {key: np.array([shot.sim_time for shot in shots]) for key, (_, shots) in traces.items()}
    totals = {
        key: np.array([sum(shot.queues) for shot in shots]) for key, (_, shots) in traces.items()
    }
    end = max(int(series[-1]) for series in times.values())

    timeline = figure.add_subplot(grid[1, :])
    style_axes(timeline, "Vehicles waiting")
    timeline.grid(axis="y", color=GRID, linewidth=0.8)
    lines = {
        key: timeline.plot([], [], color=COLORS[key], linewidth=2, label=label)[0]
        for key, (label, _) in traces.items()
    }
    start = 0.0
    for window_end, description in windows:
        if start >= end:
            break
        if window_end < end:
            timeline.axvline(window_end / 60, color=AXIS, linewidth=0.8)
        middle = (start + min(window_end, end)) / 120
        timeline.text(
            middle,
            0.92,
            description,
            transform=timeline.get_xaxis_transform(),
            ha="center",
            va="top",
            color=MUTED,
            fontsize=8,
        )
        start = window_end
    timeline.set_xlim(0, end / 60)
    timeline.set_ylim(0, 1.45 * max(1, *(int(series.max()) for series in totals.values())))
    timeline.set_xlabel("Minute", color=SECONDARY)
    timeline.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1),
        ncol=len(traces),
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
    )
    clock = figure.text(0.02, 0.97, "", color=INK, fontsize=12, va="top")
    figure.text(0.98, 0.97, caption, color=MUTED, fontsize=9, ha="right", va="top")

    writer = PillowWriter(fps=fps)
    with writer.saving(figure, str(path), dpi=figure.dpi):
        for moment in sorted({*range(seconds_per_frame, end, seconds_per_frame), end}):
            clock.set_text(f"Minute {moment / 60:.0f} of {end / 60:.0f}")
            for key, (_, shots) in traces.items():
                shown = max(1, int(np.searchsorted(times[key], moment, side="right")))
                panels[key](shots[shown - 1])
                lines[key].set_data(times[key][:shown] / 60, totals[key][:shown])
            writer.grab_frame()
        for _ in range(hold_seconds * fps):
            writer.grab_frame()
