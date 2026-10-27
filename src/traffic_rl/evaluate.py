import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from traffic_rl.baselines.fixed_timer import FixedTimer
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv
from traffic_rl.episode import METRIC_KEYS, Controller, run_episode
from traffic_rl.train import AGENTS, load_config
from traffic_rl.utils.logging_utils import read_column, write_rows
from traffic_rl.utils.plotting import LABELS, plot_comparison, plot_learning_curves

TRAIN_SEEDS = range(10)
EVAL_SEEDS = range(1000, 1010)
GREEN_GRID = range(10, 65, 5)

Row = dict[str, float | str]


def summarize(env: SimpleTrafficEnv, controller: Controller, seeds: Sequence[int]) -> Row:
    runs = [run_episode(env, controller, seed=seed) for seed in seeds]
    summary: Row = {"episodes": len(runs)}
    for key in METRIC_KEYS:
        values = [run[key] for run in runs]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def tune_fixed_timer(
    env: SimpleTrafficEnv, grid: Sequence[int], seeds: Sequence[int]
) -> FixedTimer:
    def mean_wait(timer: FixedTimer) -> float:
        return float(np.mean([run_episode(env, timer, seed=seed)["avg_wait"] for seed in seeds]))

    return min((FixedTimer(ns, ew) for ns in grid for ew in grid), key=mean_wait)


def find_controllers(output: Path, tuned: FixedTimer) -> dict[str, tuple[str, Controller]]:
    found: dict[str, tuple[str, Controller]] = {
        "fixed_default": ("Fixed timer 30/30 s", FixedTimer()),
        "fixed_tuned": (f"Fixed timer {tuned.green_ns}/{tuned.green_ew} s, tuned", tuned),
    }
    for name, agent_class in AGENTS.items():
        model_path = output / "models" / f"{name}.npz"
        if model_path.exists():
            found[name] = (LABELS[name], agent_class.load(model_path))
    return found


def evaluate(
    output: Path,
    env_config: SimpleEnvConfig,
    grid: Sequence[int] = GREEN_GRID,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    eval_seeds: Sequence[int] = EVAL_SEEDS,
) -> list[Row]:
    env = SimpleTrafficEnv(env_config)
    tuned = tune_fixed_timer(env, grid, train_seeds)
    rows: list[Row] = [
        {
            "controller": name,
            "description": description,
            **(asdict(controller) if isinstance(controller, FixedTimer) else {}),
            **summarize(env, controller, eval_seeds),
        }
        for name, (description, controller) in find_controllers(output, tuned).items()
    ]
    write_rows(output / "evaluation.csv", rows)

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows, plots / "wait_comparison.png")
    curves = {
        name: read_column(metrics_path, "avg_wait")
        for name in AGENTS
        if (metrics_path := output / "runs" / name / "metrics.csv").exists()
    }
    if curves:
        baseline = next(row for row in rows if row["controller"] == "fixed_tuned")
        reference = (f"{baseline['description']}, held-out mean", float(baseline["avg_wait_mean"]))
        plot_learning_curves(curves, reference, plots / "learning_curve.png")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare trained agents with fixed timers.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--env-config", type=Path, default=Path("configs") / "q_learning.yaml")
    args = parser.parse_args(argv)
    env_config = SimpleEnvConfig(**load_config(args.env_config).get("env", {}))
    for row in evaluate(args.output, env_config):
        mean, std = float(row["avg_wait_mean"]), float(row["avg_wait_std"])
        print(f"{row['description']:<36} avg wait {mean:6.2f} +- {std:.2f} s")


if __name__ == "__main__":
    main()
