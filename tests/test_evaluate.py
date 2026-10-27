import csv
from pathlib import Path

from traffic_rl.baselines.fixed_timer import FixedTimer
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv
from traffic_rl.episode import run_episode
from traffic_rl.evaluate import evaluate, tune_fixed_timer
from traffic_rl.train import load_config, train

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "q_learning.yaml"


def test_tuning_returns_the_grid_point_with_lowest_wait() -> None:
    env = SimpleTrafficEnv()
    grid = (10, 40)
    waits = {
        (ns, ew): run_episode(env, FixedTimer(ns, ew), seed=0)["avg_wait"]
        for ns in grid
        for ew in grid
    }
    tuned = tune_fixed_timer(env, grid, seeds=[0])
    assert (tuned.green_ns, tuned.green_ew) == min(waits, key=lambda greens: waits[greens])


def test_evaluate_writes_results_and_plots(tmp_path: Path) -> None:
    train("q_learning", load_config(CONFIG_PATH), tmp_path, episodes=2)
    rows = evaluate(
        tmp_path, SimpleEnvConfig(), grid=(15, 30), train_seeds=[0], eval_seeds=[1000, 1001]
    )

    assert [row["controller"] for row in rows] == ["fixed_default", "fixed_tuned", "q_learning"]
    assert all(row["episodes"] == 2 for row in rows)
    with (tmp_path / "evaluation.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    for plot in ("wait_comparison.png", "learning_curve.png"):
        assert (tmp_path / "plots" / plot).stat().st_size > 0
