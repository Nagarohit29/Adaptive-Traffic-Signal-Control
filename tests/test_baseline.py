import csv
from pathlib import Path

import pytest

from traffic_rl.baselines.fixed_timer import FixedTimer
from traffic_rl.envs.base_env import ELAPSED_INDEX, PHASE_INDEX
from traffic_rl.envs.simple_env import SimpleTrafficEnv
from traffic_rl.episode import METRIC_KEYS, run_episode
from traffic_rl.utils.logging_utils import write_rows


def test_switches_after_configured_green_times() -> None:
    env = SimpleTrafficEnv()
    obs, _ = env.reset(seed=0)
    timer = FixedTimer(green_ns=20, green_ew=10)
    switch_points = set()
    for _ in range(100):
        action = timer.act(obs)
        if action != obs[PHASE_INDEX]:
            switch_points.add((int(obs[PHASE_INDEX]), int(obs[ELAPSED_INDEX])))
        obs, *_ = env.step(action)
    assert switch_points == {(0, 20), (1, 10)}


@pytest.mark.parametrize("greens", [(0, 30), (30, -5)])
def test_rejects_non_positive_green(greens: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        FixedTimer(*greens)


def test_episode_metrics_are_consistent_and_reproducible() -> None:
    env = SimpleTrafficEnv()
    metrics = run_episode(env, FixedTimer(), seed=5)
    assert tuple(metrics) == METRIC_KEYS
    assert metrics == run_episode(env, FixedTimer(), seed=5)
    assert metrics["throughput"] > 0
    assert metrics["return"] < 0
    assert metrics["switches"] == pytest.approx(2 * 3600 / (30 + 30 + 2 * 3), abs=2)


def test_metrics_round_trip_through_csv(tmp_path: Path) -> None:
    rows = [{"episode": k, **run_episode(SimpleTrafficEnv(), FixedTimer(), seed=k)} for k in (0, 1)]
    path = tmp_path / "runs" / "metrics.csv"
    write_rows(path, rows)
    with path.open(newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert [float(row["avg_wait"]) for row in loaded] == [row["avg_wait"] for row in rows]
