from pathlib import Path

import pytest
from PIL import Image

from traffic_rl.baselines.fixed_timer import FixedTimer, tuned_timer
from traffic_rl.demo import demand_windows, trace_episode
from traffic_rl.envs.simple_env import DEFAULT_DEMAND, SimpleEnvConfig, SimpleTrafficEnv
from traffic_rl.episode import run_episode
from traffic_rl.utils.logging_utils import write_rows
from traffic_rl.utils.plotting import render_demo_gif

SHORT = SimpleEnvConfig(episode_length=120, demand=((120.0, (0.2, 0.2, 0.05, 0.05)),))


def test_trace_follows_the_episode() -> None:
    env = SimpleTrafficEnv(SHORT)
    trace = trace_episode(env, FixedTimer(20, 20), seed=0)
    times = [shot.sim_time for shot in trace]
    assert times == sorted(times)
    assert times[-1] == 120
    assert trace[-1].queues == tuple(env.queues)
    phase_changes = sum(a.phase != b.phase for a, b in zip(trace, trace[1:], strict=False))
    assert phase_changes == run_episode(env, FixedTimer(20, 20), seed=0)["switches"]


def test_tuned_timer_is_read_back_from_the_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.csv"
    write_rows(
        path,
        [
            {
                "controller": "fixed_default",
                "description": "Default",
                "green_ns": 30,
                "green_ew": 30,
            },
            {"controller": "fixed_tuned", "description": "Tuned", "green_ns": 20, "green_ew": 15},
            {"controller": "q_learning", "description": "Q-learning"},
        ],
    )
    assert tuned_timer(path) == ("Tuned", FixedTimer(20, 15))

    write_rows(path, [{"controller": "q_learning", "description": "Q-learning"}])
    with pytest.raises(ValueError):
        tuned_timer(path)


def test_demand_windows_name_the_heavier_direction() -> None:
    assert demand_windows(DEFAULT_DEMAND) == [
        (1200.0, "north-south heavy"),
        (2400.0, "balanced"),
        (3600.0, "east-west heavy"),
    ]


def test_gif_has_one_frame_per_interval(tmp_path: Path) -> None:
    trace = trace_episode(SimpleTrafficEnv(SHORT), FixedTimer(), seed=0)
    path = tmp_path / "demo.gif"
    render_demo_gif(
        {"q_learning": ("Agent", trace), "fixed_tuned": ("Timer", trace)},
        demand_windows(SHORT.demand),
        path,
        caption="test",
        seconds_per_frame=30,
        fps=4,
        hold_seconds=0,
    )
    with Image.open(path) as gif:
        assert gif.n_frames == 4
