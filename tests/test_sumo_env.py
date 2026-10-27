from collections.abc import Iterator

import pytest

pytest.importorskip("traci")

from gymnasium.utils.env_checker import check_env

from traffic_rl.baselines.fixed_timer import FixedTimer
from traffic_rl.envs.base_env import INFO_KEYS
from traffic_rl.envs.simple_env import SimpleEnvConfig
from traffic_rl.envs.sumo_env import SumoTrafficEnv
from traffic_rl.episode import run_episode


@pytest.fixture
def env() -> Iterator[SumoTrafficEnv]:
    sumo_env = SumoTrafficEnv()
    yield sumo_env
    sumo_env.close()


def test_passes_gymnasium_env_checker(env: SumoTrafficEnv) -> None:
    check_env(env, skip_render_check=True)


def test_red_approaches_queue_up(env: SumoTrafficEnv) -> None:
    env.reset(seed=0)
    for _ in range(60):
        obs, _, terminated, _, info = env.step(0)
        assert not terminated
    assert set(info) == set(INFO_KEYS)
    assert info["sim_time"] == 300
    assert obs[2] + obs[3] > obs[0] + obs[1]
    assert env.observation_space.contains(obs)


def test_fixed_timer_episode_is_reproducible(env: SumoTrafficEnv) -> None:
    first = run_episode(env, FixedTimer(), seed=7)
    assert first == run_episode(env, FixedTimer(), seed=7)
    assert first["throughput"] > 0
    assert first["return"] < 0


def test_rejects_rates_sumo_cannot_insert() -> None:
    with pytest.raises(ValueError):
        SumoTrafficEnv(SimpleEnvConfig(demand=((3600.0, (1.5, 0.1, 0.1, 0.1)),)))
