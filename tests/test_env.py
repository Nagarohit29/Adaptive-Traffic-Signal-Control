import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from traffic_rl.envs.base_env import INFO_KEYS
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv


def rollout(seed: int, action_seed: int = 0) -> list[tuple[list[float], float]]:
    env = SimpleTrafficEnv()
    env.reset(seed=seed)
    actions = np.random.default_rng(action_seed).integers(0, 2, size=800)
    trajectory = []
    for action in actions:
        obs, reward, _, truncated, _ = env.step(int(action))
        trajectory.append((obs.tolist(), float(reward)))
        if truncated:
            break
    return trajectory


def test_passes_gymnasium_env_checker() -> None:
    check_env(SimpleTrafficEnv(), skip_render_check=True)


def test_reset_starts_empty_on_phase_zero() -> None:
    obs, info = SimpleTrafficEnv().reset(seed=0)
    assert obs.tolist() == [0, 0, 0, 0, 0, 0]
    assert set(info) == set(INFO_KEYS)


def test_holding_truncates_exactly_at_episode_end() -> None:
    env = SimpleTrafficEnv()
    env.reset(seed=0)
    steps, truncated = 0, False
    while not truncated:
        _, _, terminated, truncated, info = env.step(0)
        assert not terminated
        steps += 1
    assert steps == 3600 // 5
    assert info["sim_time"] == 3600


def test_same_seed_gives_same_trajectory() -> None:
    assert rollout(seed=3) == rollout(seed=3)
    assert rollout(seed=3) != rollout(seed=4)


def test_vehicles_are_conserved() -> None:
    env = SimpleTrafficEnv()
    env.reset(seed=1)
    arrivals = departures = 0
    for action in np.random.default_rng(1).integers(0, 2, size=300):
        _, _, _, _, info = env.step(int(action))
        arrivals += info["arrivals"]
        departures += info["departures"]
        assert arrivals == departures + int(env.queues.sum())
    assert departures > 0


def test_switch_inserts_yellow_with_no_departures() -> None:
    saturated = SimpleEnvConfig(demand=((3600.0, (2.0, 2.0, 2.0, 2.0)),), saturation_flow=1.0)
    env = SimpleTrafficEnv(saturated)
    env.reset(seed=0)
    for _ in range(3):
        env.step(0)

    _, _, _, _, hold = env.step(0)
    assert hold["departures"] == 2 * 5
    assert hold["sim_time"] == 20

    obs, _, _, _, switch = env.step(1)
    assert switch["switched"]
    assert switch["departures"] == 2 * 5
    assert switch["sim_time"] == 28
    assert obs[4:].tolist() == [1, 5]
    assert env.observation_space.contains(obs)


def test_reward_is_scaled_queue_seconds_minus_switch_penalty() -> None:
    env = SimpleTrafficEnv()
    env.reset(seed=2)
    switched = []
    for action in (0, 1, 1, 0):
        _, reward, _, _, info = env.step(action)
        switched.append(info["switched"])
        expected = -0.01 * info["queue_seconds"] - 0.1 * info["switched"]
        assert reward == pytest.approx(expected)
    assert switched == [False, True, False, True]


def test_rejects_invalid_action() -> None:
    env = SimpleTrafficEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(2)


@pytest.mark.parametrize(
    "demand",
    [
        ((3600.0, (0.1, -0.1, 0.1, 0.1)),),
        ((1800.0, (0.1, 0.1, 0.1, 0.1)),),
        ((3600.0, (0.1, 0.1, 0.1)),),
        (),
    ],
)
def test_rejects_invalid_demand(demand: tuple) -> None:
    with pytest.raises(ValueError):
        SimpleEnvConfig(demand=demand)
