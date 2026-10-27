import csv
from pathlib import Path

import numpy as np
import pytest

from traffic_rl.agents.q_learning import Discretizer, QLearningAgent
from traffic_rl.envs.base_env import Observation
from traffic_rl.train import epsilon_at, load_config, train

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "q_learning.yaml"


def obs(*values: float) -> Observation:
    return np.array(values, dtype=np.float32)


@pytest.fixture
def discretizer() -> Discretizer:
    return Discretizer([3, 7, 13], [10, 30])


def test_discretizer_bins_match_design(discretizer: Discretizer) -> None:
    assert discretizer.n_states == 4**4 * 2 * 3
    expected = np.ravel_multi_index((0, 1, 2, 3, 0, 0), discretizer.shape)
    assert discretizer(obs(2, 3, 12, 13, 0, 5)) == expected
    assert discretizer(obs(13, 0, 0, 0, 1, 30)) == discretizer(obs(40, 2, 1, 0, 1, 120))
    assert discretizer(obs(40, 40, 40, 40, 1, 120)) == discretizer.n_states - 1


def test_discretizer_rejects_unsorted_edges() -> None:
    with pytest.raises(ValueError):
        Discretizer([7, 3], [10, 30])


def test_update_matches_hand_calculation(discretizer: Discretizer) -> None:
    agent = QLearningAgent(discretizer, alpha=0.5, gamma=0.9)
    state, next_state = obs(0, 0, 0, 0, 0, 5), obs(5, 0, 0, 0, 0, 10)
    agent.q[discretizer(next_state)] = [1.0, 3.0]

    agent.update(state, 1, -2.0, next_state, 0, terminated=False)
    assert agent.q[discretizer(state), 1] == pytest.approx(0.5 * (-2.0 + 0.9 * 3.0))

    agent.update(state, 0, -2.0, next_state, 0, terminated=True)
    assert agent.q[discretizer(state), 0] == pytest.approx(0.5 * -2.0)


def test_greedy_takes_argmax_and_exploration_tries_both(discretizer: Discretizer) -> None:
    agent = QLearningAgent(discretizer, alpha=0.1, gamma=0.9, epsilon=1.0, seed=0)
    state = obs(4, 4, 0, 0, 0, 15)
    agent.q[discretizer(state)] = [-1.0, 2.0]
    assert agent.act(state) == 1
    assert {agent.act(state, greedy=False) for _ in range(50)} == {0, 1}


def test_greedy_breaks_ties_randomly_so_unseen_states_cannot_trap_it(
    discretizer: Discretizer,
) -> None:
    agent = QLearningAgent(discretizer, alpha=0.1, gamma=0.9, seed=0)
    unseen = obs(1, 1, 30, 30, 0, 120)
    assert {agent.act(unseen) for _ in range(50)} == {0, 1}


def test_save_and_load_round_trip(discretizer: Discretizer, tmp_path: Path) -> None:
    agent = QLearningAgent(discretizer, alpha=0.2, gamma=0.8)
    agent.q[7] = [0.5, -0.5]
    path = tmp_path / "model.npz"
    agent.save(path)
    loaded = QLearningAgent.load(path)
    np.testing.assert_array_equal(loaded.q, agent.q)
    assert (loaded.alpha, loaded.gamma) == (0.2, 0.8)
    assert loaded.discretizer.shape == discretizer.shape


@pytest.mark.parametrize("alpha, gamma", [(0.0, 0.9), (0.1, 1.5)])
def test_rejects_invalid_hyperparameters(
    discretizer: Discretizer, alpha: float, gamma: float
) -> None:
    with pytest.raises(ValueError):
        QLearningAgent(discretizer, alpha=alpha, gamma=gamma)


def test_epsilon_decays_linearly_then_holds() -> None:
    schedule = [epsilon_at(k, 100, 1.0, 0.05, 0.7) for k in (0, 35, 70, 99)]
    assert schedule == pytest.approx([1.0, 0.525, 0.05, 0.05])


def test_train_writes_metrics_checkpoints_and_model(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["training"]["checkpoint_every"] = 1
    model_path = train("q_learning", config, tmp_path, episodes=2)

    with (tmp_path / "runs" / "q_learning" / "metrics.csv").open(newline="") as handle:
        assert [row["episode"] for row in csv.DictReader(handle)] == ["0", "1"]
    assert len(list((tmp_path / "checkpoints" / "q_learning").glob("*.npz"))) == 2
    assert np.any(QLearningAgent.load(model_path).q != 0)
