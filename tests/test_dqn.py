import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from traffic_rl.envs.simple_env import SimpleTrafficEnv
from traffic_rl.episode import METRIC_KEYS
from traffic_rl.train import load_config

CONFIG_DIR = Path(__file__).parents[1] / "configs"
SRC_DIR = Path(__file__).parents[1] / "src"


def test_core_modules_do_not_import_torch_or_sb3() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(SRC_DIR)!r}); "
        "import traffic_rl.train, traffic_rl.evaluate, traffic_rl.demo; "
        "assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_training_env_scales_observations_and_seeds_episode_k_with_k() -> None:
    pytest.importorskip("stable_baselines3")
    from traffic_rl.agents.dqn import TrainingEnv

    env = TrainingEnv(SimpleTrafficEnv())
    raw = SimpleTrafficEnv()
    for seed in range(3):
        env.reset(seed=999)
        raw.reset(seed=seed)
        for _ in range(40):
            obs, *_ = env.step(1)
            raw_obs, *_ = raw.step(1)
            assert env.observation_space.contains(obs)
        np.testing.assert_allclose(obs, raw_obs / raw.observation_space.high)
    assert env.episode == 2


def test_short_training_writes_metrics_and_a_loadable_model(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from traffic_rl.agents.dqn import DQNController, train_dqn

    config = load_config(CONFIG_DIR / "dqn.yaml")
    config["env"] = {**config["env"], "episode_length": 300}
    config["agent"]["learning_starts"] = 50
    model_path = train_dqn(config, tmp_path, episodes=3)

    with (tmp_path / "runs" / "dqn" / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["episode"] for row in rows] == ["0", "1", "2"]
    assert set(METRIC_KEYS) <= set(rows[0])

    controller = DQNController.load(model_path)
    env = SimpleTrafficEnv()
    obs, _ = env.reset(seed=0)
    controller.reset()
    for _ in range(20):
        action = controller.act(obs)
        assert env.action_space.contains(action)
        obs, *_ = env.step(action)
