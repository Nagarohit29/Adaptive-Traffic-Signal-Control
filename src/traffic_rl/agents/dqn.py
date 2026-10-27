from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from traffic_rl.envs.base_env import Action, Observation, TrafficSignalEnv
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv
from traffic_rl.episode import EpisodeTally
from traffic_rl.utils.logging_utils import write_rows


def observation_scale(env: TrafficSignalEnv) -> Observation:
    return cast(spaces.Box, env.observation_space).high.astype(np.float32)


class TrainingEnv(gym.Wrapper[Observation, Action, Observation, Action]):
    def __init__(self, env: TrafficSignalEnv) -> None:
        super().__init__(env)
        self.scale = observation_scale(env)
        self.observation_space = spaces.Box(0.0, 1.0, self.scale.shape, np.float32)
        self.episode = -1
        self.tally = EpisodeTally()
        self.rows: list[dict[str, float]] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        self.episode += 1
        self.tally = EpisodeTally()
        obs, info = self.env.reset(seed=self.episode, options=options)
        return obs / self.scale, info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.tally.add(float(reward), info)
        if terminated or truncated:
            self.rows.append({"episode": self.episode, **self.tally.metrics()})
        return obs / self.scale, float(reward), terminated, truncated, info


class DQNController:
    def __init__(self, model: DQN, scale: Observation) -> None:
        self.model = model
        self.scale = scale

    def reset(self) -> None:
        return None

    def act(self, obs: Observation, greedy: bool = True) -> int:
        action, _ = self.model.predict(obs / self.scale, deterministic=greedy)
        return int(action)

    @classmethod
    def load(cls, path: Path, env_config: SimpleEnvConfig | None = None) -> "DQNController":
        scale = observation_scale(SimpleTrafficEnv(env_config))
        return cls(DQN.load(path, device="cpu"), scale)


def train_dqn(config: dict[str, Any], output: Path, episodes: int | None = None) -> Path:
    env_config = SimpleEnvConfig(**config.get("env", {}))
    env = TrainingEnv(SimpleTrafficEnv(env_config))
    agent_config, training = dict(config["agent"]), config["training"]
    episodes = episodes or training["episodes"]
    policy_kwargs = {"net_arch": agent_config.pop("net_arch")}
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=training["seed"],
        device="cpu",
        **agent_config,
    )
    max_steps = -(-env_config.episode_length // env_config.decision_interval)
    model.learn(episodes * max_steps, callback=StopTrainingOnMaxEpisodes(episodes))
    write_rows(output / "runs" / "dqn" / "metrics.csv", env.rows)
    model_path = output / "models" / "dqn.zip"
    model.save(model_path)
    return model_path
