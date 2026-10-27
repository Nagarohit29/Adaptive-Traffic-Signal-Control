from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from traffic_rl.envs.base_env import (
    APPROACHES,
    GREEN_APPROACHES,
    Action,
    Observation,
    TrafficSignalEnv,
    make_info,
)

Demand = tuple[tuple[float, tuple[float, ...]], ...]

DEFAULT_DEMAND: Demand = (
    (1200.0, (0.16, 0.16, 0.06, 0.06)),
    (2400.0, (0.11, 0.11, 0.11, 0.11)),
    (3600.0, (0.06, 0.06, 0.16, 0.16)),
)


@dataclass(frozen=True)
class SimpleEnvConfig:
    episode_length: int = 3600
    decision_interval: int = 5
    yellow_time: int = 3
    saturation_flow: float = 0.5
    demand: Demand = DEFAULT_DEMAND
    reward_scale: float = 0.01
    switch_penalty: float = 0.1
    max_queue_obs: int = 40
    max_elapsed_obs: int = 120

    def __post_init__(self) -> None:
        if self.episode_length <= 0 or self.decision_interval <= 0:
            raise ValueError("episode_length and decision_interval must be positive")
        if self.yellow_time < 0:
            raise ValueError("yellow_time must be non-negative")
        if not 0 < self.saturation_flow <= 1:
            raise ValueError("saturation_flow must be in (0, 1]")
        if not self.demand:
            raise ValueError("demand needs at least one window")
        ends = [end for end, _ in self.demand]
        if ends != sorted(set(ends)) or ends[-1] < self.episode_length:
            raise ValueError("demand window ends must strictly increase and cover the episode")
        for _, rates in self.demand:
            if len(rates) != len(APPROACHES) or min(rates) < 0:
                raise ValueError("each demand window needs one non-negative rate per approach")


class SimpleTrafficEnv(TrafficSignalEnv):
    def __init__(self, config: SimpleEnvConfig | None = None) -> None:
        self.config = config or SimpleEnvConfig()
        super().__init__(self.config.max_queue_obs, self.config.max_elapsed_obs)
        self._window_ends = np.array([end for end, _ in self.config.demand])
        self._window_rates = np.array([rates for _, rates in self.config.demand])
        self.queues = np.zeros(len(APPROACHES), dtype=np.int64)
        self.phase = 0
        self.elapsed = 0
        self.sim_time = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self.queues = np.zeros(len(APPROACHES), dtype=np.int64)
        self.phase = 0
        self.elapsed = 0
        self.sim_time = 0
        return self.make_observation(self.queues, self.phase, self.elapsed), make_info()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in {self.action_space}, got {action!r}")
        next_phase = int(action)
        switched = next_phase != self.phase
        yellow_totals = (0, 0, 0)
        if switched:
            yellow_totals = self._advance(self.config.yellow_time, green=None)
            self.phase, self.elapsed = next_phase, 0
        green_totals = self._advance(self.config.decision_interval, GREEN_APPROACHES[self.phase])
        arrivals, departures, queue_seconds = (
            y + g for y, g in zip(yellow_totals, green_totals, strict=True)
        )
        reward = -self.config.reward_scale * queue_seconds - self.config.switch_penalty * switched
        truncated = self.sim_time >= self.config.episode_length
        info = make_info(self.sim_time, arrivals, departures, queue_seconds, switched)
        observation = self.make_observation(self.queues, self.phase, self.elapsed)
        return observation, reward, False, truncated, info

    def _advance(self, seconds: int, green: NDArray[np.bool_] | None) -> tuple[int, int, int]:
        arrivals = departures = queue_seconds = 0
        for _ in range(min(seconds, self.config.episode_length - self.sim_time)):
            arrived = self.np_random.poisson(self._current_rates())
            self.queues += arrived
            if green is not None:
                leaving = green & (self.queues > 0)
                leaving &= self.np_random.random(len(APPROACHES)) < self.config.saturation_flow
                self.queues -= leaving.astype(np.int64)
                departures += int(leaving.sum())
                self.elapsed += 1
            self.sim_time += 1
            arrivals += int(arrived.sum())
            queue_seconds += int(self.queues.sum())
        return arrivals, departures, queue_seconds

    def _current_rates(self) -> NDArray[np.float64]:
        window = int(np.searchsorted(self._window_ends, self.sim_time, side="right"))
        return self._window_rates[window]
