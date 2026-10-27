from collections.abc import Callable
from typing import Any, Protocol

from traffic_rl.envs.base_env import Observation, TrafficSignalEnv

METRIC_KEYS = ("return", "avg_wait", "avg_queue", "throughput", "switches")

Transition = Callable[[Observation, int, float, Observation, int, bool], None]


class EpisodeTally:
    def __init__(self) -> None:
        self.episode_return = 0.0
        self.arrivals = self.departures = self.queue_seconds = self.switches = self.sim_time = 0

    def add(self, reward: float, info: dict[str, Any]) -> None:
        self.episode_return += reward
        self.arrivals += info["arrivals"]
        self.departures += info["departures"]
        self.queue_seconds += info["queue_seconds"]
        self.switches += info["switched"]
        self.sim_time = info["sim_time"]

    def metrics(self) -> dict[str, float]:
        return {
            "return": self.episode_return,
            "avg_wait": self.queue_seconds / max(self.arrivals, 1),
            "avg_queue": self.queue_seconds / max(self.sim_time, 1),
            "throughput": float(self.departures),
            "switches": float(self.switches),
        }


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: Observation, greedy: bool = True) -> int: ...


def run_episode(
    env: TrafficSignalEnv,
    controller: Controller,
    seed: int,
    greedy: bool = True,
    on_transition: Transition | None = None,
) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    controller.reset()
    action = controller.act(obs, greedy)
    tally = EpisodeTally()
    done = False
    while not done:
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_action = controller.act(next_obs, greedy)
        if on_transition is not None:
            on_transition(obs, action, float(reward), next_obs, next_action, terminated)
        obs, action = next_obs, next_action
        tally.add(float(reward), info)
        done = terminated or truncated
    return tally.metrics()
