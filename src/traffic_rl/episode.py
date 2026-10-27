from typing import Protocol

from traffic_rl.envs.base_env import Observation, TrafficSignalEnv

METRIC_KEYS = ("return", "avg_wait", "avg_queue", "throughput", "switches")


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: Observation, greedy: bool = True) -> int: ...


def run_episode(env: TrafficSignalEnv, controller: Controller, seed: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    controller.reset()
    episode_return = 0.0
    arrivals = departures = queue_seconds = switches = sim_time = 0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller.act(obs))
        episode_return += float(reward)
        arrivals += info["arrivals"]
        departures += info["departures"]
        queue_seconds += info["queue_seconds"]
        switches += info["switched"]
        sim_time = info["sim_time"]
        done = terminated or truncated
    return {
        "return": episode_return,
        "avg_wait": queue_seconds / max(arrivals, 1),
        "avg_queue": queue_seconds / max(sim_time, 1),
        "throughput": float(departures),
        "switches": float(switches),
    }
