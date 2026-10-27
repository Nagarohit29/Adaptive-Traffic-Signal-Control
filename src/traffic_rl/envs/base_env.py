from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

APPROACHES = ("N", "S", "E", "W")
GREEN_APPROACHES = np.array([[True, True, False, False], [False, False, True, True]])
INFO_KEYS = ("sim_time", "arrivals", "departures", "queue_seconds", "switched")
PHASE_INDEX = len(APPROACHES)
ELAPSED_INDEX = PHASE_INDEX + 1

Observation = NDArray[np.float32]
Action = int | np.integer[Any]


def make_info(
    sim_time: int = 0,
    arrivals: int = 0,
    departures: int = 0,
    queue_seconds: int = 0,
    switched: bool = False,
) -> dict[str, Any]:
    return dict(
        zip(INFO_KEYS, (sim_time, arrivals, departures, queue_seconds, switched), strict=True)
    )


class TrafficSignalEnv(gym.Env[Observation, Action]):
    def __init__(self, max_queue_obs: int, max_elapsed_obs: int) -> None:
        self.max_queue_obs = max_queue_obs
        self.max_elapsed_obs = max_elapsed_obs
        high = np.array(
            [max_queue_obs] * len(APPROACHES) + [len(GREEN_APPROACHES) - 1, max_elapsed_obs],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=np.zeros_like(high), high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(GREEN_APPROACHES))

    def make_observation(self, queues: NDArray[np.int64], phase: int, elapsed: int) -> Observation:
        clipped = np.minimum(queues, self.max_queue_obs)
        return np.array([*clipped, phase, min(elapsed, self.max_elapsed_obs)], dtype=np.float32)
