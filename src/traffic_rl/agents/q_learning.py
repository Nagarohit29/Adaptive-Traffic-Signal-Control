from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from traffic_rl.envs.base_env import (
    APPROACHES,
    ELAPSED_INDEX,
    GREEN_APPROACHES,
    PHASE_INDEX,
    Observation,
)


class Discretizer:
    def __init__(self, queue_edges: Sequence[float], elapsed_edges: Sequence[float]) -> None:
        self.queue_edges = np.asarray(queue_edges, dtype=np.float32)
        self.elapsed_edges = np.asarray(elapsed_edges, dtype=np.float32)
        for edges in (self.queue_edges, self.elapsed_edges):
            if edges.ndim != 1 or edges.size == 0 or np.any(np.diff(edges) <= 0):
                raise ValueError("bin edges must be a non-empty, strictly increasing list")
        queue_bins = (self.queue_edges.size + 1,) * len(APPROACHES)
        self.shape = (*queue_bins, len(GREEN_APPROACHES), self.elapsed_edges.size + 1)
        self.n_states = int(np.prod(self.shape))

    def __call__(self, obs: Observation) -> int:
        queues = np.digitize(obs[:PHASE_INDEX], self.queue_edges)
        elapsed = np.digitize(obs[ELAPSED_INDEX], self.elapsed_edges)
        index = (*queues, int(obs[PHASE_INDEX]), elapsed)
        return int(np.ravel_multi_index(index, self.shape))


class QLearningAgent:
    def __init__(
        self,
        discretizer: Discretizer,
        alpha: float,
        gamma: float,
        epsilon: float = 0.0,
        seed: int | None = None,
        q: NDArray[np.float64] | None = None,
    ) -> None:
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha must be in (0, 1] and gamma in [0, 1]")
        shape = (discretizer.n_states, len(GREEN_APPROACHES))
        if q is not None and q.shape != shape:
            raise ValueError(f"Q-table shape {q.shape} does not match {shape}")
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros(shape) if q is None else q
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def act(self, obs: Observation, greedy: bool = True) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        values = self.q[self.discretizer(obs)]
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation,
        next_action: int,
        terminated: bool,
    ) -> None:
        state = self.discretizer(obs)
        target = reward
        if not terminated:
            target += self.gamma * self.bootstrap_value(self.discretizer(next_obs), next_action)
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def bootstrap_value(self, next_state: int, next_action: int) -> float:
        return float(self.q[next_state].max())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            q=self.q,
            queue_edges=self.discretizer.queue_edges,
            elapsed_edges=self.discretizer.elapsed_edges,
            alpha=self.alpha,
            gamma=self.gamma,
        )

    @classmethod
    def load(cls, path: Path, seed: int | None = 0) -> Self:
        with np.load(path) as data:
            discretizer = Discretizer(data["queue_edges"], data["elapsed_edges"])
            return cls(
                discretizer, float(data["alpha"]), float(data["gamma"]), seed=seed, q=data["q"]
            )
