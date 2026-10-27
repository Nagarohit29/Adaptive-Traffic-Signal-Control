from dataclasses import dataclass

from traffic_rl.envs.base_env import ELAPSED_INDEX, PHASE_INDEX, Observation


@dataclass(frozen=True)
class FixedTimer:
    green_ns: int = 30
    green_ew: int = 30

    def __post_init__(self) -> None:
        if self.green_ns <= 0 or self.green_ew <= 0:
            raise ValueError("green times must be positive")

    def reset(self) -> None:
        return None

    def act(self, obs: Observation, greedy: bool = True) -> int:
        phase = int(obs[PHASE_INDEX])
        green = (self.green_ns, self.green_ew)[phase]
        return 1 - phase if obs[ELAPSED_INDEX] >= green else phase
