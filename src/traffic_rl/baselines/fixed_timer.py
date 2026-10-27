import csv
from dataclasses import dataclass
from pathlib import Path

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


def tuned_timer(evaluation: Path) -> tuple[str, FixedTimer]:
    with evaluation.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["controller"] == "fixed_tuned"]
    if not rows:
        raise ValueError(f"no tuned timer in {evaluation}; run python -m traffic_rl.evaluate first")
    row = rows[0]
    return row["description"], FixedTimer(int(row["green_ns"]), int(row["green_ew"]))
