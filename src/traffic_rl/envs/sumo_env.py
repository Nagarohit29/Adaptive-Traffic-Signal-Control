import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import sumo
import traci
from numpy.typing import NDArray

from traffic_rl.envs.base_env import APPROACHES, GREEN_APPROACHES, Observation
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv

TLS_ID = "C"
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
COORDINATES = {"N": (0, 200), "S": (0, -200), "E": (200, 0), "W": (-200, 0)}
INCOMING_LANES = tuple(f"{approach}2C_0" for approach in APPROACHES)


def sumo_binary(name: str) -> str:
    return os.path.join(sumo.SUMO_HOME, "bin", name)


def write_network(directory: Path) -> Path:
    outer_nodes = "".join(
        f'<node id="{approach}" x="{x}" y="{y}"/>' for approach, (x, y) in COORDINATES.items()
    )
    edges = "".join(
        f'<edge id="{approach}2C" from="{approach}" to="C" numLanes="1" speed="13.89"/>'
        f'<edge id="C2{approach}" from="C" to="{approach}" numLanes="1" speed="13.89"/>'
        for approach in APPROACHES
    )
    nodes_file = directory / "net.nod.xml"
    edges_file = directory / "net.edg.xml"
    net_file = directory / "net.net.xml"
    nodes_file.write_text(
        f'<nodes><node id="C" x="0" y="0" type="traffic_light"/>{outer_nodes}</nodes>'
    )
    edges_file.write_text(f"<edges>{edges}</edges>")
    subprocess.run(
        [
            sumo_binary("netconvert"),
            "--node-files",
            str(nodes_file),
            "--edge-files",
            str(edges_file),
            "--output-file",
            str(net_file),
            "--no-turnarounds",
            "true",
            "--no-warnings",
            "true",
        ],
        check=True,
        capture_output=True,
    )
    return net_file


def write_routes(directory: Path, config: SimpleEnvConfig) -> Path:
    routes = "".join(
        f'<route id="{approach}" edges="{approach}2C C2{OPPOSITE[approach]}"/>'
        for approach in APPROACHES
    )
    flows = []
    begin = 0.0
    for window, (end, rates) in enumerate(config.demand):
        for approach, rate in zip(APPROACHES, rates, strict=True):
            if rate > 0:
                flows.append(
                    f'<flow id="{approach}{window}" route="{approach}" begin="{begin}" '
                    f'end="{end}" probability="{rate}" departSpeed="max"/>'
                )
        begin = end
    routes_file = directory / "routes.rou.xml"
    routes_file.write_text(f"<routes>{routes}{''.join(flows)}</routes>")
    return routes_file


class SumoTrafficEnv(SimpleTrafficEnv):
    def __init__(self, config: SimpleEnvConfig | None = None) -> None:
        super().__init__(config)
        if max(max(rates) for _, rates in self.config.demand) > 1:
            raise ValueError("SUMO inserts at most one vehicle per approach per second")
        self._workdir = tempfile.TemporaryDirectory()
        directory = Path(self._workdir.name)
        self._sumo_args = [
            "-n",
            str(write_network(directory)),
            "-r",
            str(write_routes(directory, self.config)),
            "--step-length",
            "1",
            "--time-to-teleport",
            "-1",
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]
        self._label = f"traffic_rl_{id(self)}"
        self._conn: Any = None
        self._link_approaches: list[int] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        args = [*self._sumo_args, "--seed", str(int(self.np_random.integers(2**31 - 1)))]
        if self._conn is None:
            traci.start([sumo_binary("sumo"), *args], label=self._label)
            self._conn = traci.getConnection(self._label)
            self._link_approaches = [
                INCOMING_LANES.index(links[0][0])
                for links in self._conn.trafficlight.getControlledLinks(TLS_ID)
            ]
        else:
            self._conn.load(args)
        return observation, info

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._workdir.cleanup()

    def _advance(self, seconds: int, green: NDArray[np.bool_] | None) -> tuple[int, int, int]:
        lit = GREEN_APPROACHES[self.phase] if green is None else green
        color = "y" if green is None else "G"
        self._conn.trafficlight.setRedYellowGreenState(
            TLS_ID, "".join(color if lit[approach] else "r" for approach in self._link_approaches)
        )
        arrivals = departures = queue_seconds = 0
        for _ in range(min(seconds, self.config.episode_length - self.sim_time)):
            self._conn.simulationStep()
            self.queues = np.array(
                [self._conn.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES],
                dtype=np.int64,
            )
            if green is not None:
                self.elapsed += 1
            self.sim_time += 1
            arrivals += self._conn.simulation.getDepartedNumber()
            departures += self._conn.simulation.getArrivedNumber()
            queue_seconds += int(self.queues.sum())
        return arrivals, departures, queue_seconds
