import argparse
from pathlib import Path

import numpy as np

from traffic_rl.baselines.fixed_timer import tuned_timer
from traffic_rl.envs.base_env import GREEN_APPROACHES
from traffic_rl.envs.simple_env import Demand, SimpleTrafficEnv
from traffic_rl.episode import Controller, run_episode
from traffic_rl.evaluate import EVAL_SEEDS
from traffic_rl.train import AGENTS
from traffic_rl.utils.plotting import LABELS, PHASE_NAMES, Snapshot, render_demo_gif


def trace_episode(env: SimpleTrafficEnv, controller: Controller, seed: int) -> list[Snapshot]:
    snapshots: list[Snapshot] = []

    def record(*_: object) -> None:
        snapshots.append(Snapshot(env.sim_time, env.phase, tuple(int(q) for q in env.queues)))

    run_episode(env, controller, seed, on_transition=record)
    return snapshots


def demand_windows(demand: Demand, balance: float = 1.2) -> list[tuple[float, str]]:
    windows = []
    for end, rates in demand:
        loads = GREEN_APPROACHES @ np.asarray(rates)
        if loads.max() <= balance * loads.min():
            windows.append((end, "balanced"))
        else:
            windows.append((end, f"{PHASE_NAMES[int(loads.argmax())]} heavy"))
    return windows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render a GIF of a trained agent next to the tuned fixed timer."
    )
    parser.add_argument("--agent", choices=sorted(AGENTS), default="q_learning")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    args = parser.parse_args(argv)

    env = SimpleTrafficEnv()
    model = AGENTS[args.agent].load(args.output / "models" / f"{args.agent}.npz")
    controllers: dict[str, tuple[str, Controller]] = {
        args.agent: (LABELS[args.agent], model),
        "fixed_tuned": tuned_timer(args.output / "evaluation.csv"),
    }
    traces = {
        key: (label, trace_episode(env, controller, args.seed))
        for key, (label, controller) in controllers.items()
    }
    path = args.output / "demo.gif"
    caption = f"Same demand schedule, held-out seed {args.seed}"
    render_demo_gif(traces, demand_windows(env.config.demand), path, caption)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
