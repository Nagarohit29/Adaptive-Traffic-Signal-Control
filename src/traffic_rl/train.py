import argparse
from pathlib import Path
from typing import Any

import yaml

from traffic_rl.agents.q_learning import Discretizer, QLearningAgent
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv
from traffic_rl.episode import run_episode
from traffic_rl.utils.logging_utils import write_rows

AGENTS = {"q_learning": QLearningAgent}


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config


def epsilon_at(episode: int, episodes: int, start: float, end: float, fraction: float) -> float:
    decay_episodes = max(1, round(episodes * fraction))
    return end + (start - end) * max(0.0, 1 - episode / decay_episodes)


def train(
    agent_name: str, config: dict[str, Any], output: Path, episodes: int | None = None
) -> Path:
    env = SimpleTrafficEnv(SimpleEnvConfig(**config.get("env", {})))
    agent_config, training = config["agent"], config["training"]
    episodes = episodes or training["episodes"]
    agent = AGENTS[agent_name](
        Discretizer(agent_config["queue_edges"], agent_config["elapsed_edges"]),
        agent_config["alpha"],
        agent_config["gamma"],
        seed=training["seed"],
    )
    metrics_path = output / "runs" / agent_name / "metrics.csv"
    rows = []
    for episode in range(episodes):
        agent.epsilon = epsilon_at(
            episode,
            episodes,
            training["epsilon_start"],
            training["epsilon_end"],
            training["epsilon_decay_fraction"],
        )
        metrics = run_episode(env, agent, seed=episode, greedy=False, on_transition=agent.update)
        rows.append({"episode": episode, "epsilon": agent.epsilon, **metrics})
        if (episode + 1) % training["checkpoint_every"] == 0:
            agent.save(output / "checkpoints" / agent_name / f"episode_{episode + 1:04d}.npz")
            write_rows(metrics_path, rows)
            print(
                f"episode {episode + 1:4d}  epsilon {agent.epsilon:.2f}  "
                f"avg wait {metrics['avg_wait']:.1f} s"
            )
    write_rows(metrics_path, rows)
    model_path = output / "models" / f"{agent_name}.npz"
    agent.save(model_path)
    return model_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a traffic signal agent.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="q_learning")
    parser.add_argument("--config", type=Path, help="defaults to configs/<agent>.yaml")
    parser.add_argument("--episodes", type=int, help="overrides the config episode count")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    config = load_config(args.config or Path("configs") / f"{args.agent}.yaml")
    print(f"saved {train(args.agent, config, args.output, args.episodes)}")


if __name__ == "__main__":
    main()
