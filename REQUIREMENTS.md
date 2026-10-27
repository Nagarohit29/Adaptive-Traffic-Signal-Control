# Requirements

The fixed target this project was built and checked against.

## Assignment requirements

| ID | Requirement | Acceptance criterion | Evidence |
|---|---|---|---|
| R1 | The agent must learn, not follow a hardcoded rule | Episode return rises and average wait falls over training, from a random-init Q-table | Learning-curve plot from training logs |
| R2 | State, action and reward defined clearly | Full MDP spec with units and value ranges | README, "How it works" |
| R3 | RL technique chosen and justified | Written justification backed by an empirical comparison against at least one other algorithm (SARSA) | README, "Results" and comparison plot |
| R4 | Learning curves and performance metrics | Return, average wait time per vehicle, average queue length, throughput | `results/plots/` |
| R5 | Comparison against a non-learning baseline | Q-learning has lower mean average-wait than both the default and the best-tuned fixed timer, over 10 held-out evaluation seeds, reported as mean ± std | `evaluate.py` output, results table |
| R6 | Implemented work separated from future work | An explicit section in the README | README, "What's implemented and what isn't" |
| R7 | Demo-ready | `results/demo.gif`, one-command pipeline, working public demo link | `scripts/run_full_pipeline.sh`, deployed app |
| R8 | External resources acknowledged | Every library, paper and tutorial used is listed | README, "References"; `pyproject.toml` |
| R9 | Every team member can explain their part and the RL concepts | Team-facing explainer kept current with the build | `TEAM_GUIDE.md` (local only) |

## Project constraints

| ID | Constraint |
|---|---|
| C1 | Free hosting only (Streamlit Community Cloud; fallback GitHub Pages) |
| C2 | Environments implement the Gymnasium `Env` API (`reset`, `step`, `observation_space`, `action_space`) |
| C3 | Python 3.11; every dependency pinned to an exact version |
| C4 | Hyperparameters come from `configs/*.yaml`, never hardcoded in training code |
| C5 | Environment, agents and baseline are decoupled and swappable |
| C6 | CI runs lint, type check, unit tests (under ~1 minute) and a 5-episode smoke train; no real training in CI |
| C7 | No secrets in git or CI logs |
| C8 | Demo inputs are validated and bounded; no unbounded simulations |
| C9 | No comments in code files (configs and docs are exempt) |
| C10 | Repo holds only what is needed to understand, run or reproduce the project |
| C11 | Plots, demo and README look plain and hand-made |
| C12 | Results are reproducible from fixed seeds |
| C13 | The core install (simulator, tabular agents, plots) needs neither SUMO nor torch |

## Out of scope

Multi-intersection coordination, PPO/actor-critic, user accounts or persisted user data
in the demo. The README's "What's implemented and what isn't" section lists what was left
out and why.
