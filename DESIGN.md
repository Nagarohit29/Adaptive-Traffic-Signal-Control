# Design

Concrete decisions for this build, implementing `REQUIREMENTS.md`. Values marked *(tuned)*
may change during Phase 4; when they do, the final value is written back here.

## 1. Environment

**Decision:** the custom simulator (`simple_env.py`) is the primary environment and always
exists. CI's smoke test, the deployed demo and the core install all need an environment
with no SUMO dependency (R7, C6, C13).

**SUMO backend (`sumo_env.py`, optional `sumo` extra):**

- Built from scratch: a generated 4-arm, single-lane network and Poisson-like `flow`
  routes from the same demand schedule.
- Drives SUMO through TraCI. It subclasses `SimpleTrafficEnv`, so the spaces, reward,
  info keys and yellow handling are the same code.
- Queue = halting vehicles (speed < 0.1 m/s) on each incoming lane. Absolute waits are
  therefore lower than in the custom simulator: compare controllers within one backend.
- Training and the demo use the custom simulator. SUMO is a realism cross-check for the
  trained policy.

Intersection model:

- Four single-lane approaches, N, S, E, W, through traffic only.
- 1-second simulation ticks.
- Arrivals: `Poisson(λ_i)` vehicles per tick on approach `i`, with `λ` taken from a
  piecewise demand schedule.
- Departures: each approach with a green light releases one queued vehicle per tick with
  probability `saturation_flow = 0.5` (1800 veh/h, a standard single-lane saturation flow).
- Two phases: `0` = NS green / EW red, `1` = EW green / NS red.
- Yellow: switching phases inserts `yellow_time = 3` s of all-red with no departures.

Default demand schedule (vehicles/second per approach):

| Window (s) | N | S | E | W |
|---|---|---|---|---|
| 0 – 1200 | 0.16 | 0.16 | 0.06 | 0.06 |
| 1200 – 2400 | 0.11 | 0.11 | 0.11 | 0.11 |
| 2400 – 3600 | 0.06 | 0.06 | 0.16 | 0.16 |

Which direction dominates changes during the episode, so no single fixed split suits all
three windows. That is the situation adaptive control is meant for. Every window stays
undersaturated (the worst case uses 0.16/0.5 + 0.06/0.5 = 0.44 of capacity before lost
time), so queues stay bounded under any reasonable controller.

## 2. MDP

| Element | Specification |
|---|---|
| Decision interval | Agent acts after every 5 s of green |
| Observation | `Box(float32, shape=(6,))`: queue length per approach N, S, E, W (clipped at 40), current phase (0/1), seconds of green elapsed in the current phase (clipped at 120) |
| Action | `Discrete(2)`: the phase to show next. Same as current means hold for 5 s. Different means 3 s yellow, then 5 s of the new green |
| Reward | `r = −reward_scale × queue_seconds − switch_penalty × switched`, where `queue_seconds` is the sum of all queue lengths over every tick of the step. `reward_scale = 0.01`, `switch_penalty = 0.1` *(tuned)* |
| Episode | 3600 simulated seconds (about 700 decisions). Ends with `truncated = True`; never `terminated` |
| Info | `sim_time`, `arrivals`, `departures`, `queue_seconds`, `switched` |

A switch step lasts 8 s instead of 5, so its reward already includes the queueing during
the yellow lost time. The switch penalty only discourages flicker.

**Tabular state** is built by a discretizer on the agent side, so the environment stays
general enough for DQN:

- queue bins with edges `[3, 7, 13]`, giving 4 bins per approach
- phase: 2 values
- elapsed-green bins with edges `[10, 30]`, giving 3 bins

That is 4⁴ × 2 × 3 = **1536 states × 2 actions** *(tuned)*.

## 3. Metrics

Recorded per episode:

- `return`: sum of rewards
- `avg_wait`: total queue-seconds ÷ arrivals. This is the mean delay per vehicle, and it
  counts vehicles still queued at the end.
- `avg_queue`: total queue-seconds ÷ episode seconds
- `throughput`: departures
- `switches`: number of phase changes

## 4. Controllers, in build order

1. **Fixed timer (baseline).** Cycles NS green for `green_ns` s and EW green for `green_ew`
   s (multiples of 5), using the environment's own yellow. It is evaluated in two forms: the
   default 30/30, and a best split tuned by grid search over 10–60 s (step 5) on training
   seeds only.
2. **Tabular Q-learning (primary).** Off-policy TD control. `α = 0.1`, `γ = 0.95`, ε falls
   linearly from 1.0 to 0.05 over the first 70% of episodes, 400 episodes, Q initialised
   to 0 *(tuned)*.
3. **SARSA (ablation).** Identical settings and discretizer; only the TD target differs
   (on-policy).
4. **DQN (stretch).** Stable-Baselines3 DQN on the raw observation.

Why Q-learning comes first: the state space is small, every value is inspectable, and it
is explainable in a two-minute demo. SARSA gives an on-policy comparison almost for free,
and DQN tests whether function approximation adds anything. The full justification goes
in `docs/mdp_formulation.md`.

## 5. Interfaces

- **Environments:** `gymnasium.Env` subclasses with identical spaces and info keys.
  `base_env.py` holds what every backend shares: the spaces, info keys and metric
  accumulation.
- **Controllers:** `reset() -> None` and `act(obs, greedy=True) -> int`. Learning agents
  add `update(...)`, `save(path)` and `load(path)`. Q-learning and SARSA share one tabular
  base class and override only the TD target.
- **One episode loop**, shared by training, evaluation and the demo app, so it is never
  copied.

## 6. Training and evaluation

- Seeds: training episode `k` uses seed `k`. Evaluation uses seeds 1000–1009, which are
  never trained or tuned on.
- `results/runs/<agent>/metrics.csv`: per-episode training log (ignored by git)
- `results/checkpoints/`: periodic checkpoints (ignored by git)
- `results/models/<agent>.npz`: final model, a few KB (committed)
- `results/evaluation.csv`: mean ± std per controller over the evaluation seeds (committed)
- `results/plots/*.png` and `results/demo.gif` (committed)
- `configs/<agent>.yaml` has `env`, `agent` and `training` sections. Code reads every
  hyperparameter from it (C4).

## 7. Repository layout

Follows `PROJECT_PLAN.md` §4 with the §10 revisions: the repo root plays the role of
`traffic-rl-agent/`, `app/` holds the deployed demo, and `sumo_env.py`/`dqn.py` appear
only when their phases start.

## 8. CI/CD and deployment

- **CI** (`.github/workflows/ci.yml`) runs on every push and on PRs to `main`:
  `ruff check`, `ruff format --check`, `mypy`, `pytest`, then a 5-episode smoke training
  run (added in Phase 4). Installs core and dev dependencies only; no SUMO and no torch.
- **Dependency updates:** Dependabot, monthly, for pip and GitHub Actions.
- **Deployment:** Streamlit Community Cloud serves `app/streamlit_app.py` from `main` and
  redeploys on every push (this is the CD step). The app loads the committed Q-table and
  runs one bounded episode:
  - demand sliders limited to 0–0.25 veh/s per approach
  - episode length fixed at 3600 s
  - no file uploads, no free-text inputs (C8)

## 9. Presentation

Small fixed palette (one colour each for the baseline, Q-learning, SARSA and DQN), one
font, plain axis labels, no figure numbers and no emoji headers (C11). Plot code follows
the installed dataviz skill.
