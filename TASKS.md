# Tasks

The phases of `PROJECT_PLAN.md` §6 broken into checkable tasks. This list is written once;
live status lives in `CHECKLIST.md`. Requirement IDs refer to `REQUIREMENTS.md`.

## Phase 0: Scope (§0.5, §0.75)
- [ ] Compatibility check of the plan, dependencies, repo and hosting; revisions written to plan §10
- [ ] `REQUIREMENTS.md`, `DESIGN.md`, `TASKS.md`, `CHECKLIST.md`

## Phase 1: Setup and scaffolding
- [ ] `pyproject.toml` with pinned dependencies and extras; `.python-version`
- [ ] `LICENSE` (MIT), placeholder `README.md`, `.gitignore`
- [ ] `.github/workflows/ci.yml` (lint, format, type check, tests); `.github/dependabot.yml`
- [ ] Package skeleton under `src/traffic_rl/` with empty modules
- [ ] `docs/architecture.md` (records the deployment decision), `docs/references.md`
- [ ] Local venv; lint and type check pass; first push to `main`

## Phase 2: Environment (R2, C2)
- [ ] `tests/test_env.py` first. It covers:
  - spaces, and the `reset`/`step` contract
  - truncation at 3600 s
  - same seed gives the same trajectory
  - vehicle conservation (arrivals = departures + queued)
  - no departures during yellow
  - reward equals the scaled queue-seconds minus the switch penalty
- [ ] `base_env.py` and `simple_env.py` pass the tests; drop CI's no-tests tolerance
- [ ] SUMO backend on `feat/sumo-env`, 90-minute cap; result logged either way

## Phase 3: Baseline
- [ ] `fixed_timer.py` and `tests/test_baseline.py` (cycle timing, reset)
- [ ] Shared episode loop and per-episode metrics written to CSV
- [ ] Baseline runs end to end through the environment and its metrics are logged

## Phase 4: Q-learning (R1)
- [ ] Discretizer and tabular agent base; `q_learning.py`
- [ ] `tests/test_agents.py`: a toy update matches a hand calculation, greedy action picks the argmax, discretizer bins correctly
- [ ] `configs/q_learning.yaml`; `train.py` CLI with checkpointing
- [ ] CI smoke step (5 episodes)
- [ ] Full training run; the learning curve visibly improves

## Phase 5: Evaluation and plots: safe-to-submit checkpoint (R4, R5)
- [ ] `evaluate.py`: default and tuned fixed timers vs Q-learning on seeds 1000–1009
- [ ] `results/evaluation.csv`, learning-curve and comparison plots
- [ ] Commit, tag `v0-checkpoint`, break

## Phase 6: SARSA (`feat/sarsa`, R3)
- [ ] `sarsa.py`, config and tests; train; add to comparison; PR merged when green

## Phase 7: DQN stretch (`feat/dqn`)
- [ ] `dqn.py` (lazy SB3 import), config; train; add to comparison; PR merged when green

## Phase 8: Demo polish
- [ ] `results/demo.gif`: the trained agent controlling the signals

## Phase 9: Docs (R2, R3, R6, R8)
- [ ] `docs/mdp_formulation.md`, updated `docs/architecture.md`, complete `docs/references.md`
- [ ] Implemented vs. future work section

## Phase 10: Final pass (R7)
- [ ] `scripts/run_full_pipeline.sh`
- [ ] Fresh clone, full pipeline and test suite pass with no manual fixes

## Phase 11: Deployment (C1, C8)
- [ ] `app/streamlit_app.py` and `app/requirements.txt`, with bounded inputs
- [ ] Deployed on Streamlit Community Cloud (fallback: GitHub Pages)
- [ ] Live link verified in a fresh browser session

## Final: README (plan rule 17)
- [ ] Final `README.md`, written against the measured results
