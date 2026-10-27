# Checklist

Live progress against `TASKS.md`. ✅ done · 🔄 in progress · ⛔ blocked (with reason).

## 2026-10-27

### Phase 0: Scope
- ✅ Read `PROJECT_PLAN.md`; checked it against the dependencies, the repo and the hosting options
- ✅ Dependency check: `uv pip compile` resolves sumo-rl 1.4.5, eclipse-sumo 1.28.0,
  stable-baselines3 2.9.2, gymnasium 1.4.0, numpy 2.4.6, matplotlib 3.11.2, pytest, ruff
  and mypy together on Python 3.11 (resolved without installing)
- ✅ Plugin/skill check (§0.5): nothing missing. The dataviz skill (plots), the GitHub
  plugin (PRs) and Playwright (live-link check) cover the plan. The user chose to skip
  Context7.
- ✅ Revisions written to `PROJECT_PLAN.md` §10:
  - Streamlit Cloud instead of HF Spaces
  - `simple_env` always built
  - torch/SUMO as optional extras
  - `main` branch, `app/` folder
- ✅ `REQUIREMENTS.md`, `DESIGN.md`, `TASKS.md`

### Phase 1: Setup and scaffolding
- ✅ `pyproject.toml` (pinned, with extras `dev`/`dqn`/`sumo`/`app`), `.python-version`,
  `LICENSE`, placeholder `README.md`, `.gitignore`
- ✅ CI workflow (ruff, format check, mypy, pytest) and Dependabot
- ✅ Package skeleton under `src/traffic_rl/`
- ✅ `docs/architecture.md` (deployment decision), `docs/references.md`
- ✅ Local venv via uv; ruff, ruff format, mypy all pass
- ✅ `TEAM_GUIDE.md` and `FUTURE_WORK.md` created; git-ignored, verified with `git check-ignore`
- Note: CI tolerates pytest exit code 5 (no tests collected) until Phase 2 adds tests.
  That tolerance is removed in Phase 2.
- ✅ Pushed to `main` as `12e7bd1`
- ⛔ GitHub Actions blocked: no CI job starts ("account is locked due to a billing
  issue"). This is not a code failure, since the same checks pass locally. Needs the
  account owner to fix billing. Until then, the local ruff, mypy and pytest run is the
  gate before every push (rule 4).
- ⛔ Dependabot PRs #1 (actions/checkout 4→7) and #2 (actions/setup-python 5→7) are on
  hold. They can't be verified until CI runs.
- ⏸ Break point after the Phase 1 push
- 🔄 CI blocker: a re-run on 2026-10-27 still showed the account lock. The account is on
  the free plan and the billing usage API shows $0 owed, so the cause is on the account
  side (payment method or verification). Only the account owner can clear it at
  github.com/settings/billing. Workaround: a local `.git/hooks/pre-push` (not committed)
  runs ruff, the format check, mypy and pytest, and blocks any push that fails.

### Phase 2: Environment
- ✅ `tests/test_env.py` written first. 12 tests cover:
  - Gymnasium `check_env`, and the reset state
  - truncation at 3600 s (720 hold steps)
  - same seed gives the same trajectory
  - vehicle conservation
  - 3 s yellow with no departures
  - reward formula
  - invalid action and invalid demand rejected
- ✅ `base_env.py` (shared spaces, observation builder, info keys) and `simple_env.py`
  (Poisson arrivals, saturation-flow departures, yellow lost time, config validation).
  All 12 tests pass.
- ✅ CI's no-tests tolerance removed
- ✅ SUMO backend on `feat/sumo-env`: started 19:27, working by 19:34 (7 of the 90
  minutes).
  - `pip install .[sumo]` took about 1 minute; SUMO_HOME resolves inside the venv, so
    there is no system install.
  - `sumo_env.py` generates the network and routes, then drives SUMO via TraCI as a
    subclass of `SimpleTrafficEnv`.
  - `tests/test_sumo_env.py` (4 tests) skips automatically when traci isn't installed,
    as in CI.
  - Decision: `sumo-rl` dropped for raw TraCI, to keep the same spaces and reward and
    avoid pandas/pettingzoo (plan §10, revision 9).
  - Cross-check on seed 1000: SUMO avg wait 7.0 / 10.2 / 12.0 s for 20/20, 30/30 and
    40/40 timers, against 14.0 / 15.8 / 20.8 s in the custom simulator. The ordering
    matches; absolute values differ because SUMO counts only halted vehicles.

### Phase 3: Baseline
- ✅ `baselines/fixed_timer.py`: switches when elapsed green reaches `green_ns`/`green_ew`
- ✅ `episode.py`: one shared episode loop plus the `Controller` protocol, reused by
  training, evaluation and the app. `utils/logging_utils.py`: CSV writer.
- ✅ `tests/test_baseline.py`: switch timing, invalid green times, reproducible metrics,
  CSV round trip. 17 tests pass in total.
- ✅ End-to-end check on seed 1000:
  - fixed 30/30: avg wait 15.8 s, 109 switches
  - fixed 20/20: avg wait 14.0 s
  - fixed 40/40: avg wait 20.8 s
  The tuned baseline will be clearly stronger than the default, so beating both (R5)
  is a fair bar.
- Fix: the first switch-count test expected one switch per cycle; a cycle has two. The
  test was wrong, not the simulator.

## Decision log

- **Hosting:** HF Spaces' Gradio/Docker Spaces now need a paid plan; only static Spaces
  are free (HF docs fetched 2026-10-27). Chose Streamlit Community Cloud (free, deploys
  from a public GitHub repo, about 2.7 GB RAM). GitHub Pages is the fallback.
- **torch kept out of the core install:** the default Linux torch wheel pulls several GB of
  CUDA libraries, too heavy for CI and the Streamlit Cloud RAM limit. DQN and SUMO sit
  behind the `dqn`/`sumo` extras with lazy imports.
- **Branch renamed** from `master` (no commits yet) to `main`, to match the plan and CI.
- **Process:** the user chose autonomous building and direct pushes to `main` for
  Phases 1–5. SARSA, DQN and SUMO go on feature branches with PRs.

## Tools used (rule 14)

- WebFetch: HF Spaces and Streamlit Cloud docs, for the hosting decision (web search was
  returning server errors)
- uv: dependency resolution check without installing
