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
- 🔄 SUMO backend attempt on `feat/sumo-env` (90-minute cap, started 19:27).
  `pip install .[sumo]` worked in about 1 minute; SUMO_HOME resolves inside the venv
  (`sumo.SUMO_HOME`), so there is no system install.

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
- ⛔ PR #3 (SUMO backend) is waiting for the user to merge it. The merge was denied by
  the auto-mode permission check.

### Phase 4: Q-learning
- ✅ `agents/q_learning.py`: `Discretizer` (1536 states) and `QLearningAgent`.
  `bootstrap_value` is the only method SARSA needs to override.
- ✅ `run_episode` takes `greedy` and an `on_transition` callback, so training reuses the
  same loop. It passes `(s, a, r, s', a')`, which covers SARSA too.
- ✅ `configs/q_learning.yaml`; `train.py` CLI with checkpoints every 20 episodes and a CSV
  metrics log
- ✅ `tests/test_agents.py`, 10 cases:
  - discretizer bins and rejection of unsorted edges
  - update matches a hand calculation, including the terminal case
  - greedy picks the argmax, exploration tries both actions, ties are broken randomly
  - save/load round trip
  - invalid hyperparameters rejected
  - epsilon schedule
  - a 2-episode training run
- ✅ CI and the pre-push hook now also run a 5-episode smoke train
- ✅ Full training: 400 episodes in 37 s. Average wait per training episode fell from
  about 24 s (ε ≈ 0.9) to about 12 s (ε = 0.05).
- Bug found and fixed: the first greedy evaluation scored 92.7 ± 240 s.
  - On seed 1007 the policy reached states never visited in training (about two thirds
    of the 1536 states have no updates).
  - With all-zero Q values, `argmax` always returned action 0 (hold), so NS stayed green
    for 483 decisions in a row and the EW queues reached 327 cars.
  - Fix: break ties randomly. A test covers it.
- ✅ Preliminary result on held-out seeds 1000–1009 (Phase 5 does the full tuned
  comparison):
  - Q-learning: 12.22 ± 0.51 s
  - fixed 15/15: 14.84 ± 1.53 s
  - fixed 30/30: 17.23 ± 0.81 s

### Phase 5: Evaluation and plots (safe-to-submit checkpoint)
- ✅ `evaluate.py`:
  - tunes the fixed timer by grid search (10–60 s, step 5, both phases) on training
    seeds 0–9 only
  - scores every controller on held-out seeds 1000–1009
  - writes `results/evaluation.csv`, `plots/wait_comparison.png` and
    `plots/learning_curve.png`
  - picks up every model in `results/models/` automatically, so SARSA and DQN need no
    evaluate changes
- ✅ `utils/plotting.py`:
  - palette checked with the dataviz skill's validator: the three agent colours pass the
    colour-blind and normal-vision checks for every pair
  - the third colour is under 3:1 contrast, so every bar has a direct value label
  - baselines are grey
- ✅ `tests/test_evaluate.py`: tuning picks the grid minimum; evaluate writes the CSV and
  both plots. 29 tests pass.
- ✅ Results, average wait per vehicle, mean ± std over 10 held-out seeds:

  | Controller | Avg wait |
  |---|---|
  | Q-learning | 12.22 ± 0.51 s |
  | Fixed 20/20, tuned on training seeds | 15.14 ± 1.41 s |
  | Fixed 30/30, default | 17.23 ± 0.81 s |

  R5 is met: Q-learning waits 19% less than the tuned timer and 29% less than the
  default. Tuning on the training seeds picked 20/20, while 15/15 does slightly better on
  the held-out seeds (14.84). The number is reported as-is, because tuning on the test
  seeds would leak.
- ✅ Tagged `v0-checkpoint`
- ⏸ Break point

### Parallel subagents (rule 14)
Three agents ran alongside Phase 5, each in its own git worktree. Their files didn't
overlap, so they could run at the same time.
- **Phase 6, SARSA:** PR #4, awaiting the user's merge.
  - `SarsaAgent` overrides only `bootstrap_value`.
  - Held-out result: SARSA 11.43 ± 0.47 s vs Q-learning 12.22 ± 0.51 s.
  - Each agent was trained once, so the gap may be seed noise. After the merge, retrain
    both with 3 agent seeds before claiming SARSA is better.
- **Phase 9 (early), MDP formulation doc:** PR #5, awaiting the user's merge. The
  mismatches it found, fixed on `main`:
  - `yellow_time` error message
  - DESIGN "about 700 decisions" → at most 720
  - DESIGN "worst case 0.44" → every window is 0.44
  The doc still calls SARSA planned; refresh it in Phase 9.
- **Phase 11 prep, Streamlit app:** PR #6, awaiting the user's merge. A local launch
  returned a healthy status. Follow-up: `docs/architecture.md` should say the app
  installs pinned streamlit, gymnasium and numpy and loads the package from `src/`.
- All four PRs (#3–#6) test-merged onto `main` together: no conflicts, 36 tests pass.
  The user is merging them on GitHub.

### Phase 8: Demo polish
- ✅ `demo.py` and `render_demo_gif` in `utils/plotting.py`. The GIF shows the trained
  agent and the tuned fixed timer side by side on held-out seed 1000:
  - top: queues drawn as cars, plus green/red lights and live counts
  - bottom: a shared "vehicles waiting" timeline labelled with the three demand windows
- ✅ `evaluate.py` now writes the tuned timer's green times to `evaluation.csv`, so the
  demo reuses the same baseline instead of re-tuning. `write_rows` handles rows with
  different keys.
- ✅ `tests/test_demo.py`: the trace matches the episode, the tuned timer is read back
  from the CSV, demand windows are labelled, and the frame count is right. 33 tests pass.
- ✅ `results/demo.gif`: 240 frames at 640×464, 2.5 MB. Frames at minutes 20 and 60
  inspected by eye.
- ⏸ Break point

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
