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
- ⏸ Break point after the Phase 1 push

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
