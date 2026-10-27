# Adaptive Traffic Signal Control — RL Hackathon Project Plan

> **For Claude Code (autonomous mode).** This is the single source of truth for *what to
> build and in what order*. Read this file fully before writing any code, and follow the
> workflow rules in the final section on every single task you execute — no exceptions.
>
> Early on, create two companion files yourself (not provided here — you write them):
> - **`TEAM_GUIDE.md`** — a deep-dive covering the RL concepts used, the tools and why each
>   was chosen, the architecture with data-flow, the CI/CD and deployment pipelines in
>   detail, and a security recap — written so any teammate can read their relevant section
>   and actually explain it to an evaluator. Keep it updated as real decisions get made
>   (e.g., which environment/algorithm actually ended up being used); don't let it drift
>   from reality.
> - **`FUTURE_WORK.md`** — a running list of good-but-out-of-scope ideas, so they don't
>   clutter this plan or get lost. No deadline on any of it.
>
> **Both of these stay local, outside git.** Create them in the repo root for easy access,
> but add them to `.gitignore` and never `git add`/commit/push them — they're working
> references for the team, not part of the submitted/deployed project.
>
> Separately, and strictly before any build work starts (§0.75): create and commit
> `REQUIREMENTS.md`, `DESIGN.md`, and `TASKS.md`. These lock in scope and decisions up
> front so there's no need to improvise architecture or requirements mid-build.

---

## 0. Project Summary

Build an RL agent that controls traffic signals at a 4-way intersection, learns to minimize
vehicle wait time through interaction with a simulated environment, and demonstrably
outperforms a fixed-timer baseline. Deliver a clean, resume-grade repo with reproducible
training, evaluation plots, a demo GIF, and a one-click way to run everything.

**Assignment constraints:** the assignment brief allots roughly 12 hours, but this plan
does not force that as a hard deadline (see §6) — work at whatever pace produces solid,
tested work, with breaks between phases. Beyond timing, the assignment requires: must show
learning (not a hardcoded solution), must define state/action/reward clearly, must justify
the RL technique chosen, must include learning curves / performance metrics, must separate
"implemented" from "future work," must be demo-ready, must acknowledge external resources
used, and every team member must be able to explain their contribution and the RL concepts
involved.

**Resume goal:** This should read as applied, production-shaped RL work — not a tutorial
clone. Baseline comparison + clean repo + short MDP writeup + plots are non-negotiable.

---

## 0.5 Before Starting — Two Things to Ask/Decide First

**Do not begin Phase 1 until both of these are resolved:**

1. **Plugins/skills check.** Look at what's available (installed skills, connected
   plugins/tools) and compare against what this project will actually need (e.g., a
   plotting/data-viz skill, a deployment-related skill or connector, anything for
   simulation/RL tooling). If something relevant is missing, **ask the user directly**
   whether to add it before building, rather than silently working around the gap or
   silently installing something. This check happens once, up front — not repeatedly
   mid-build.

2. **Deployment research, before any code is written.** Research the current best **free**
   way to deploy/host the finished demo (e.g., a static results page + GIF on GitHub
   Pages, a small interactive demo on Hugging Face Spaces or Streamlit Community Cloud,
   etc. — actually check current options rather than assuming). Choose the best one
   yourself and proceed with it — no need to stop and ask for confirmation on this one.
   Note the chosen platform and the reasoning in `docs/architecture.md` so the decision is
   visible. If the chosen platform turns out not to work when you actually try to deploy
   (quota issues, unsupported dependency, etc.), don't just give up or silently drop
   deployment — troubleshoot it, and if it's genuinely not workable, pick the next-best
   free option, note why the first one failed, and get it working. Deployment is not
   optional; some working free hosting of the demo is expected by the end.

---

## 0.75 Lock Down Scope Before Writing Any Code

Once §0.5 is resolved, and **before Phase 1 of the build plan (§6) starts**, create three
more files in the repo root (committed this time — these are real project deliverables,
not working-only files like `TEAM_GUIDE.md`/`FUTURE_WORK.md`):

- **`REQUIREMENTS.md`** — what the project must actually satisfy: the assignment's own
  requirements pulled from the brief (state/action/reward defined, learning demonstrated,
  RL technique justified, metrics/plots shown, baseline comparison, demo-ready, external
  resources acknowledged), plus any constraints decided in §0/§0.5 (free deployment only,
  Gymnasium-compatible env, etc.). This is the fixed target everything else is checked
  against.
- **`DESIGN.md`** — the concrete design decisions for *this specific build*: which
  environment (SUMO vs. custom simulator) and why, the state/action/reward spec from §1,
  which algorithms and in what order (§3), the repo/architecture layout (§4/§5). This is
  where the choices from §1–§5 of this plan get turned into one committed reference
  document, so later phases implement against a decided design instead of re-deciding
  things mid-build.
- **`TASKS.md`** — the phases from §6 broken into concrete, checkable tasks (finer-grained
  than `CHECKLIST.md`'s live progress log — this is the task breakdown itself, written
  once up front; `CHECKLIST.md` then tracks status against it as work happens).

**Why this matters:** doing this upfront, before any environment/agent code is written,
means there's a fixed design and task list to work against for the rest of the build —
instead of re-deciding architecture or scope mid-implementation. If something in these
three files turns out to be wrong once building starts, update them the same way
`PROJECT_PLAN.md` itself gets updated (rule 1 in §9) — don't just improvise around them
silently.

---

## 1. Problem Framing (MDP Definition)

- **Agent:** Traffic signal controller at a single 4-way intersection (extend to 2–3
  coordinated intersections only if Phase 1–4 finish early — see §6 stretch goals).
- **Environment:** Simulated intersection with 4 approaches (N/S/E/W), each with one or more
  lanes, vehicles arriving stochastically (Poisson arrival process is a reasonable default).
- **State (s):** Per-lane queue length (or a discretized/binned version), current active
  phase, elapsed time in current phase. Keep it low-dimensional for tabular Q-learning;
  expand later for DQN.
- **Action (a):** Which signal phase to activate next (e.g., NS-green/EW-red,
  NS-red/EW-green, plus yellow transition handling). Discrete action space.
- **Reward (r):** Negative of (total waiting time this step) or (sum of queue lengths this
  step). Optionally add a small penalty for unnecessary phase switching to discourage
  flickering behavior.
- **Episode:** Fixed simulated time window (e.g., 1 hour of simulated traffic) or fixed
  number of steps.
- **Learning objective:** Minimize average vehicle wait time / maximize throughput relative
  to a fixed-timer baseline, shown via rising episode reward and falling wait-time curves.

Document this section verbatim (refined as needed) in `docs/mdp_formulation.md` — this is
what gets you marks for "clearly define state/action/reward" and is a strong resume
talking point.

**Acknowledge external resources.** The assignment explicitly requires this. Keep a short
`docs/references.md` (or a "References" section in the README) listing any libraries,
papers, tutorials, or datasets the project draws on beyond standard course material —
update it as you go, not as an afterthought at the end.

---

## 2. Approach Options — Environment Simulation

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **SUMO + `sumo-rl` / `traci`** | Realistic, industry-standard traffic sim, looks impressive, real literature to cite | Installation can be finicky, steeper learning curve, risk of eating hours 1–3 on setup | **Best approach if time allows** — try first, with a hard time-box |
| **Custom lightweight simulator (NumPy/Python, Poisson arrivals, simple queue model)** | Full control, zero install risk, fast to iterate, still realistic enough for the MDP story | Less "real" than SUMO, less visually rich by default | **Best fallback — and arguably best overall given the time-sensitivity of environment setup specifically** |
| **Gym-style wrapper around either** | Clean interface, works with Stable-Baselines3 out of the box, standard RL tooling | Slight extra boilerplate | Do this regardless of which simulator you pick |

**Recommended decision rule (time-boxed):**
1. Spend at most **90 minutes** attempting SUMO setup (`pip install sumo-rl`, `traci`).
2. If not cleanly running by the 90-minute mark → **abandon and switch to the custom
   simulator.** Do not let environment setup risk the whole project. Log this decision in
   the checklist either way.
3. Either way, wrap the environment as a **Gymnasium-compatible `Env`** (`reset()`,
   `step()`, `observation_space`, `action_space`) so the RL algorithm code doesn't care
   which simulator is underneath.

---

## 3. Approach Options — RL Algorithm

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Tabular Q-learning** (discretized state) | Simple, fast to train, easy to explain fully (every Q-value is inspectable), perfect for justifying "why this algorithm" in a 2-min demo | Doesn't scale, needs state discretization/binning | **Best approach for the core/primary deliverable** |
| **SARSA** (on-policy variant) | Near-zero extra effort once Q-learning exists, gives you a built-in comparison/ablation | Marginal novelty alone | **Good secondary comparison, not primary** |
| **DQN** (via Stable-Baselines3) | Handles larger/continuous state space, "looks" more advanced, good resume keyword | More training time, more hyperparameter sensitivity, harder to fully explain under pressure | **Good stretch goal after Q-learning works end-to-end**, not the starting point |
| **PPO / Actor-Critic** | Impressive if it works | Overkill for this problem size, high risk of not converging in time | **Avoid for this timeline** |

**Recommended path:** Tabular Q-learning first (guaranteed to work, guaranteed to be
explainable) → fixed-timer baseline → SARSA comparison → DQN stretch goal if time remains.
This progression also directly satisfies "select an RL technique and justify the choice"
because you'll have empirical comparison data, not just an assertion.

---

## 4. Repository Structure

```
traffic-rl-agent/
├── README.md                     # Problem, approach, results, how to run — the "front door"
├── PROJECT_PLAN.md               # This file
├── REQUIREMENTS.md               # Fixed target: what the project must satisfy (see §0.75)
├── DESIGN.md                     # Locked-in design decisions for this build (see §0.75)
├── TASKS.md                      # Concrete task breakdown of §6's phases (see §0.75)
├── CHECKLIST.md                  # Live progress tracker (see §9)
├── requirements.txt / pyproject.toml
├── .python-version               # Pin exact Python version for reproducibility
├── LICENSE                       # MIT or Apache-2.0 — public resume-facing repo deserves one
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                # CI pipeline (see §7)
├── docs/
│   ├── mdp_formulation.md        # State/action/reward spec, justification of RL technique
│   └── architecture.md           # System diagram + component responsibilities
├── src/
│   └── traffic_rl/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── base_env.py       # Gymnasium-compatible interface
│       │   ├── sumo_env.py       # SUMO-backed implementation (if used)
│       │   └── simple_env.py     # Custom lightweight simulator (fallback/primary)
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── q_learning.py
│       │   ├── sarsa.py
│       │   └── dqn.py            # Thin wrapper around Stable-Baselines3 (stretch)
│       ├── baselines/
│       │   └── fixed_timer.py    # Non-learning baseline controller
│       ├── train.py              # Entry point: python -m traffic_rl.train --agent q_learning
│       ├── evaluate.py           # Runs trained agent vs baseline, produces metrics
│       └── utils/
│           ├── logging_utils.py
│           └── plotting.py       # Learning curves, wait-time plots, GIF generation
├── configs/
│   ├── q_learning.yaml
│   ├── sarsa.yaml
│   └── dqn.yaml
├── tests/
│   ├── test_env.py               # Env sanity checks (reset/step contracts, reward shape)
│   ├── test_agents.py            # Agent update-step correctness on toy transitions
│   └── test_baseline.py
├── notebooks/
│   └── results_analysis.ipynb    # Optional — exploratory analysis, not required for grading
├── results/
│   ├── plots/                    # Generated learning curves, comparison bar charts
│   └── demo.gif                  # Trained agent behavior, for README + demo
├── app/
│   ├── streamlit_app.py          # Deployed demo (see §10, revision 1)
│   └── requirements.txt          # Slim deploy deps for Streamlit Community Cloud
└── scripts/
    └── run_full_pipeline.sh      # One command: train all agents, evaluate, generate plots
```

**Local-only, not in git:** `TEAM_GUIDE.md` and `FUTURE_WORK.md` live in the repo root
alongside everything above, but are listed in `.gitignore` and never committed — see the
note at the top of this file and rule 13 below.

**Design principle:** environment, agent, and baseline are fully decoupled. Swapping
SUMO↔custom-sim or Q-learning↔DQN should never require touching unrelated files. This is
the "production-quality" signal you want on your resume.

---

## 5. Architecture Overview

```
                ┌──────────────────┐
                │   train.py /      │   CLI entry points, config-driven
                │   evaluate.py     │
                └─────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼───────┐  ┌──────▼───────┐
│  Environment  │  │     Agent     │  │   Baseline    │
│ (Gym interface)│  │ (Q-learning/  │  │ (fixed-timer) │
│  simple_env /  │  │  SARSA / DQN) │  │               │
│   sumo_env     │  └───────────────┘  └───────────────┘
└───────┬───────┘
        │
┌───────▼───────┐
│  Logging /     │  Episode reward, wait time, queue length → CSV/JSON
│  Metrics store │
└───────┬───────┘
        │
┌───────▼───────┐
│  Plotting /    │  Learning curves, baseline-vs-agent bar charts, GIF
│  Reporting     │
└───────────────┘
```

Config-driven (`configs/*.yaml`) so hyperparameters are never hardcoded in training code —
this alone is a strong "production habits" signal for reviewers.

---

## 6. Build Plan — Ordered Phases (no fixed time budget)

There is no hard deadline for this build. Move through the phases in order, at whatever
pace produces solid, tested, committed work — faster is fine if quality holds up, and
there's no need to rush or pad time. **Take a genuine break between phases** (not just a
pause mid-task) rather than pushing straight through all of them — a short reset between
phases, especially after the harder ones (Environment, Q-learning, DQN), tends to produce
better decisions than grinding non-stop. Note break points in `CHECKLIST.md` just like any
other step, so progress is easy to pick back up.

| Phase | Tasks |
|---|---|
| **1. Setup & scaffolding** | First: `REQUIREMENTS.md`, `DESIGN.md`, `TASKS.md` per §0.75 (if not already done in §0.75 itself). Then: repo init, folder structure, `requirements.txt`, git init, CI skeleton, empty modules |
| **2. Environment (time-boxed, not the whole project)** | Attempt SUMO (90-min cap on this specific sub-task only); on timeout/failure, pivot to `simple_env.py`. Implement `reset()`/`step()`, write `test_env.py` first (TDD-lite), confirm contract passes |
| **3. Baseline** | Fixed-timer controller, run it through the env, confirm metrics logging works end-to-end before touching RL |
| **4. Q-learning agent** | State discretization, Q-table, epsilon-greedy policy, training loop, checkpointing. Train until reward curve visibly improves |
| **5. Evaluation + plots (core deliverable checkpoint)** | `evaluate.py`: Q-learning vs baseline, learning curve, wait-time comparison plot. **This is the safe-to-submit checkpoint — commit and tag it.** |
| **6. SARSA comparison** | Reuse agent interface, train, add to comparison plot |
| **7. DQN stretch goal** | Only once Phases 1–6 are done and committed. Stable-Baselines3 DQN on the same Gym env |
| **8. Demo polish** | Generate `demo.gif` (pygame/matplotlib animation of agent controlling signals) |
| **9. Docs + MDP writeup** | `docs/mdp_formulation.md`, `docs/architecture.md`, fill in "implemented vs. future work" section |
| **10. Final pass** | Full pipeline re-run from clean clone, run test suite, fix anything broken, update CHECKLIST.md |
| **11. Deployment** | Deploy the finished demo to the free platform chosen in §0.5, troubleshooting as needed until it actually works. Confirm the live link works from a fresh/incognito browser session before calling it done |

**Only hard rule:** Phase 5's checkpoint (working Q-learning agent beating baseline, with
plots) is the minimum viable deliverable and should be reached and committed before moving
on to anything optional (Phases 6–7). Everything after Phase 5 is upside, not required for
the assignment to be submittable. Never let later work put the Phase-5 state at risk —
always work on a branch for SARSA/DQN and merge only when green. The only still-fixed
time-box in this whole plan is the 90-minute SUMO attempt in Phase 2, because that one
specifically risks stalling everything behind it.

---

## 7. CI/CD Pipeline

Keep it simple and genuinely useful — not decorative.

**`.github/workflows/ci.yml`** triggers on every push and PR to `main`:
1. **Lint** — `ruff` (fast, catches real issues) on `src/` and `tests/`
2. **Type check** — `mypy` on `src/` (optional strictness; don't let this block early commits)
3. **Unit tests** — `pytest tests/` (env contract tests, agent update-step tests, baseline
   sanity tests — these must run in under ~1 minute, so they should NOT do full training,
   only small-scale/mocked logic)
4. **Smoke test** — run `train.py` for a trivial number of episodes (e.g., 5) on the custom
   env to confirm the full pipeline executes without crashing (not for convergence,
   purely for "does it run")

**Branch strategy:**
- `main` — always green, always demo-ready
- Feature branches (`feat/sarsa`, `feat/dqn`, `feat/sumo-env`) — merged via PR only after
  tests pass locally
- Tag `v0-checkpoint` right after Phase 5 (§6) as an insurance point you can roll back to

**What NOT to do:** Don't wire up heavy training jobs into CI (no GPU runners, no
multi-hour training in GitHub Actions). CI validates correctness and that the code runs —
actual training happens locally and checkpoints/results get committed as artifacts.

---

## 8. Security

This is a small academic project, not a production service handling real user data — keep
security proportionate, but don't skip the basics:

- **No secrets in git, ever.** Any API key, token, or credential (deployment platform
  tokens included) goes in a `.env` file that is `.gitignore`d, never hardcoded in source.
  Provide a `.env.example` with placeholder values so teammates know what's needed without
  exposing real values.
- **No secrets in CI logs.** If CI needs any credential, use GitHub Actions secrets, never
  plaintext in the workflow file.
- **Dependency hygiene.** Pin dependency versions (`requirements.txt` with versions, not
  bare package names) so the project doesn't silently pull in a compromised or broken
  future release. If a dependency-scanning tool (e.g., `pip-audit`, GitHub's Dependabot) is
  easy to enable for free, enable it — otherwise don't over-engineer this for a hackathon
  project.
- **No arbitrary code execution surfaces.** If the deployed demo takes any user input
  (e.g., letting a visitor tweak traffic arrival rates), validate/bound that input —
  nothing that lets a visitor inject code, hit the filesystem, or run unbounded
  simulations that could be used to hammer the free hosting tier.
- **Scope the deployed demo appropriately.** It's a read-mostly demo of a trained agent,
  not an app with accounts or persistent user data — keep it that way rather than growing
  surface area that would need real security hardening.

---

## 9. Workflow Rules (read this every time, follow for every task)

These rules apply to **every single task** in this plan, no exceptions, from Phase 1
through Phase 11.

1. **Check this plan first.** Before starting any task, re-read the relevant section of
   `PROJECT_PLAN.md`. If something in the plan turns out to be wrong, infeasible, or a
   worse approach than what you discover mid-build (e.g., SUMO setup is hopeless, or a
   better state representation emerges), **update this file** with the change and a short
   note on why, then proceed with the updated plan. The plan is living, not fixed in stone
   — but changes must be written back here, not just acted on silently.

2. **Maintain `CHECKLIST.md`.** Keep a running, dated checklist of:
   - What phase/task you're on
   - What's done (✅) vs. in progress (🔄) vs. blocked (⛔, with reason)
   - Any deviations from this plan and why
   - Any decisions made (e.g., "SUMO abandoned at 1:20 mark, switched to simple_env")
   Update it after every meaningful unit of work, not just at the end of a phase.

3. **Keep a running conversation/decision log.** Append brief notes to `CHECKLIST.md` (or a
   `docs/decision_log.md` if it grows long) summarizing what was tried, what worked, what
   didn't, and why — so anyone (including future-you) can reconstruct the reasoning without
   re-reading all the code.

4. **Test before every commit.** Run `pytest tests/` (and lint) locally before every git
   push. Never push code that fails its own test suite. If a test doesn't exist yet for new
   functionality, write it first or immediately after — don't defer test-writing
   indefinitely.

5. **Commit and push after every successful, working unit of work.** "Successful" means:
   it runs, its tests pass, and it doesn't break anything that previously worked. Small,
   frequent, well-described commits — not one giant end-of-day commit. Suggested commit
   message format: `[phase N] short description — what changed and why`.

6. **Never leave `main` broken.** Do risky/experimental work on a feature branch. Only
   merge to `main` when tests are green.

7. **Keep architecture decoupled.** Environment, agent, and baseline must remain swappable
   without cross-contamination (see §4/§5). If a shortcut under time pressure would break
   this separation, prefer the shortcut that preserves it, and note the tradeoff in the
   checklist.

8. **Distinguish implemented vs. future work explicitly in the final README and
   `docs/` writeups** (not in code comments — see rule 10 below) — this is an explicit
   grading criterion.

9. **At the end of the project**, do a final pass: fresh clone, run `scripts/run_full_pipeline.sh`
   from scratch, confirm it works end-to-end with zero manual fixes, and confirm the README
   alone is enough for a stranger (or evaluator) to understand and run the project.

10. **Do not write comments in code files.** Code should be self-explanatory through clear
    naming, small functions, and sensible structure — not through inline/docstring
    commentary explaining what each line does. Keep logic simple and readable enough that
    comments aren't needed. (Config files, `README.md`, and `docs/` are documentation, not
    code, and are exempt from this.)

11. **Maintain the codebase, don't just extend it.** Before adding new code, check whether
    existing code can be reused, refactored, or cleaned up instead of duplicated. Remove
    dead code, unused imports, and leftover debug/experimental snippets as you go rather
    than leaving them for a "cleanup phase" that may not happen. Keep the codebase in a
    state you'd be comfortable handing to someone else at any point, not just at the end.

12. **If the project has any UI/visual presentation (plots, demo page, dashboard, GIF
    framing, etc.), make it look simple and human-made — not like default AI-generated
    output.** Concretely this means:
    - No default matplotlib styling with the default blue/orange palette, default fonts,
      or default titles like "Figure 1" — pick a small, deliberate color palette and clean
      labels.
    - Avoid generic AI-tool tells: no emoji-heavy headers, no "🚀 Results" / "✨ Features"
      style sections, no excessive gradients, glassmorphism, or stock dashboard templates
      if a custom HTML page is built.
    - Favor plain, confident, minimal design — whitespace, one or two fonts, restrained
      color use — over "busy and feature-complete looking."
    - The goal is a presentation that looks like a student who cares about craft made it
      by hand, not like it was auto-generated. This applies to README formatting too: avoid
      badge-walls, heavy emoji use, and templated section headers — write it plainly, like
      you're explaining the project to someone, not marketing it.

13. **Keep the repo clean — nothing unnecessary gets added or committed.** Before every
    `git add`, check what's actually being staged (`git status` / `git add -p` over blind
    `git add .`) and keep out: `__pycache__/`, `.pyc` files, virtual env folders, OS files
    (`.DS_Store`), IDE config (`.vscode/`, `.idea/`), trained model checkpoints and large
    result artifacts unless they're a deliberate, small final result, one-off scratch
    scripts, stray `.ipynb_checkpoints/`, duplicate/renamed-leftover files, and any
    experiment that didn't pan out. Keep `.gitignore` accurate and updated from Phase 1
    onward — don't let it fall behind as new file types show up. The repo should only ever
    contain what's actually needed to understand, run, or reproduce the project — nothing
    left over from exploration. This explicitly includes `TEAM_GUIDE.md` and
    `FUTURE_WORK.md` — add both to `.gitignore` the moment either one is created, whenever
    that happens to be, and never stage or push them.

14. **Use parallel agents, skills, and plugins wherever they genuinely speed things up or
    improve quality — and say so when you do.** If independent pieces of work can run
    concurrently (e.g., writing environment tests while scaffolding the agent module, or
    researching deployment options while the baseline trains), use parallel agents for
    them rather than doing everything strictly sequentially. Likewise, reach for any
    installed skill or plugin that fits a task (plotting, deployment, linting, etc.)
    instead of reinventing it from scratch. Every time you use a parallel agent, skill, or
    plugin mid-build, note briefly in `CHECKLIST.md` what was used and why. (This is
    separate from §0.5's one-time pre-build check — that's about whether to *add* new
    ones; this rule is about actually *using* what's available throughout the build.)

15. **Keep deferred/future-work ideas in `FUTURE_WORK.md`, not in this plan.** Anything
    that's a good idea but explicitly out of scope for the current build (multi-intersection
    coordination, more advanced algorithms, a richer UI, etc.) goes in `FUTURE_WORK.md` so
    it's there to revisit later without cluttering this plan or the README. There's no
    deadline forcing any of it — it's fine for `FUTURE_WORK.md` to just sit there until
    there's time.

16. **Be mindful of token/compute usage — don't do unnecessary work for its own sake.**
    There's no deadline pressure here, but that's not license to burn effort doing more
    than the task needs: don't regenerate large outputs (full training runs, large plots,
    long explanations) repeatedly when a smaller check would confirm the same thing, don't
    re-read files that haven't changed, and don't over-explain routine steps. Use parallel
    agents (rule 14) and extra thoroughness where they genuinely add value — tests,
    correctness checks, exploring a real design tradeoff — not as a default for every
    minor step. Where it's unclear whether extra effort is worth it, use judgment rather
    than defaulting to maximum exhaustiveness.

17. **Write/finalize `README.md` only after everything else is complete.** A placeholder
    `README.md` can exist from Phase 1 (title + one-line description is fine), but the real,
    final version — problem statement, architecture summary, results table with
    agent-vs-baseline numbers, learning curve + comparison images, demo GIF, setup/run
    instructions, implemented-vs-future-work section — gets written last, once results,
    plots, and the demo GIF actually exist to put into it. Do not write the final README
    against planned/expected results; write it against what was actually built and measured.

---

## 10. Plan Revisions

Changes from the pre-build compatibility check (2026-10-27). Each overrides the matching
text above.

1. **Deployment target: Streamlit Community Cloud** (fallback: GitHub Pages static page).
   Hugging Face Spaces now requires a paid plan for Gradio/Docker Spaces; only static
   Spaces are free (HF docs, checked 2026-10-27). Streamlit Cloud is free, deploys straight
   from this public GitHub repo, and has native bounded input widgets (§8). Details in
   `docs/architecture.md`.
2. **`simple_env.py` is built first and always exists.** The CI smoke test (§7) and the
   deployed demo both need an environment with no SUMO install, so the custom simulator is
   not optional. The 90-minute SUMO attempt runs afterwards on `feat/sumo-env` as a second
   backend. SUMO is now pip-installable (`eclipse-sumo`), which lowers that risk.
3. **Dependencies live in `pyproject.toml`, pinned, with optional extras** `dev`, `dqn`
   (stable-baselines3 + torch), `sumo` (eclipse-sumo + traci), `app` (streamlit). The
   default Linux torch wheel pulls several GB of CUDA libraries, too heavy for CI and for
   Streamlit Cloud's ~2.7 GB RAM tier. DQN/SUMO code imports these lazily.
4. **Default branch is `main`** (the local unborn branch was `master`).
5. **Repo root is this repository** (`Adaptive-Traffic-Signal-Control/`); it plays the role
   of `traffic-rl-agent/` in §4. `app/` added for the deployed demo.
6. **`sumo_env.py` and `dqn.py` are created when their phase starts**, not as empty
   Phase-1 stubs, since both are conditional.
7. **No `.env.example` yet.** Nothing needs a secret: Streamlit Cloud connects to GitHub
   through its own OAuth and CI uses no credentials. `.env` stays in `.gitignore`; add the
   example file if a secret ever appears.
8. **Rule 10 (no code comments) wins over any tooling habit** of marking shortcuts in code;
   deliberate shortcuts are recorded in `CHECKLIST.md` instead.
9. **SUMO backend uses TraCI directly, not `sumo-rl`** (2026-10-27). `sumo-rl` brings its
   own observation, reward and phase scheme and pulls in pandas and pettingzoo. Driving
   SUMO through TraCI from a subclass of `SimpleTrafficEnv` keeps the spaces, info keys,
   reward and yellow handling identical, so the agents can't tell the backends apart.
   SUMO measures queues as halting vehicles, so absolute wait times differ between
   backends: compare controllers within one backend only.
