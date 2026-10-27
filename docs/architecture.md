# Architecture

## Components

```
configs/*.yaml ──► train.py / evaluate.py ◄── app/streamlit_app.py
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
     Environment      Agent         Baseline
   (Gymnasium Env)  (Q-learning,   (fixed timer)
    simple_env /     SARSA, DQN)
    sumo_env
          │
          ▼
   episode loop ──► metrics CSV ──► plots / GIF / evaluation table
```

| Component | Responsibility | Depends on |
|---|---|---|
| `envs/` | Simulates the intersection behind the Gymnasium API | numpy, gymnasium |
| `agents/` | Learning controllers: discretizer, Q-table, update rule | numpy |
| `baselines/` | Non-learning fixed-timer controller | none |
| `train.py` | Runs the training loop from a config, writes metrics and the model | envs, agents |
| `evaluate.py` | Compares controllers on held-out seeds, writes results and plots | envs, agents, baselines |
| `utils/` | CSV logging, plotting, GIF rendering | matplotlib |
| `app/` | Deployed demo: loads the committed model, runs one bounded episode | core package, streamlit |

Agents and baselines never import an environment module. They see only observations and
actions, so the simulator can be swapped without touching them.

## Deployment decision

**Platform: Streamlit Community Cloud.** Fallback: GitHub Pages.

Free options compared on 2026-10-27, from each provider's own docs:

| Option | Free? | Fit |
|---|---|---|
| Hugging Face Spaces (Gradio/Docker) | No. Creating them now needs a PRO plan; only static Spaces are free | Not usable |
| Hugging Face Spaces (static) | Yes | Results page only, no live simulation |
| Streamlit Community Cloud | Yes: about 2 vCPU and 2.7 GB RAM, sleeps after 12 h without traffic | Interactive Python demo, deploys from this public GitHub repo, redeploys on push |
| GitHub Pages | Yes | Static results page and GIF; no server-side Python |

Streamlit Cloud is the only free option that can run the trained agent live. The app
installs only the core package and Streamlit (no torch, no SUMO), which keeps it well
under the memory limit. If Streamlit Cloud can't be used, the fallback is a static GitHub
Pages page showing the plots, the GIF and the results table.

## CI/CD

- **CI:** GitHub Actions (`.github/workflows/ci.yml`) runs ruff, mypy, pytest and a short
  smoke training run on every push. It has read-only permissions and no secrets.
- **CD:** Streamlit Cloud redeploys from `main` on every push.
- **Dependabot:** monthly dependency update PRs.
