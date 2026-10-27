import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.baselines.fixed_timer import FixedTimer, tuned_timer
from traffic_rl.envs.simple_env import DEFAULT_DEMAND, Demand, SimpleEnvConfig, SimpleTrafficEnv
from traffic_rl.episode import Controller, run_episode

MODEL_PATH = ROOT / "results" / "models" / "q_learning.npz"
TUNED_LABEL, TUNED_TIMER = tuned_timer(ROOT / "results" / "evaluation.csv")
CONTROLLERS = {
    "Q-learning": lambda: QLearningAgent.load(MODEL_PATH),
    TUNED_LABEL: lambda: TUNED_TIMER,
    "Fixed timer 30/30 s": lambda: FixedTimer(30, 30),
}
MAX_RATE = 0.25
MAX_SEED = 999


@st.cache_data(max_entries=64)
def simulate(demand: Demand, seed: int) -> tuple[dict[str, dict[str, float]], dict[str, list]]:
    metrics = {}
    chart: dict[str, list] = {"Minute": [], "Vehicles waiting": [], "Controller": []}
    for name, build in CONTROLLERS.items():
        env = SimpleTrafficEnv(SimpleEnvConfig(demand=demand))
        controller: Controller = build()

        def record(*_: object, env: SimpleTrafficEnv = env, name: str = name) -> None:
            chart["Minute"].append(env.sim_time / 60)
            chart["Vehicles waiting"].append(int(env.queues.sum()))
            chart["Controller"].append(name)

        metrics[name] = run_episode(env, controller, seed, on_transition=record)
    return metrics, chart


st.set_page_config(page_title="Adaptive traffic signal demo")
st.title("Adaptive traffic signal control")
st.write(
    "One simulated hour at a four-way intersection, run three times with the same random "
    "arrivals: once with the trained Q-learning agent, once with the fixed timer tuned on "
    f"training traffic ({TUNED_TIMER.green_ns}/{TUNED_TIMER.green_ew} s of green), and once "
    "with a default 30/30 s timer. Set the arrival rate for each direction in each "
    "20-minute window below. The agent was trained only on the default demand, so settings "
    "far from the defaults are outside what it has seen, and it may do worse than a fixed "
    "timer there."
)

with st.form("demand"):
    windows = []
    start = 0
    for end, rates in DEFAULT_DEMAND:
        st.caption(f"Minutes {start // 60} to {int(end) // 60}, vehicles per second")
        ns_col, ew_col = st.columns(2)
        ns = ns_col.slider("North-south", 0.0, MAX_RATE, rates[0], 0.01, key=f"ns_{end}")
        ew = ew_col.slider("East-west", 0.0, MAX_RATE, rates[2], 0.01, key=f"ew_{end}")
        windows.append((end, ns, ew))
        start = int(end)
    seed = st.number_input("Random seed", 0, MAX_SEED, 0, 1)
    st.form_submit_button("Run")


def bounded(rate: float) -> float:
    return min(max(float(rate), 0.0), MAX_RATE)


demand: Demand = tuple(
    (float(end), (bounded(ns), bounded(ns), bounded(ew), bounded(ew))) for end, ns, ew in windows
)
metrics, chart = simulate(demand, min(max(int(seed), 0), MAX_SEED))

for column, (name, result) in zip(st.columns(len(metrics)), metrics.items(), strict=True):
    column.subheader(name)
    column.metric("Average wait (s)", f"{result['avg_wait']:.1f}")
    column.metric("Average queue (vehicles)", f"{result['avg_queue']:.1f}")
    column.metric("Throughput (vehicles)", f"{result['throughput']:.0f}")
    column.metric("Phase switches", f"{result['switches']:.0f}")

st.subheader("Vehicles waiting over the hour")
st.line_chart(chart, x="Minute", y="Vehicles waiting", color="Controller")
