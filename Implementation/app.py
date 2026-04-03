import os
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from simulation import InsulinEnv


st.set_page_config(page_title="RL Insulin Demo", layout="wide")


@st.cache_resource
def load_q_table(path: str = "qtable.pkl"):
    base_dir = Path(__file__).resolve().parent
    candidates = [
        Path(path),
        base_dir / path,
        base_dir / "Result" / "qtable.pkl",
        base_dir.parent / "Result" / "qtable.pkl",
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "rb") as f:
                return pickle.load(f)

    return None


def select_action(policy: str, state, env: InsulinEnv, q_table):
    if policy == "q_agent":
        if q_table is None:
            raise FileNotFoundError(
                "qtable.pkl not found. Run training.py first to use q_agent policy."
            )
        action = int(np.argmax(q_table[state]))
    elif policy == "fixed":
        # 3U morning/afternoon, else 0U
        action = 3 if state[4] in [1, 2] else 0
    elif policy == "random":
        action = np.random.randint(env.n_actions)
    else:
        action = np.random.randint(env.n_actions)

    g_bin = state[0]
    current_glucose = env.glucose_history[-1]
    hour = env.current_hour()

    if current_glucose < 90 or g_bin == 0:
        return 0
    if not env.in_meal_window(hour) and action > 3:
        return 3
    return action


def run_episode(policy: str, patient: str, seed: int, max_steps: int, live: bool, speed: float):
    q_table = load_q_table()
    env = InsulinEnv(patient_name=patient, seed=seed)
    state = env.reset()

    chart_slot = st.empty()
    status_slot = st.empty()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    done = False
    step = 0

    while not done and step < max_steps:
        action = select_action(policy, state, env, q_table)
        state, reward, done = env.step(action)
        step += 1

        if live and (step % 2 == 0 or done):
            glucose = np.array(env.glucose_history)
            doses = np.array(env.dose_history)
            t = np.arange(len(glucose)) * 5 / 60

            ax1.clear()
            ax2.clear()

            ax1.fill_between(t, 70, 140, alpha=0.12, color="green")
            ax1.axhline(70, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax1.axhline(140, color="orange", linestyle="--", linewidth=1, alpha=0.7)
            ax1.plot(t, glucose, color="steelblue", linewidth=2, label="CGM")
            ax1.set_ylabel("Glucose (mg/dL)")
            ax1.set_ylim(40, 320)
            ax1.grid(alpha=0.3)
            ax1.legend(loc="upper right")

            ax2.bar(t, doses, width=0.06, color="steelblue", alpha=0.7)
            ax2.set_ylabel("Dose (U)")
            ax2.set_xlabel("Time (hours)")
            ax2.grid(alpha=0.3)

            chart_slot.pyplot(fig, clear_figure=False)
            status_slot.info(
                f"Step: {step}/{max_steps} | Current BG: {glucose[-1]:.1f}"
            )
            time.sleep(speed)

    plt.close(fig)
    return env


def summarize_env(env: InsulinEnv):
    glucose = np.array(env.glucose_history)
    return {
        "TIR": float(env.time_in_range()),
        "Reward": float(sum(env.reward_history)),
        "MeanBG": float(np.mean(glucose)),
        "Hypos": int(np.sum(glucose < 70)),
        "Hypers": int(np.sum(glucose > 140)),
    }


def render_static_plot(env: InsulinEnv):
    glucose = np.array(env.glucose_history)
    doses = np.array(env.dose_history)
    t = np.arange(len(glucose)) * 5 / 60

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.fill_between(t, 70, 140, alpha=0.12, color="green")
    ax1.axhline(70, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axhline(140, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax1.plot(t, glucose, color="steelblue", linewidth=2, label="CGM")
    hypo = glucose < 70
    if hypo.any():
        ax1.scatter(t[hypo], glucose[hypo], color="red", s=20, zorder=5, label="Hypo")

    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_ylim(40, 320)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.bar(t, doses, width=0.06, color="steelblue", alpha=0.7)
    ax2.set_ylabel("Dose (U)")
    ax2.set_xlabel("Time (hours)")
    ax2.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


st.title("Reinforcement Learning for Insulin Dosing")
st.caption("Interactive simulation demo for semester presentation")

with st.sidebar:
    st.header("Simulation Settings")
    patient = st.selectbox(
        "Patient",
        ["adult#001", "adult#003", "adult#007", "adolescent#003", "child#002"],
        index=0,
    )
    policy = st.selectbox("Policy", ["q_agent", "fixed", "random"], index=0)
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
    max_steps = st.slider("Max steps", min_value=48, max_value=144, value=144, step=12)
    live_mode = st.toggle("Live playback", value=True)
    speed = st.slider("Playback delay (seconds)", min_value=0.0, max_value=0.2, value=0.02, step=0.01)

    run_btn = st.button("Run simulation", type="primary")

if run_btn:
    try:
        env = run_episode(policy, patient, int(seed), int(max_steps), bool(live_mode), float(speed))
        metrics = summarize_env(env)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TIR (%)", f"{metrics['TIR']:.1f}")
        c2.metric("Total Reward", f"{metrics['Reward']:.1f}")
        c3.metric("Mean BG", f"{metrics['MeanBG']:.1f}")
        c4.metric("Hypos", f"{metrics['Hypos']}")
        c5.metric("Hypers", f"{metrics['Hypers']}")

        st.subheader("Final Episode Plot")
        render_static_plot(env)

        glucose = np.array(env.glucose_history)
        doses = np.array(env.dose_history)
        df = pd.DataFrame(
            {
                "step": np.arange(len(glucose)),
                "time_h": np.arange(len(glucose)) * 5 / 60,
                "glucose": glucose,
                "dose": doses,
            }
        )
        st.download_button(
            "Download episode CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"episode_{policy}_{patient}_{seed}.csv",
            mime="text/csv",
        )

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run training.py first to generate qtable.pkl, or choose fixed/random policy.")
    except Exception as e:
        st.error(f"Simulation failed: {e}")
else:
    st.info("Choose settings in the sidebar and click 'Run simulation'.")
