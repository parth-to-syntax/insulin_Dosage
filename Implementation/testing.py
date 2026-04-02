import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from simulation import InsulinEnv, plot_episode

# ── load Q-table ──────────────────────────────
with open('qtable.pkl', 'rb') as f:
    Q = pickle.load(f)


def safety_cap_action(action, state):
    g_bin, d_bin = state[0], state[1]
    if g_bin == 0:                  # <70
        return 0
    if g_bin == 1 and d_bin == 0:   # 70-140 and falling
        return min(action, 1)
    if g_bin == 1 and d_bin == 1:   # 70-140 and stable
        return min(action, 2)
    return action

# ── policy runner ─────────────────────────────
def run_episode(policy, patient_name, seed, max_steps=288):
    env   = InsulinEnv(patient_name=patient_name, seed=seed)
    state = env.reset()
    done  = False; step = 0
    while not done and step < max_steps:
        if   policy == 'q_agent': action = int(np.argmax(Q[state]))
        elif policy == 'fixed':   action = 3 if state[4] in [1,2] else 0  # 6U morning/afternoon
        elif policy == 'random':  action = np.random.randint(env.n_actions)
        action = safety_cap_action(action, state)
        state, _, done = env.step(action)
        step += 1
    return env

# ── evaluate across patients + seeds ─────────
test_patients = ['adult#001', 'adult#003', 'adult#007', 'adolescent#003', 'child#002']
policies      = ['q_agent', 'fixed', 'random']
N_SEEDS       = 15
rows = []

for policy in policies:
    for patient in test_patients:
        for seed in range(N_SEEDS):
            env = run_episode(policy, patient, seed)
            rows.append({
                'policy':  policy,
                'patient': patient,
                'tir':     env.time_in_range(),
                'reward':  sum(env.reward_history),
                'mean_bg': np.mean(env.glucose_history),
                'hypo':    sum(1 for g in env.glucose_history if g < 70),
                'hyper':   sum(1 for g in env.glucose_history if g > 140),
            })
    print(f"{policy} done")

results = pd.DataFrame(rows)

# ── summary table ─────────────────────────────
summary = results.groupby('policy').agg(
    TIR    = ('tir',     'mean'),
    Reward = ('reward',  'mean'),
    MeanBG = ('mean_bg', 'mean'),
    Hypos  = ('hypo',    'mean'),
    Hypers = ('hyper',   'mean'),
).round(2)
summary_std = results.groupby('policy').agg(
    TIR    = ('tir',     'std'),
    Reward = ('reward',  'std'),
    MeanBG = ('mean_bg', 'std'),
    Hypos  = ('hypo',    'std'),
    Hypers = ('hyper',   'std'),
).round(2)

summary_pm = pd.DataFrame(index=summary.index)
for col in summary.columns:
    summary_pm[col] = summary[col].map(lambda x: f"{x:.2f}") + " ± " + summary_std[col].map(lambda x: f"{x:.2f}")

print("\n── Test Results ──────────────────────────")
print("(mean ± std)")
print(summary_pm.to_string())

# ── bar chart comparison ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics = [('tir', 'Time in Range (%)'), ('hypo', 'Hypo Events'), ('reward', 'Total Reward')]
colors  = ['steelblue', 'tomato', 'green']

for ax, (metric, label), color in zip(axes, metrics, colors):
    means = results.groupby('policy')[metric].mean()
    stds  = results.groupby('policy')[metric].std()
    bars  = ax.bar(means.index, means.values, yerr=stds.values,
                   color=color, alpha=0.75, capsize=5)
    ax.set_title(label); ax.set_ylabel(label); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds.values.mean()*0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Policy Comparison — Test Patients', fontsize=12)
plt.tight_layout()
plt.savefig('policy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── visualize one full episode per policy ─────
for policy in policies:
    env = run_episode(policy, 'adult#003', seed=99)
    plot_episode(env, title=f'{policy} — adult#003')