import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from simulation import InsulinEnv, plot_episode

# ── hyperparameters ──────────────────────────
EPISODES      = 1500
MAX_STEPS     = 144
ALPHA         = 0.1
GAMMA         = 0.9
EPSILON       = 1.0
EPSILON_DECAY = 0.997
EPSILON_MIN   = 0.05

# ── convergence controls ─────────────────────
MIN_EPISODES      = 600
CHECK_WINDOW      = 100
EARLY_STOP_PATIENCE = 3
TIR_IMPROVE_THR   = 0.10
REWARD_IMPROVE_THR = 0.8

# ── Q-table ──────────────────────────────────
Q = np.zeros((4, 3, 3, 3, 4, 6))   # glucose x delta x meal x lag x tod x action

# ── training loop ────────────────────────────
episode_rewards, episode_tir = [], []
epsilon = EPSILON
env     = InsulinEnv(patient_name='adult#001')

best_window_tir = -np.inf
best_window_reward = -np.inf
no_improve_count = 0
trained_episodes = 0

for ep in range(EPISODES):
    env.seed = ep
    state    = env.reset()
    done     = False; step = 0; total_r = 0

    while not done and step < MAX_STEPS:
        if np.random.rand() < epsilon:
            # Safer exploration: no aggressive bolus outside meal windows.
            if env.glucose_history[-1] < 90:
                action = 0
            elif not env.in_meal_window(env.current_hour()):
                action = np.random.randint(0, 4)
            else:
                action = np.random.randint(env.n_actions)
        else:
            action = int(np.argmax(Q[state]))               # exploit

        next_state, reward, done = env.step(action)

        # bellman update
        Q[state][action] += ALPHA * (
            reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
        )
        state = next_state; total_r += reward; step += 1

    episode_rewards.append(total_r)
    episode_tir.append(env.time_in_range())
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    trained_episodes = ep + 1

    if (ep + 1) % 300 == 0:
        print(f"Ep {ep+1:4d} | Reward: {np.mean(episode_rewards[-300:]):7.1f} "
              f"| TIR: {np.mean(episode_tir[-300:]):.1f}% | ε: {epsilon:.3f}")

    if (ep + 1) >= MIN_EPISODES and (ep + 1) % CHECK_WINDOW == 0:
        window_tir = float(np.mean(episode_tir[-CHECK_WINDOW:]))
        window_reward = float(np.mean(episode_rewards[-CHECK_WINDOW:]))

        tir_improved = window_tir > (best_window_tir + TIR_IMPROVE_THR)
        reward_improved = window_reward > (best_window_reward + REWARD_IMPROVE_THR)

        if tir_improved or reward_improved:
            best_window_tir = max(best_window_tir, window_tir)
            best_window_reward = max(best_window_reward, window_reward)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOP_PATIENCE and epsilon <= 0.15:
            print(
                f"Early stop at episode {ep+1} | "
                f"Window TIR: {window_tir:.2f}% | Window Reward: {window_reward:.2f}"
            )
            break

# ── save ─────────────────────────────────────
with open('qtable.pkl', 'wb') as f:
    pickle.dump(Q, f)
print(f"Q-table saved (episodes trained: {trained_episodes})")

# ── training curves ───────────────────────────
def smooth(arr, w=50):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w)/w, mode='valid')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax1.plot(smooth(episode_rewards), color='steelblue', linewidth=1.5)
ax1.set_ylabel('Total Reward'); ax1.set_title('Training Curves'); ax1.grid(alpha=0.3)
ax2.plot(smooth(episode_tir), color='green', linewidth=1.5)
ax2.axhline(70, color='gray', linestyle='--', linewidth=1, label='Clinical target 70%')
ax2.set_ylabel('TIR (%)'); ax2.set_xlabel('Episode')
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Q-table heatmap (slice: normal glucose, morning) ─────
q_slice = np.array([[np.max(Q[1, d, m, 1, 1]) for d in range(3)] for m in range(3)])
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(q_slice, annot=True, fmt='.1f', cmap='Blues', ax=ax,
            xticklabels=['Falling','Stable','Rising'],
            yticklabels=['No meal','Moderate','Heavy'])
ax.set_xlabel('Rate of change'); ax.set_ylabel('Meal size')
ax.set_title('Q-values — Normal glucose, Morning')
plt.tight_layout()
plt.savefig('qtable_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ── learned policy printout ───────────────────
labels = ['Low(<70)','Normal(70-140)','High(140-180)','Critical(>180)']
print(f"\n{'Glucose':<22} {'Best dose':>10}")
print("-" * 34)
for g in range(4):
    best = int(np.argmax(Q[g, 1, 0, 1, 1]))   # stable, no meal, medium lag, morning
    print(f"{labels[g]:<22} {InsulinEnv.ACTIONS[best]}U")