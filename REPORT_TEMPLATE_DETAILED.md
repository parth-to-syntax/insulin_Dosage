# Reinforcement Learning for Insulin Dosing

## 1. Project Title
Safety-Aware Tabular Reinforcement Learning for Insulin Dosing in a Type-1 Diabetes Simulator

## 2. Executive Summary
This project builds a complete simulation-first RL pipeline for insulin dose recommendation in Type-1 Diabetes. The solution is based on tabular Q-learning and runs on top of a physiological simulator instead of a static CSV dataset. The codebase includes:
- a custom environment wrapper with state discretization, reward shaping, and safety constraints,
- a training pipeline with epsilon-greedy exploration and early stopping,
- a multi-patient evaluation pipeline against fixed and random baselines,
- and a Streamlit interface for interactive demonstration.

The central goal is to improve glycemic control while avoiding unsafe insulin actions.

## 3. Why This Project Matters
Insulin dosing is a sequential decision problem with delayed effects and safety-critical outcomes. A dose that looks good now can create hypoglycemia later. Traditional static prediction setups do not model this closed-loop control behavior well. RL is appropriate because it learns a policy that optimizes long-term return over trajectories, not one-step predictions.

## 4. Codebase Overview
Core project files:
- Environment and plotting: [Implementation/simulation.py](Implementation/simulation.py)
- Training: [Implementation/training.py](Implementation/training.py)
- Testing and policy benchmarking: [Implementation/testing.py](Implementation/testing.py)
- Interactive demo UI: [Implementation/app.py](Implementation/app.py)

Execution flow:
1. Environment emits a discretized state from simulator signals.
2. Agent selects an insulin action from a finite action set.
3. Safety logic post-processes unsafe actions.
4. Simulator advances by one step and returns next observation.
5. Reward is computed and Q-table is updated (during training).
6. Policy is evaluated on unseen seeds and multiple virtual patients.

## 5. Technology Stack and Rationale
| Technology | Role | Why it is used |
|---|---|---|
| Python | Primary language | Fast iteration for RL + simulation workflows |
| simglucose | T1D physiological simulator | Generates realistic glucose trajectories without patient data collection |
| NumPy | Q-table and numerical ops | Efficient tensor indexing and vectorized statistics |
| Pandas | Evaluation aggregation | Convenient policy-level summary tables |
| Matplotlib | Episode/training/test plots | Clear clinical-style trajectory visualization |
| Seaborn | Q-value heatmap | Better visual interpretation of learned value patterns |
| Streamlit | Live demo UI | Fast, presentation-ready interactive front-end |

## 6. Environment Design (Detailed)

### 6.1 Simulator Construction
The class `InsulinEnv` wraps `T1DSimEnv` from simglucose with:
- virtual patient (`T1DPatient`),
- CGM sensor (`CGMSensor`),
- insulin pump (`InsulinPump`),
- random scenario (`RandomScenario`) with a controlled seed.

Each step uses a 5-minute control interval, and each episode is capped at 12 hours:
- `DT_MIN = 5`
- `TOTAL_HOURS = 12`
- `TOTAL_STEPS = 144`

### 6.2 State Space (Discretized)
State is a 5-tuple:
1. Glucose bin (`GLUCOSE_BINS = [70, 140, 180]`) -> 4 bins
2. Glucose delta/trend (`DELTA_BINS = [-2, 2]`) -> 3 bins
3. Meal-context bin (`MEAL_BINS = [20, 60]`) -> 3 bins
4. Dose-lag bin (`LAG_BINS = [1, 3]`) -> 3 bins
5. Time-of-day bin (`TOD_BINS = [6, 12, 18]`) -> 4 bins

Total tabular state cardinality = $4 \times 3 \times 3 \times 3 \times 4 = 432$ states.

### 6.3 Action Space
Discrete insulin doses (units):
- `[0, 1, 2, 3, 5, 8]`

This gives 6 actions per state.

### 6.4 Safety Layer
Safety is enforced in environment stepping (and mirrored in testing/UI policy wrappers):
- If current glucose < 90 mg/dL, force action to 0U.
- Outside meal windows (2-3h and 8-9h), aggressive actions are capped: actions above index 3 are clamped to index 3 (3U).

This design preserves learnability while reducing high-risk dosing behavior.

### 6.5 Episode Termination
Episode ends when either:
- glucose drops below 50 mg/dL (critical low), or
- horizon reaches 144 steps (12 hours).

## 7. Reward Function (Current Code)
Implemented reward in [Implementation/simulation.py](Implementation/simulation.py):
- +3.0 if glucose in [70, 140]
- -8.0 if glucose < 70
- -4.0 if glucose > 180
- -1.0 otherwise
- insulin cost penalty: `-0.05 * dose`

Interpretation:
- Strongly favors in-range glycemia.
- Penalizes hypoglycemia more than hyperglycemia.
- Adds mild control effort regularization to discourage unnecessary high doses.

## 8. RL Algorithm and Learning Setup

### 8.1 Algorithm
Tabular Q-learning with Bellman update:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right]
$$

### 8.2 Hyperparameters (Current)
From [Implementation/training.py](Implementation/training.py):
- `ALPHA = 0.1`
- `GAMMA = 0.9`
- `EPISODES = 1500` (reduced for faster turnaround)
- `MAX_STEPS = 144`
- `EPSILON = 1.0`
- `EPSILON_DECAY = 0.997`
- `EPSILON_MIN = 0.05`

Q-table shape:
- `(4, 3, 3, 3, 4, 6)` corresponding to state bins x action count.

### 8.3 Exploration Policy
Exploration is epsilon-greedy with safety-aware random action sampling:
- if glucose < 90 -> choose 0U
- outside meal windows -> sample conservative actions `[0..3]`
- within meal windows -> sample full action set `[0..5]`

### 8.4 Early Stopping
Training uses a convergence gate:
- `MIN_EPISODES = 600`
- `CHECK_WINDOW = 100`
- `EARLY_STOP_PATIENCE = 3`
- `TIR_IMPROVE_THR = 0.10`
- `REWARD_IMPROVE_THR = 0.8`

Stopping occurs when both rolling TIR/reward improvements stall and epsilon is sufficiently small.

## 9. Evaluation Protocol

### 9.1 Policies Compared
From [Implementation/testing.py](Implementation/testing.py):
- `q_agent`: learned policy from Q-table
- `fixed`: rule baseline (3U in morning/afternoon bins, else 0U)
- `random`: random policy

### 9.2 Population and Seeds
- Patients: `adult#001`, `adult#003`, `adult#007`, `adolescent#003`, `child#002`
- Seeds per patient per policy: 15

### 9.3 Metrics Reported
- Time in Range (70-140)
- Total reward
- Mean blood glucose
- Hypoglycemia events (`glucose < 70`)
- Hyperglycemia events (`glucose > 140`)

All metrics are aggregated as mean ± std.

## 10. Streamlit Demo Design
The demo in [Implementation/app.py](Implementation/app.py) provides:
- policy selection (`q_agent`, `fixed`, `random`),
- patient and seed controls,
- live playback with adjustable delay,
- KPI cards for TIR/reward/mean BG/hypo/hyper,
- final episode plot,
- CSV export of trajectory.

It also includes robust `qtable.pkl` lookup in multiple candidate directories to support project folder organization.

## 11. Why This Is Not a Dataset Project
No static training dataset is required because the simulator generates trajectories online. Each episode produces a new glucose-control sequence given policy actions and scenario dynamics. This is a control/optimization loop, not a one-shot supervised prediction task.

## 12. Current Strengths
- End-to-end reproducible RL workflow.
- Clear and interpretable tabular policy.
- Safety-aware action post-processing.
- Multi-patient, multi-seed benchmarking.
- Presentation-ready live UI.

## 13. Current Limitations
- Tabular discretization may lose fine temporal/physiological detail.
- Reward engineering still influences behavior strongly.
- Simulator-to-reality transfer is not guaranteed.
- Training runtime and stability are sensitive to exploration/reward settings.

## 14. Suggested Future Improvements
- Move from tabular Q-learning to DQN/Safe-RL.
- Add explicit trend-risk penalties (rapid downward drift).
- Evaluate additional clinical metrics (TBR/TAR, risk indices).
- Add patient-specific adaptation.
- Include policy explainability panel in Streamlit.

## 15. Reproducibility Commands
Run from repository root:

1. Activate environment
- `source .venv_311/bin/activate`

2. Install dependencies
- `python -m pip install -r Implementation/requirement.txt`

3. Train policy
- `cd Implementation`
- `python training.py`

4. Evaluate policy
- `python testing.py`

5. Launch UI
- `streamlit run app.py`

6. Move artifacts (optional)
- `setopt null_glob`
- `mv -- *.png *.pkl *.csv ../Result/`

## 16. Results Table Template
| Policy | TIR (mean ± std) | Reward (mean ± std) | Mean BG (mean ± std) | Hypos (mean ± std) | Hypers (mean ± std) |
|---|---|---|---|---|---|
| q_agent |  |  |  |  |  |
| fixed |  |  |  |  |  |
| random |  |  |  |  |  |

## 17. Conclusion
This project demonstrates a practical, safety-aware RL control pipeline for insulin dosing in simulation. The implementation is modular, reproducible, and presentation-ready. While tabular RL has modeling limits, the current system is a strong foundation for future safe-RL and deep-RL upgrades.
