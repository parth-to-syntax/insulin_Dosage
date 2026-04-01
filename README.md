# Reinforcement Learning for Insulin Dosage Regulation

This repository contains a Reinforcement Learning (RL) approach to regulating insulin delivery for Type 1 Diabetes treatment. The project uses the [`simglucose`](https://github.com/jxx123/simglucose) simulator and OpenAI Gym environments to train an agent using Q-Learning.

## Overview

The main goal of the RL agent is to maximize the **Time In Range (TIR)** (maintaining blood glucose levels between 70 and 140 mg/dL) while severely penalizing hypoglycemia (low blood glucose) and hyperglycemia (high blood glucose). 

The repository consists of three main components:
*   `simulation.py`: Sets up the `T1DSimEnv` from `simglucose`, defining the rewards, the action spaces (insulin doses), and discretizing the observation state (glucose level, rate of change, previous meals, etc.).
*   `training.py`: Runs a Q-learning algorithm for 3,000 episodes on a simulated adult patient, outputting the learned policy into a Q-table (`qtable.pkl`).
*   `testing.py`: Tests the learned Q-agent policy across multiple different patients (`adolescent` and `child` profiles) with varying random scenarios. It directly compares the agent's performance against two baseline policies: a `fixed` clinical dosage script and a `random` baseline dosage.

## Requirements & Setup

Because `simglucose` relies on older Gym and Numpy packages with outdated compilation setups, **Python 3.11** is heavily recommended (Python 3.12 may fail to build certain numpy and setuptools wheels).

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv_311

# Activate the virtual environment
source .venv_311/bin/activate

# Optional: downgrade setuptools to fix `pkg_resources` missing errors:
pip install "setuptools<70.0.0"

# Install requirements
pip install -r requirement.txt
```

## Running the Code

### 1. Training the Agent

Run the `training.py` code to build the `qtable.pkl`. Over 3,000 episodes, the script will output live statistics (TIR, Reward, Epsilon) and eventually generate two visualizations: `training_curves.png` and `qtable_heatmap.png`.

```bash
MPLBACKEND=Agg python training.py
```
*(Note: `MPLBACKEND=Agg` is used here to ensure Matplotlib silently saves the resulting images without opening interactive GUI popup windows down the pipeline).* 

### 2. Testing the Models

Once training has ended and `qtable.pkl` was successfully stored, you can execute the testing sequence.

```bash
MPLBACKEND=Agg python testing.py
```

The sequence will parse all patients and provide an evaluation table logging the **Time in Range (TIR)**, mean rewards, and number of Hypo/Hyper events recorded against fixed and random dosing rules. It also produces `policy_comparison.png`.

## Code Structure

*   **State Space**: Described by `_get_state` inside `simulation.py`, discretizing five dimensions: *glucose level, glucose delta, meal sizes, lag, and time-of-day*.
*   **Actions**: Evaluated doses ranging from `0` to `8` units. 
*   **Reward Function**: Calculated via `get_reward()`, granting significant positive rewards for staying in normal targets (+10), while aggressively penalizing hypo incidents (<70 outputs -25 and <54 outputs -50).
