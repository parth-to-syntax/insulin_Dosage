import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
from datetime import datetime


class InsulinEnv:
    ACTIONS      = [0, 1, 2, 3, 5, 8]
    GLUCOSE_BINS = [70, 140, 180]
    DELTA_BINS   = [-2, 2]
    MEAL_BINS    = [20, 60]
    LAG_BINS     = [1, 3]
    TOD_BINS     = [6, 12, 18]
    DT_MIN       = 5
    TOTAL_HOURS  = 12
    TOTAL_STEPS  = int(TOTAL_HOURS * 60 / DT_MIN)

    def __init__(self, patient_name='adult#001', seed=42):
        self.patient_name    = patient_name
        self.seed            = seed
        self.n_actions       = len(self.ACTIONS)
        self.glucose_history = []
        self.dose_history    = []
        self.reward_history  = []
        self.step_count      = 0
        self.last_dose_step  = -10
        self.last_meal       = 0
        self.prev_glucose    = None
        self._build_env()

    def _build_env(self):
        patient  = T1DPatient.withName(self.patient_name)
        sensor   = CGMSensor.withName('Dexcom', seed=self.seed)
        pump     = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=datetime(2024,1,1,0,0), seed=self.seed)
        self.env = T1DSimEnv(patient, sensor, pump, scenario)

    @staticmethod
    def _digitize(val, bins):
        for i, b in enumerate(bins):
            if val < b: return i
        return len(bins)

    def _get_state(self, glucose, meal=None):
        delta      = (glucose - self.prev_glucose) if self.prev_glucose else 0.0
        meal_carbs = meal if meal is not None else self.last_meal
        dose_lag   = (self.step_count - self.last_dose_step) * 5 / 60
        tod        = (self.step_count * 5 / 60) % 24
        return (
            self._digitize(glucose,    self.GLUCOSE_BINS),
            self._digitize(delta,      self.DELTA_BINS),
            self._digitize(meal_carbs, self.MEAL_BINS),
            self._digitize(dose_lag,   self.LAG_BINS),
            self._digitize(tod,        self.TOD_BINS),
        )

    @staticmethod
    def get_reward(glucose, dose):
        if 70 <= glucose <= 140:
            reward = +3.0
        elif glucose < 70:
            reward = -8.0
        elif glucose > 180:
            reward = -4.0
        else:
            reward = -1.0

        reward -= 0.05 * dose
        return reward

    def current_hour(self):
        return (self.step_count * self.DT_MIN) / 60.0

    @staticmethod
    def in_meal_window(hour):
        return (2.0 <= hour <= 3.0) or (8.0 <= hour <= 9.0)

    def reset(self):
        self._build_env()
        obs, _, _, _ = self.env.step(Action(basal=0, bolus=0))
        glucose = max(40.0, float(obs.CGM))
        self.glucose_history = [glucose]
        self.dose_history    = [0]
        self.reward_history  = []
        self.step_count      = 0
        self.last_dose_step  = -10
        self.last_meal       = 0
        self.prev_glucose    = None
        return self._get_state(glucose)

    def step(self, action_idx):
        action_idx = int(np.clip(action_idx, 0, self.n_actions - 1))
        hour = self.current_hour()
        current_glucose = self.glucose_history[-1]

        # Always block insulin when already low.
        if current_glucose < 90:
            action_idx = 0
        # Outside meal windows, cap to conservative doses (0-3U).
        elif not self.in_meal_window(hour) and action_idx > 3:
            action_idx = 3

        dose = self.ACTIONS[action_idx]
        if dose > 0:
            self.last_dose_step = self.step_count
        obs, _, done, _ = self.env.step(Action(basal=0.0, bolus=dose/60.0))

        glucose = max(40.0, float(obs.CGM))

        meal = 60 if self.in_meal_window(hour) else 0
        self.last_meal    = meal
        self.prev_glucose = self.glucose_history[-1]
        reward = self.get_reward(glucose, dose)
        self.glucose_history.append(glucose)
        self.dose_history.append(dose)
        self.reward_history.append(reward)
        self.step_count += 1
        if glucose < 50: done = True
        if self.step_count >= self.TOTAL_STEPS: done = True
        return self._get_state(glucose, meal), reward, done

    def time_in_range(self):
        arr = np.array(self.glucose_history)
        return 100 * np.mean((arr >= 70) & (arr <= 140))


def plot_episode(env, title='Episode'):
    glucose = np.array(env.glucose_history)
    doses   = np.array(env.dose_history)
    time_h  = np.arange(len(glucose)) * 5 / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.fill_between(time_h, 70, 140, alpha=0.12, color='green')
    ax1.axhline(70,  color='red',    linestyle='--', linewidth=1, alpha=0.7)
    ax1.axhline(140, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax1.plot(time_h, glucose, color='steelblue', linewidth=2, label='CGM')
    hypo = glucose < 70
    if hypo.any():
        ax1.scatter(time_h[hypo], glucose[hypo], color='red', s=40, zorder=5, label='Hypo')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title(f'{title}  |  TIR: {env.time_in_range():.1f}%  |  Reward: {sum(env.reward_history):.0f}')
    ax1.legend(fontsize=9); ax1.set_ylim(40, 320); ax1.grid(alpha=0.3)
    ax2.bar(time_h, doses, width=0.06, color='steelblue', alpha=0.7)
    ax2.set_ylabel('Dose (U)'); ax2.set_xlabel('Time (hours)')
    ax2.set_yticks([0,2,4,6,8]); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ","_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()


# --- run random agent to verify env works ---
if __name__ == '__main__':
    env   = InsulinEnv(patient_name='adult#001', seed=42)
    state = env.reset()
    done  = False; step = 0
    while not done and step < env.TOTAL_STEPS:
        state, reward, done = env.step(np.random.randint(env.n_actions))
        step += 1
    print(f"TIR    : {env.time_in_range():.1f}%")
    print(f"Reward : {sum(env.reward_history):.1f}")
    print(f"Hypos  : {sum(1 for g in env.glucose_history if g < 70)}")
    plot_episode(env, title='Random Agent Baseline')