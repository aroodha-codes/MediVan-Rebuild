"""
q_learning_agent.py -- Tabular Q-Learning agent for junction decisions.

Replaces hardcoded if/else junction logic with a trained reinforcement
learning agent.  The agent learns optimal decisions at hospital
corridor junctions (proceed, wait, slow down, or reroute) based on
the current environment state.

RPi4 efficiency:
  - Q-table is a 54x4 numpy array (~1.7 KB in memory)
  - Action selection is a single numpy argmax = < 1 microsecond
  - Trains online during simulation, saves/loads as .npy file
  - No neural network, no GPU, no framework overhead
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    Q_TABLE_PATH,
    Q_LEARNING_RATE, Q_DISCOUNT_FACTOR,
    Q_EPSILON_START, Q_EPSILON_DECAY, Q_EPSILON_MIN,
    JunctionAction,
)

# State discretization dimensions
_DIST_LEVELS = 3       # NEAR=0, MEDIUM=1, FAR=2
_OBS_LEVELS = 2        # NO=0, YES=1
_SPEED_LEVELS = 3      # SLOW=0, MEDIUM=1, FAST=2
_BATTERY_LEVELS = 3    # LOW=0, MEDIUM=1, HIGH=2

N_STATES: int = _DIST_LEVELS * _OBS_LEVELS * _SPEED_LEVELS * _BATTERY_LEVELS  # 54
N_ACTIONS: int = len(JunctionAction)  # 4


class QLearningAgent:
    """Tabular Q-Learning agent for junction navigation decisions.

    State space (54 states):
      - junction_distance: NEAR(<10px) / MEDIUM(10-20px) / FAR(>20px)
      - obstacle_nearby:   NO / YES
      - speed_level:       SLOW(<0.1m/s) / MEDIUM(0.1-0.3) / FAST(>0.3)
      - battery_level:     LOW(<30%) / MEDIUM(30-70%) / HIGH(>70%)

    Action space (4 actions):
      - PROCEED: continue at current speed through junction
      - WAIT:    full stop, wait for obstacle to clear
      - SLOW:    reduce to 30% PWM through junction
      - REROUTE: trigger A* re-plan to avoid junction

    The agent trains online using epsilon-greedy exploration during
    simulation and saves the learned Q-table for deployment.
    """

    def __init__(self) -> None:
        self.q_table: np.ndarray = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.epsilon: float = Q_EPSILON_START
        self.learning_rate: float = Q_LEARNING_RATE
        self.discount: float = Q_DISCOUNT_FACTOR
        self.episode_count: int = 0
        self.total_decisions: int = 0
        self.total_reward: float = 0.0

        # Last state-action for delayed reward
        self._last_state: Optional[int] = None
        self._last_action: Optional[int] = None

        # Try to load existing Q-table
        self._try_load()

    # -- public API -----------------------------------------

    def choose_action(
        self,
        junction_dist_px: float,
        obstacle_nearby: bool,
        speed_ms: float,
        battery_pct: float,
    ) -> JunctionAction:
        """Select an action for the current junction state.

        Uses epsilon-greedy: with probability epsilon, explore randomly;
        otherwise exploit the learned Q-table.

        Parameters
        ----------
        junction_dist_px : float
            Distance to junction in map pixels.
        obstacle_nearby : bool
            True if any obstacle detected near the junction.
        speed_ms : float
            Current vehicle speed in m/s.
        battery_pct : float
            Current battery percentage.

        Returns
        -------
        JunctionAction
            The chosen action (PROCEED / WAIT / SLOW / REROUTE).
        """
        state = self._discretize(junction_dist_px, obstacle_nearby,
                                  speed_ms, battery_pct)
        self._last_state = state

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(N_ACTIONS)
        else:
            action_idx = int(np.argmax(self.q_table[state]))

        self._last_action = action_idx
        self.total_decisions += 1

        return JunctionAction(action_idx)

    def learn(
        self,
        reward: float,
        next_junction_dist_px: float,
        next_obstacle_nearby: bool,
        next_speed_ms: float,
        next_battery_pct: float,
        done: bool = False,
    ) -> None:
        """Update Q-table using the Bellman equation.

        Called after the action outcome is observed.

        Parameters
        ----------
        reward : float
            Reward signal for the last action.
        next_* : float / bool
            State after taking the action.
        done : bool
            True if episode is over (junction fully passed).
        """
        if self._last_state is None or self._last_action is None:
            return

        s = self._last_state
        a = self._last_action
        next_s = self._discretize(next_junction_dist_px, next_obstacle_nearby,
                                   next_speed_ms, next_battery_pct)

        # Q-Learning update: Q(s,a) += lr * [r + gamma * max(Q(s')) - Q(s,a)]
        if done:
            target = reward
        else:
            target = reward + self.discount * float(np.max(self.q_table[next_s]))

        self.q_table[s, a] += self.learning_rate * (target - self.q_table[s, a])

        self.total_reward += reward

        # Decay epsilon
        if done:
            self.episode_count += 1
            self.epsilon = max(
                Q_EPSILON_MIN,
                self.epsilon * Q_EPSILON_DECAY,
            )

    def compute_reward(
        self,
        action: JunctionAction,
        collision: bool,
        obstacle_present: bool,
        time_spent_frames: int,
        safely_passed: bool,
    ) -> float:
        """Compute the reward for a junction encounter.

        Parameters
        ----------
        action : JunctionAction taken.
        collision : True if a collision occurred.
        obstacle_present : True if obstacle was actually present.
        time_spent_frames : Frames spent at junction.
        safely_passed : True if junction was cleared without incident.

        Returns
        -------
        float : Reward value.
        """
        reward = 0.0

        if collision:
            reward -= 5.0
            return reward

        if safely_passed:
            reward += 10.0
            # Efficiency bonus: less time = better
            if time_spent_frames < 15:
                reward += 5.0
            elif time_spent_frames < 30:
                reward += 2.0

        # Correct yielding: waited when obstacle was present
        if action == JunctionAction.WAIT and obstacle_present:
            reward += 3.0
        # Unnecessary waiting: waited when no obstacle
        elif action == JunctionAction.WAIT and not obstacle_present:
            reward -= 2.0

        # Time penalty for waiting
        reward -= time_spent_frames * (1.0 / 30.0)

        return reward

    def save(self, path: Optional[str] = None) -> None:
        """Save Q-table to a .npy file."""
        save_path = path or os.path.join(
            os.path.dirname(__file__), "..", Q_TABLE_PATH
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.q_table)
        print(f"[QLearning] Q-table saved -> {save_path}  "
              f"(episodes={self.episode_count}, eps={self.epsilon:.3f})")

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            "episodes": self.episode_count,
            "decisions": self.total_decisions,
            "epsilon": round(self.epsilon, 4),
            "total_reward": round(self.total_reward, 2),
            "q_table_nonzero": int(np.count_nonzero(self.q_table)),
            "q_table_max": round(float(np.max(self.q_table)), 2),
        }

    # -- internals ------------------------------------------

    def _try_load(self) -> None:
        """Load an existing Q-table if available."""
        paths_to_try = [
            os.path.join(os.path.dirname(__file__), "..", Q_TABLE_PATH),
            Q_TABLE_PATH,
        ]
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    loaded = np.load(path)
                    if loaded.shape == (N_STATES, N_ACTIONS):
                        self.q_table = loaded.astype(np.float64)
                        # Start with low epsilon (exploit learned policy)
                        self.epsilon = Q_EPSILON_MIN
                        print(f"[QLearning] Q-table loaded from {path}")
                        return
                except Exception as e:
                    print(f"[QLearning] Failed to load Q-table: {e}")
                    break

        print("[QLearning] No Q-table found -- starting fresh training")

    def _discretize(
        self,
        junction_dist_px: float,
        obstacle_nearby: bool,
        speed_ms: float,
        battery_pct: float,
    ) -> int:
        """Convert continuous state to discrete state index.

        State index = d * (2 * 3 * 3) + o * (3 * 3) + s * 3 + b
        """
        # Distance: NEAR=0, MEDIUM=1, FAR=2
        if junction_dist_px < 10:
            d = 0
        elif junction_dist_px < 20:
            d = 1
        else:
            d = 2

        # Obstacle: NO=0, YES=1
        o = 1 if obstacle_nearby else 0

        # Speed: SLOW=0, MEDIUM=1, FAST=2
        if speed_ms < 0.10:
            s = 0
        elif speed_ms < 0.30:
            s = 1
        else:
            s = 2

        # Battery: LOW=0, MEDIUM=1, HIGH=2
        if battery_pct < 30.0:
            b = 0
        elif battery_pct < 70.0:
            b = 1
        else:
            b = 2

        state = d * (_OBS_LEVELS * _SPEED_LEVELS * _BATTERY_LEVELS) + \
                o * (_SPEED_LEVELS * _BATTERY_LEVELS) + \
                s * _BATTERY_LEVELS + \
                b

        return min(state, N_STATES - 1)


# -- Standalone test ----------------------------------------
if __name__ == "__main__":
    agent = QLearningAgent()
    print(f"Q-table shape: {agent.q_table.shape}")
    print(f"States: {N_STATES}, Actions: {N_ACTIONS}")

    # Simulate 100 junction encounters
    for episode in range(100):
        dist = np.random.uniform(0, 30)
        obs = np.random.random() > 0.7
        speed = np.random.uniform(0, 0.4)
        battery = np.random.uniform(10, 100)

        action = agent.choose_action(dist, obs, speed, battery)

        # Simulate outcome
        collision = (action == JunctionAction.PROCEED and obs
                     and np.random.random() < 0.3)
        safely = not collision

        reward = agent.compute_reward(
            action, collision, obs,
            time_spent_frames=np.random.randint(5, 60),
            safely_passed=safely,
        )

        agent.learn(reward, dist + 10, obs, speed, battery, done=True)

    stats = agent.get_stats()
    print(f"\nTraining stats: {stats}")
    print(f"Epsilon: {agent.epsilon:.4f}")
    print(f"Non-zero Q entries: {stats['q_table_nonzero']} / {N_STATES * N_ACTIONS}")

    # Test action for a specific state
    a = agent.choose_action(5.0, True, 0.2, 80.0)
    print(f"\nNear junction, obstacle, medium speed, high battery -> {a.name}")

    a = agent.choose_action(5.0, False, 0.3, 50.0)
    print(f"Near junction, no obstacle, fast, medium battery -> {a.name}")

    agent.save()
