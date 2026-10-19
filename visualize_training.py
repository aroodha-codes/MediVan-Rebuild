"""
visualize_training.py — Q-Learning convergence visualization.

Trains the Q-Learning agent for configurable episodes and plots:
  1. Epsilon decay curve (exploration → exploitation)
  2. Cumulative reward per episode
  3. Q-table heatmap (state × action values)
  4. Policy map (best action per state)

Usage:
    python visualize_training.py              # 500 episodes
    python visualize_training.py --episodes 1000
"""

import argparse
import math
import os
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import JunctionAction
from modules.q_learning_agent import QLearningAgent, N_STATES, N_ACTIONS


def train_agent(n_episodes: int = 500) -> Dict:
    """Train the Q-Learning agent and record metrics per episode.

    Simulates junction encounters with randomized scenarios:
    - Random obstacle presence
    - Random distance to junction
    - Random speed and battery

    Returns dict with training history.
    """
    agent = QLearningAgent()
    # Reset to fresh state
    agent.q_table[:] = 0.0
    agent.epsilon = 1.0
    agent.episode_count = 0
    agent.total_decisions = 0
    agent.total_reward = 0.0

    history = {
        "epsilon": [],
        "episode_reward": [],
        "cumulative_reward": [],
        "q_nonzero": [],
        "q_mean": [],
        "q_max": [],
    }

    cumulative = 0.0

    for ep in range(n_episodes):
        # Random scenario
        junction_dist = np.random.uniform(0, 30)
        obstacle = np.random.random() > 0.5
        speed = np.random.uniform(0.05, 0.4)
        battery = np.random.uniform(15, 100)

        # Agent decides
        action = agent.choose_action(junction_dist, obstacle, speed, battery)

        # Simulate outcome
        collision = (action == JunctionAction.PROCEED and obstacle
                     and np.random.random() < 0.3)
        safely_passed = not collision
        time_spent = np.random.randint(3, 40)

        # Compute reward
        reward = agent.compute_reward(
            action, collision, obstacle, time_spent, safely_passed)

        # Learn
        next_dist = np.random.uniform(0, 30)
        next_obs = np.random.random() > 0.6
        agent.learn(reward, next_dist, next_obs, speed, battery, done=True)

        cumulative += reward

        # Record
        history["epsilon"].append(agent.epsilon)
        history["episode_reward"].append(reward)
        history["cumulative_reward"].append(cumulative)
        history["q_nonzero"].append(int(np.count_nonzero(agent.q_table)))
        history["q_mean"].append(float(np.mean(np.abs(agent.q_table))))
        history["q_max"].append(float(np.max(agent.q_table)))

    return {
        "agent": agent,
        "history": history,
        "n_episodes": n_episodes,
    }


def plot_training(result: Dict, save_dir: str = ".") -> None:
    """Generate 4-panel training visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("[visualize] matplotlib not available. Install: pip install matplotlib")
        return

    history = result["history"]
    agent = result["agent"]
    n_ep = result["n_episodes"]
    episodes = list(range(1, n_ep + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MediVan Q-Learning Training Analysis",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Plot 1: Epsilon Decay ────────────────────
    ax1 = axes[0, 0]
    ax1.plot(episodes, history["epsilon"], color="#2196F3", linewidth=1.5)
    ax1.fill_between(episodes, history["epsilon"], alpha=0.15, color="#2196F3")
    ax1.set_title("Exploration Rate (ε) Decay", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Epsilon (ε)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="ε_min = 0.01")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate key points
    for pct in [0.25, 0.5, 0.75]:
        idx = int(n_ep * pct) - 1
        if 0 <= idx < len(history["epsilon"]):
            ax1.annotate(f'ε={history["epsilon"][idx]:.3f}',
                         xy=(idx + 1, history["epsilon"][idx]),
                         fontsize=8, ha="center",
                         xytext=(0, 15), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", color="gray"))

    # ── Plot 2: Cumulative Reward ────────────────
    ax2 = axes[0, 1]
    ax2.plot(episodes, history["cumulative_reward"],
             color="#4CAF50", linewidth=1.5)
    ax2.fill_between(episodes, history["cumulative_reward"],
                     alpha=0.15, color="#4CAF50")
    ax2.set_title("Cumulative Reward Over Training", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True, alpha=0.3)

    # Rolling average reward
    window = max(10, n_ep // 20)
    if len(history["episode_reward"]) > window:
        rolling = np.convolve(history["episode_reward"],
                              np.ones(window) / window, mode="valid")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(window, len(rolling) + window), rolling,
                      color="#FF9800", linewidth=1.0, alpha=0.7,
                      label=f"Rolling avg ({window} ep)")
        ax2_twin.set_ylabel("Avg Reward/Episode", color="#FF9800")
        ax2_twin.legend(loc="lower right")

    # ── Plot 3: Q-Table Heatmap ──────────────────
    ax3 = axes[1, 0]
    q = agent.q_table.copy()
    action_labels = ["PROCEED", "WAIT", "SLOW", "REROUTE"]

    # Create custom colormap (red-white-blue)
    cmap = LinearSegmentedColormap.from_list(
        "rwb", ["#D32F2F", "#FFFFFF", "#1976D2"])

    vmax = max(abs(q.min()), abs(q.max()), 0.1)
    im = ax3.imshow(q.T, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
    ax3.set_title("Q-Table Heatmap (State × Action)", fontweight="bold")
    ax3.set_xlabel(f"State Index (0-{N_STATES - 1})")
    ax3.set_ylabel("Action")
    ax3.set_yticks(range(N_ACTIONS))
    ax3.set_yticklabels(action_labels)
    fig.colorbar(im, ax=ax3, label="Q-Value", shrink=0.8)

    # ── Plot 4: Learned Policy & Statistics ──────
    ax4 = axes[1, 1]

    # Best action per state
    best_actions = np.argmax(q, axis=1)
    colors_map = {0: "#4CAF50", 1: "#FF9800", 2: "#2196F3", 3: "#9C27B0"}
    bar_colors = [colors_map.get(a, "#888") for a in best_actions]

    ax4.bar(range(N_STATES), best_actions, color=bar_colors, alpha=0.8, width=1.0)
    ax4.set_title("Learned Policy (Best Action per State)", fontweight="bold")
    ax4.set_xlabel(f"State Index (0-{N_STATES - 1})")
    ax4.set_ylabel("Action Index")
    ax4.set_yticks(range(N_ACTIONS))
    ax4.set_yticklabels(action_labels, fontsize=8)
    ax4.set_ylim(-0.5, N_ACTIONS - 0.5)

    # Legend for actions
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors_map[i], label=action_labels[i])
                      for i in range(N_ACTIONS)]
    ax4.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # Stats annotation
    stats = agent.get_stats()
    stats_text = (
        f"Episodes: {stats['episodes']}\n"
        f"Decisions: {stats['decisions']}\n"
        f"ε final: {stats['epsilon']:.4f}\n"
        f"Q nonzero: {stats['q_table_nonzero']}/{N_STATES * N_ACTIONS}\n"
        f"Total reward: {stats['total_reward']:.1f}"
    )
    ax4.text(0.02, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    plot_path = os.path.join(save_dir, "q_learning_training.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[visualize] Training plot saved -> {plot_path}")

    # Also show
    try:
        plt.show(block=False)
        plt.pause(5.0)
    except Exception:
        pass
    plt.close()

    return plot_path


def print_training_summary(result: Dict) -> None:
    """Print a text summary of training results."""
    agent = result["agent"]
    history = result["history"]
    stats = agent.get_stats()

    print("\n" + "=" * 60)
    print("  Q-LEARNING TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Episodes trained:     {stats['episodes']}")
    print(f"  Total decisions:      {stats['decisions']}")
    print(f"  Final epsilon:        {stats['epsilon']:.4f}")
    print(f"  Q-table nonzero:      {stats['q_table_nonzero']}/{N_STATES * N_ACTIONS} "
          f"({stats['q_table_nonzero'] / (N_STATES * N_ACTIONS) * 100:.1f}%)")
    print(f"  Total reward:         {stats['total_reward']:.1f}")
    print(f"  Max Q-value:          {history['q_max'][-1]:.3f}")
    print(f"  Mean |Q|:             {history['q_mean'][-1]:.4f}")

    # Convergence analysis
    eps = history["epsilon"]
    if eps[-1] <= 0.02:
        conv_ep = next(i for i, e in enumerate(eps) if e <= 0.02) + 1
        print(f"\n  Convergence (eps<=0.02): episode {conv_ep}")
    else:
        print(f"\n  Not converged yet (eps={eps[-1]:.3f})")

    # Policy summary
    q = agent.q_table
    best = np.argmax(q, axis=1)
    action_names = ["PROCEED", "WAIT", "SLOW", "REROUTE"]
    counts = {a: 0 for a in action_names}
    for b in best:
        counts[action_names[b]] += 1
    print(f"\n  Policy distribution:")
    for name, count in counts.items():
        bar = "#" * (count * 2)
        print(f"    {name:8s}: {count:3d} states  {bar}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train and visualize Q-Learning convergence.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes (default: 500)")
    args = parser.parse_args()

    print(f"Training Q-Learning agent for {args.episodes} episodes...")
    result = train_agent(args.episodes)

    print_training_summary(result)
    plot_training(result, save_dir=PROJECT_ROOT)

    # Save the trained Q-table
    agent = result["agent"]
    agent.save()
    print(f"\n[visualize] Q-table saved to assets/q_table.npy")


if __name__ == "__main__":
    main()
