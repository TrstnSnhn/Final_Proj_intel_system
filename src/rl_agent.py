from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class QLearningAgent:
    def __init__(self, num_classes: int = 38, alpha: float = 0.1, gamma: float = 0.95):
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros((num_classes, 3), dtype=np.float32)

    def act(self, class_idx: int, eps: float) -> int:
        if random.random() < eps:
            return random.randint(0, 2)
        return int(np.argmax(self.q[class_idx]))

    def update(self, s: int, a: int, r: float, s2: int):
        self.q[s, a] += self.alpha * (r + self.gamma * np.max(self.q[s2]) - self.q[s, a])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    agent = QLearningAgent()
    rewards = []

    for ep in range(args.episodes):
        eps = max(0.1, 1.0 - ep / max(1, args.episodes - 1))
        cum_reward = 0.0
        for cls in range(agent.num_classes):
            a = agent.act(cls, eps)
            r = np.random.normal(loc=0.1 if a == 1 else 0.0, scale=0.1)
            nxt = (cls + 1) % agent.num_classes
            agent.update(cls, a, float(r), nxt)
            cum_reward += float(r)
        rewards.append(cum_reward)

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title("RL Learning Curve (placeholder)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.tight_layout()
    plt.savefig(out / "rl_learning_curve.png", dpi=150)
    print("Saved experiments/results/rl_learning_curve.png")


if __name__ == "__main__":
    main()
