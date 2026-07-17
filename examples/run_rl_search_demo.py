"""Train and evaluate the classical-RL (tabular Q-learning) search demo.

Usage:
    python3 -m examples.run_rl_search_demo --episodes 2000
    python3 -m examples.run_rl_search_demo --stochastic
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from src.training.rl_agent import QLearningAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    agent = QLearningAgent()
    print(f"Training for {args.episodes} episodes (stochastic={args.stochastic})...\n")
    train_stats = agent.train(
        num_episodes=args.episodes, verbose=True, stochastic=args.stochastic
    )
    print("Training summary:")
    print(f"  Victory rate: {train_stats['victory_rate']:.2%}")
    print(f"  Q-table size: {train_stats['q_table_size']}\n")

    eval_stats = agent.evaluate(
        num_episodes=args.eval_episodes, stochastic=args.stochastic
    )
    print("Evaluation (greedy):")
    print(f"  Victory rate: {eval_stats['victory_rate']:.2%}")
    print(f"  Avg reward:   {eval_stats['avg_reward']:.2f}")
    print(f"  Avg length:   {eval_stats['avg_length']:.2f}")


if __name__ == "__main__":
    main()
