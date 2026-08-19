"""Tabular Q-learning: the classical-RL foil to the LLM search stack.

A self-contained demo — a Q-table over a small synthetic search environment,
with no connection to the GRPO/PPO machinery next door. It lived under the RL
package until that package was renamed `grpo`, at which point it was the
largest thing in there that was not GRPO.

Pure Python and numpy: no torch, no transformers, nothing that a CI job without
heavy ML packages cannot import. Keep it that way — `examples/run_rl_search_demo.py`
is meant to run anywhere.
"""

from .agent import QLearningAgent as QLearningAgent
from .environment import SearchEnvironment as SearchEnvironment

__all__ = ["QLearningAgent", "SearchEnvironment"]
