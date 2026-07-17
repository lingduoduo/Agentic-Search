# Classical-RL Search Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recast the sampled treasure-hunt `rl_agent.py` into a self-contained, search-shaped tabular Q-learning demo living in `src/training/`.

**Architecture:** A new synthetic `SearchEnvironment` (topics-hold-facts, questions-need-fact-subsets) replaces the missing `game_environment.TreasureHuntGame`. The existing `QLearningAgent` is retargeted to it (new state hash, fixed victory heuristic) and its dead `DQNAgent` placeholder is deleted. A runnable example trains + evaluates; unit tests cover env dynamics and the learning loop.

**Tech Stack:** Python, `numpy`, stdlib `random`/`pickle`/`collections`, `pytest`, `ruff`.

## Global Constraints

- Relative imports within `src/training/` (e.g. `from .search_environment import SearchEnvironment`); start new modules with `from __future__ import annotations`.
- Runnable entrypoints live in `examples/` and run as `python3 -m examples.<name>`.
- Three environment actions only: `retrieve:<topic>`, `answer`, `stop`. No `rewrite`.
- No serving / GRPO / PPO integration; no neural networks; no reading `data/corpus.jsonl`.
- No `game_environment` import anywhere; no `DQNAgent` after this work.
- `ruff check .` and `ruff format .` must be clean; pre-commit runs ruff on commit.
- Reward scale (verbatim): retrieve relevant+new `+2` with `-1` step cost (net `+1`); retrieve irrelevant/dup `-1`; `answer` correct `+50` (`victory=True`); `answer` early `-10`; `stop` `-5`; step-budget exceeded `-10`. All terminal outcomes set `game_over`.

---

### Task 1: `SearchEnvironment` — world + reset + action listing

**Files:**
- Create: `src/training/search_environment.py`
- Test: `tests/unit/test_rl_search_agent.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class SearchEnvironment(stochastic: bool = False, max_steps: int = 20, seed: int | None = None)`
  - Attributes: `topics: dict[str, str]` (topic id → fact token), `questions: list[dict]` (each `{"id": str, "required_facts": set[str], "answer": str}`), `current_question: str`, `gathered: set[str]`, `steps: int`, `game_over: bool`, `victory: bool`, `max_steps: int`.
  - `reset() -> None` — round-robin selects next question, clears `gathered`, resets `steps`/`game_over`/`victory`.
  - `get_available_actions() -> list[str]` — `[f"retrieve:{t}" for t in topics]` + `["answer", "stop"]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_rl_search_agent.py`:

```python
from __future__ import annotations

import random

import numpy as np
import pytest

from src.training.search_environment import SearchEnvironment


def _env(**kw):
    kw.setdefault("seed", 0)
    return SearchEnvironment(**kw)


def test_reset_selects_a_known_question_and_clears_state():
    env = _env()
    env.reset()
    qids = {q["id"] for q in env.questions}
    assert env.current_question in qids
    assert env.gathered == set()
    assert env.steps == 0
    assert env.game_over is False
    assert env.victory is False


def test_reset_round_robins_over_questions():
    env = _env()
    seen = []
    for _ in range(len(env.questions) + 1):
        env.reset()
        seen.append(env.current_question)
    # First question repeats after a full cycle.
    assert seen[0] == seen[len(env.questions)]
    assert len(set(seen[: len(env.questions)])) == len(env.questions)


def test_available_actions_cover_all_topics_plus_answer_stop():
    env = _env()
    env.reset()
    actions = env.get_available_actions()
    for topic in env.topics:
        assert f"retrieve:{topic}" in actions
    assert "answer" in actions
    assert "stop" in actions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_rl_search_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.search_environment'`

- [ ] **Step 3: Write minimal implementation**

Create `src/training/search_environment.py`:

```python
"""A small synthetic search environment for the classical-RL (Q-learning) demo.

Replaces the treasure-hunt world the sampled ``rl_agent.py`` was written for.
Topics each hold one fact; each question needs a subset of facts. The agent
retrieves facts and must answer once it holds every required fact.
"""

from __future__ import annotations

import random


TOPICS: dict[str, str] = {
    "faiss": "faiss is a dense vector index",
    "bm25": "bm25 is a sparse lexical ranker",
    "rrf": "rrf fuses ranked lists",
    "e5": "e5 is a dense embedding model",
    "chunking": "chunking splits documents",
    "rerank": "a cross-encoder reranks candidates",
}

QUESTIONS: list[dict] = [
    {"id": "what_is_faiss", "required_facts": {"faiss"}, "answer": "faiss"},
    {"id": "dense_vs_sparse", "required_facts": {"faiss", "bm25"}, "answer": "hybrid"},
    {"id": "hybrid_pipeline", "required_facts": {"bm25", "e5", "rrf"}, "answer": "rrf"},
    {"id": "rerank_stage", "required_facts": {"chunking", "rerank"}, "answer": "rerank"},
]


class SearchEnvironment:
    """Synthetic retrieve-until-sufficient environment.

    Actions: ``retrieve:<topic>`` for each topic, plus ``answer`` and ``stop``.
    """

    def __init__(
        self,
        stochastic: bool = False,
        max_steps: int = 20,
        seed: int | None = None,
        fail_prob: float = 0.2,
    ):
        self.stochastic = stochastic
        self.max_steps = max_steps
        self.fail_prob = fail_prob
        self._rng = random.Random(seed)

        self.topics = dict(TOPICS)
        self.questions = [
            {"id": q["id"], "required_facts": set(q["required_facts"]), "answer": q["answer"]}
            for q in QUESTIONS
        ]

        self._question_idx = 0
        self.current_question = self.questions[0]["id"]
        self.gathered: set[str] = set()
        self.steps = 0
        self.game_over = False
        self.victory = False

    def _current(self) -> dict:
        return next(q for q in self.questions if q["id"] == self.current_question)

    def reset(self) -> None:
        self.current_question = self.questions[self._question_idx]["id"]
        self._question_idx = (self._question_idx + 1) % len(self.questions)
        self.gathered = set()
        self.steps = 0
        self.game_over = False
        self.victory = False

    def get_available_actions(self) -> list[str]:
        return [f"retrieve:{t}" for t in self.topics] + ["answer", "stop"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_rl_search_agent.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/training/search_environment.py tests/unit/test_rl_search_agent.py
git commit -m "feat: add synthetic SearchEnvironment world, reset, and action listing"
```

---

### Task 2: `SearchEnvironment.execute_action` — dynamics + rewards

**Files:**
- Modify: `src/training/search_environment.py`
- Test: `tests/unit/test_rl_search_agent.py`

**Interfaces:**
- Consumes: `SearchEnvironment` from Task 1.
- Produces: `execute_action(action: str) -> tuple[str, float, bool]` returning `(feedback, reward, done)`; sets `game_over`/`victory`/`gathered`/`steps` per the reward table.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_rl_search_agent.py`:

```python
def _reset_to(env, question_id):
    """Reset until the given question is active (round-robin makes this deterministic)."""
    for _ in range(len(env.questions)):
        env.reset()
        if env.current_question == question_id:
            return
    raise AssertionError(f"question {question_id} not found")


def test_retrieve_relevant_new_fact_nets_positive_and_gathers():
    env = _env()
    _reset_to(env, "what_is_faiss")  # requires {"faiss"}
    feedback, reward, done = env.execute_action("retrieve:faiss")
    assert "faiss" in env.gathered
    assert reward == 1  # +2 relevance, -1 step cost
    assert done is False


def test_retrieve_irrelevant_costs_one_and_gathers_nothing():
    env = _env()
    _reset_to(env, "what_is_faiss")
    feedback, reward, done = env.execute_action("retrieve:bm25")
    assert env.gathered == set()
    assert reward == -1
    assert done is False


def test_retrieve_duplicate_costs_one():
    env = _env()
    _reset_to(env, "what_is_faiss")
    env.execute_action("retrieve:faiss")
    _, reward, _ = env.execute_action("retrieve:faiss")
    assert reward == -1


def test_answer_with_all_facts_wins():
    env = _env()
    _reset_to(env, "what_is_faiss")
    env.execute_action("retrieve:faiss")
    feedback, reward, done = env.execute_action("answer")
    assert reward == 50
    assert env.victory is True
    assert done is True
    assert env.game_over is True


def test_answer_early_penalized_and_terminal():
    env = _env()
    _reset_to(env, "dense_vs_sparse")  # requires {"faiss", "bm25"}
    env.execute_action("retrieve:faiss")
    feedback, reward, done = env.execute_action("answer")
    assert reward == -10
    assert env.victory is False
    assert done is True


def test_stop_penalized_and_terminal():
    env = _env()
    _reset_to(env, "what_is_faiss")
    _, reward, done = env.execute_action("stop")
    assert reward == -5
    assert done is True
    assert env.game_over is True


def test_step_budget_terminates_with_penalty():
    env = _env(max_steps=3)
    _reset_to(env, "hybrid_pipeline")  # requires 3 facts, unreachable in 3 wasted steps
    rewards = []
    done = False
    for _ in range(3):
        _, reward, done = env.execute_action("retrieve:faiss")  # irrelevant here
        rewards.append(reward)
    assert done is True
    assert rewards[-1] == -10  # budget-exceeded penalty replaces the step reward
    assert env.game_over is True


def test_stochastic_relevant_retrieval_can_fail():
    env = _env(stochastic=True, fail_prob=1.0, seed=0)
    _reset_to(env, "what_is_faiss")
    _, reward, _ = env.execute_action("retrieve:faiss")
    assert env.gathered == set()  # forced failure
    assert reward == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_rl_search_agent.py -k "retrieve or answer or stop or step_budget or stochastic" -v`
Expected: FAIL — `AttributeError: 'SearchEnvironment' object has no attribute 'execute_action'`

- [ ] **Step 3: Write minimal implementation**

Add to `SearchEnvironment` in `src/training/search_environment.py`:

```python
    def execute_action(self, action: str) -> tuple[str, float, bool]:
        if self.game_over:
            return ("episode over", 0.0, True)

        self.steps += 1
        required = self._current()["required_facts"]

        # Budget check first: an over-budget step terminates with a fixed penalty.
        if self.steps >= self.max_steps and action not in ("answer", "stop"):
            self.game_over = True
            return ("out of steps", -10.0, True)

        if action == "answer":
            self.game_over = True
            if required.issubset(self.gathered):
                self.victory = True
                return ("correct answer", 50.0, True)
            return ("answered too early", -10.0, True)

        if action == "stop":
            self.game_over = True
            return ("gave up", -5.0, True)

        if action.startswith("retrieve:"):
            topic = action.split(":", 1)[1]
            relevant_new = topic in required and topic not in self.gathered
            if relevant_new:
                if self.stochastic and self._rng.random() < self.fail_prob:
                    return (f"retrieve {topic} failed", -1.0, False)
                self.gathered.add(topic)
                return (f"retrieved {topic}", 1.0, False)
            return (f"retrieve {topic} wasted", -1.0, False)

        return ("unknown action", -1.0, False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_rl_search_agent.py -v`
Expected: PASS (all env tests)

- [ ] **Step 5: Commit**

```bash
git add src/training/search_environment.py tests/unit/test_rl_search_agent.py
git commit -m "feat: add SearchEnvironment.execute_action dynamics and rewards"
```

---

### Task 3: Retarget `QLearningAgent` to `SearchEnvironment`; delete `DQNAgent`

**Files:**
- Modify: `src/training/rl_agent.py`
- Test: `tests/unit/test_rl_search_agent.py`

**Interfaces:**
- Consumes: `SearchEnvironment` (Tasks 1–2).
- Produces: `QLearningAgent` whose methods take `SearchEnvironment`; `_get_state_hash(env) -> str` keyed on `(current_question, frozenset(gathered), steps_remaining_bucket)`; `train_episode`, `train`, `evaluate`, `save`, `load` unchanged in behavior. `DQNAgent` removed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_rl_search_agent.py`:

```python
from src.training.rl_agent import QLearningAgent


def test_dqn_agent_is_gone():
    import src.training.rl_agent as mod

    assert not hasattr(mod, "DQNAgent")


def test_state_hash_is_stable_and_reflects_gathered():
    env = _env()
    _reset_to(env, "what_is_faiss")
    agent = QLearningAgent()
    h0 = agent._get_state_hash(env)
    assert agent._get_state_hash(env) == h0  # deterministic
    env.execute_action("retrieve:faiss")
    assert agent._get_state_hash(env) != h0  # gathering changes the state


def test_update_q_value_matches_bellman():
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)
    agent.q_table["s'"]["a1"] = 10.0
    agent.q_table["s'"]["a2"] = 4.0
    # current Q(s,a) defaults to 0.0
    agent.update_q_value("s", "a", reward=2.0, next_state="s'",
                         next_actions=["a1", "a2"], done=False)
    # target = 2 + 0.9*10 = 11 ; Q <- 0 + 0.5*(11-0) = 5.5
    assert agent.q_table["s"]["a"] == pytest.approx(5.5)


def test_update_q_value_terminal_uses_reward_only():
    agent = QLearningAgent(learning_rate=1.0, discount_factor=0.9)
    agent.update_q_value("s", "a", reward=7.0, next_state="s'",
                         next_actions=[], done=True)
    assert agent.q_table["s"]["a"] == pytest.approx(7.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_rl_search_agent.py -k "dqn or state_hash or bellman or terminal" -v`
Expected: FAIL — import error on `game_environment` (module missing), so the whole file errors at collection.

- [ ] **Step 3: Write minimal implementation**

Edit `src/training/rl_agent.py`. Replace the module docstring + imports (lines 1–11) with:

```python
"""Tabular Q-learning agent for the synthetic search environment.

The classical-RL foil to this repo's LLM-based search stack: it learns a
retrieve-until-sufficient policy on ``SearchEnvironment`` and needs many
episodes of training to do so.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from .search_environment import SearchEnvironment
```

Replace the `_get_state_hash` method body with a search-shaped hash:

```python
    def _get_state_hash(self, game: SearchEnvironment) -> str:
        """Small, discrete state key for tabular Q-learning."""
        steps_remaining = max(0, game.max_steps - game.steps)
        # Coarse bucket keeps the state space small.
        remaining_bucket = min(steps_remaining, 5)
        state_parts = (
            game.current_question,
            tuple(sorted(game.gathered)),
            remaining_bucket,
        )
        return str(state_parts)
```

Replace every remaining `TreasureHuntGame` type hint (in `choose_action`,
`train_episode`, `train`, `evaluate` signatures) with `SearchEnvironment`.

Replace the constructor of the game in `train` and `evaluate`:
`game = TreasureHuntGame(stochastic=stochastic)` → `game = SearchEnvironment(stochastic=stochastic)`.

In `train`, fix the victory heuristic. The current per-100 block computes
`recent_victories` from `reward > 50`; track true victories instead. Change the
verbose block to:

```python
            if verbose and (episode + 1) % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_victories = sum(self.recent_victory_flags[-100:])
                avg_reward = np.mean(recent_rewards)

                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Victories (last 100): {recent_victories}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Q-table size: {len(self.q_table)}")
                print()
```

Add `self.recent_victory_flags: list[int] = []` to `__init__` (next to the other
statistics), and in `train_episode`, right after `if game.victory:` bookkeeping,
append the flag:

```python
        self.recent_victory_flags.append(1 if game.victory else 0)
```

Delete the entire `DQNAgent` class (from `class DQNAgent:` to end of file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_rl_search_agent.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/training/rl_agent.py tests/unit/test_rl_search_agent.py
git commit -m "feat: retarget QLearningAgent to SearchEnvironment, drop DQN placeholder"
```

---

### Task 4: Learning + save/load integration tests

**Files:**
- Test: `tests/unit/test_rl_search_agent.py`

**Interfaces:**
- Consumes: `QLearningAgent`, `SearchEnvironment` (Tasks 1–3).
- Produces: nothing (tests only).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_rl_search_agent.py`:

```python
def test_epsilon_decays_toward_min():
    random.seed(0)
    np.random.seed(0)
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1)
    env = SearchEnvironment(seed=0)
    start = agent.epsilon
    for _ in range(50):
        agent.train_episode(env)
    assert agent.epsilon < start
    assert agent.epsilon >= agent.epsilon_min


def test_agent_learns_single_fact_question():
    random.seed(0)
    np.random.seed(0)
    # Environment with only the trivial one-fact question, so learning is fast.
    env = SearchEnvironment(seed=0)
    env.questions = [{"id": "what_is_faiss", "required_facts": {"faiss"}, "answer": "faiss"}]
    env._question_idx = 0
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05)
    for _ in range(400):
        agent.train_episode(env)
    result = agent.evaluate(num_episodes=50, stochastic=False)
    # evaluate() builds its own env (all questions); assert on the shared trained
    # policy instead by running greedy episodes on the single-fact env:
    wins = 0
    agent.epsilon = 0.0
    for _ in range(50):
        env.reset()
        while not env.game_over:
            action = agent.choose_action(env, training=False)
            env.execute_action(action)
        wins += int(env.victory)
    assert wins / 50 > 0.9


def test_save_and_load_round_trips(tmp_path):
    agent = QLearningAgent()
    agent.q_table["s"]["a"] = 3.14
    agent.total_episodes = 5
    agent.victories = 2
    path = tmp_path / "qtable.pkl"
    agent.save(str(path))

    fresh = QLearningAgent()
    fresh.load(str(path))
    assert fresh.q_table["s"]["a"] == pytest.approx(3.14)
    assert fresh.total_episodes == 5
    assert fresh.victories == 2
```

Note on `test_agent_learns_single_fact_question`: the `evaluate()` call is
exercised for coverage but the assertion uses greedy rollouts on the single-fact
env because `evaluate()` constructs its own full-question env. If the learning
threshold proves flaky under the seed, raise the episode count to 800 rather than
lowering the 0.9 bar.

- [ ] **Step 2: Run tests to verify they fail then pass**

Run: `pytest tests/unit/test_rl_search_agent.py -k "epsilon or learns or round_trips" -v`
Expected: PASS (implementation already exists from Task 3). If `learns` fails on
the seed, bump episodes to 800 as noted.

- [ ] **Step 3: Run the full test file**

Run: `pytest tests/unit/test_rl_search_agent.py -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_rl_search_agent.py
git commit -m "test: cover Q-learning epsilon decay, learning, and save/load"
```

---

### Task 5: Runnable example entrypoint

**Files:**
- Create: `examples/run_rl_search_demo.py`

**Interfaces:**
- Consumes: `QLearningAgent`, `SearchEnvironment`.
- Produces: `python3 -m examples.run_rl_search_demo` CLI.

- [ ] **Step 1: Write the implementation**

Create `examples/run_rl_search_demo.py`:

```python
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
```

- [ ] **Step 2: Run the demo end-to-end**

Run: `python3 -m examples.run_rl_search_demo --episodes 2000 --seed 0`
Expected: prints rising per-100 victory counts during training and a final
evaluation victory rate well above 0.5 (deterministic mode). No traceback.

- [ ] **Step 3: Commit**

```bash
git add examples/run_rl_search_demo.py
git commit -m "feat: add runnable classical-RL search demo entrypoint"
```

---

### Task 6: Export symbols + lint

**Files:**
- Modify: `src/training/__init__.py`

**Interfaces:**
- Consumes: `QLearningAgent`, `SearchEnvironment`.
- Produces: package-level re-exports.

- [ ] **Step 1: Add exports**

In `src/training/__init__.py`, inside the existing `try:` import block, add:

```python
    from .rl_agent import QLearningAgent as QLearningAgent
    from .search_environment import SearchEnvironment as SearchEnvironment
```

- [ ] **Step 2: Verify the import surface**

Run: `python3 -c "from src.training import QLearningAgent, SearchEnvironment; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Lint the whole change**

Run: `ruff check . --fix && ruff format .`
Expected: no errors; formatting clean.

- [ ] **Step 4: Full test sweep for the new file**

Run: `pytest tests/unit/test_rl_search_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/__init__.py
git commit -m "feat: export QLearningAgent and SearchEnvironment from src.training"
```

---

## Self-Review

**Spec coverage:**
- `SearchEnvironment` world/reset/actions → Task 1. ✓
- `execute_action` reward table + stochastic mode → Task 2. ✓
- `QLearningAgent` import fix, state hash, victory-heuristic fix, `DQNAgent` deletion → Task 3. ✓
- Learning / epsilon / save-load tests → Task 4. ✓
- `examples/run_rl_search_demo.py` with `--episodes/--eval-episodes/--stochastic/--seed` → Task 5. ✓
- Exports via `__init__.py` → Task 6. ✓
- Non-goals (no serving/GRPO, no neural nets, no corpus.jsonl, no `rewrite`) → enforced by Global Constraints; no task violates them. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. The one
conditional ("bump episodes to 800") is a concrete fallback with exact values, not a placeholder.

**Type consistency:** `_get_state_hash(env)`, `execute_action -> (feedback, reward, done)`,
`update_q_value(state, action, reward, next_state, next_actions, done)`,
`recent_victory_flags` all used consistently across Tasks 2–4. ✓
