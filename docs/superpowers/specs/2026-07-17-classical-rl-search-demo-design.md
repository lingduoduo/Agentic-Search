# Classical-RL Search Demo — Design

**Date:** 2026-07-17
**Branch:** `feat/classical-rl-search-demo`
**Status:** Approved (brainstorming)

## Motivation

A sampled `rl_agent.py` (tabular Q-learning + a DQN placeholder) was dropped into
`src/training/`. It was written for a **treasure-hunt game** and imports
`from game_environment import TreasureHuntGame`, a module that does not exist in
this repo — so the file does not even import. Its docstring frames it as
*"the classical RL approach that requires extensive training."*

We adapt it into a self-contained, **search-shaped** classical-RL demo: the
classical-RL foil to this repo's LLM-based search stack. It trains in seconds,
touches no serving code, and does not compete with the existing GRPO/PPO action
policy.

### Why a self-contained toy (and not a real search controller)

Tabular Q-learning needs a *small, discrete* state space and *cheap, fast*
episodes. Wiring it into the real agentic-search loop fails on both counts:

- Real state is query text + document contents — continuous and
  high-dimensional; bucketing it discards the one thing that matters (is this
  doc relevant to this query?).
- Each real episode runs retrieval + an LLM answer + a judge — minutes per
  episode locally, and can crash the dev servers. "Extensive training" of a
  tabular learner against that is infeasible here.
- That control decision is already solved better in-repo by the GRPO action
  policy; a tabular learner would be a weaker, redundant second controller.

A tabular Q-learner is a toy in this domain by nature. The design is honest
about that: it teaches the retrieve-until-sufficient policy on a toy
search environment, mirroring how the original sample baked its world into code.

## Scope

### In scope
- New `src/training/search_environment.py` — `SearchEnvironment`.
- Adapt existing `src/training/rl_agent.py` — `QLearningAgent` retargeted to
  `SearchEnvironment`; delete the `DQNAgent` placeholder.
- New `examples/run_rl_search_demo.py` — runnable train + evaluate entrypoint.
- New `tests/unit/test_rl_search_agent.py`.
- Export `QLearningAgent` and `SearchEnvironment` from `src/training/__init__.py`.

### Non-goals
- No serving / GRPO / PPO integration.
- No neural networks (the `DQNAgent` placeholder is removed).
- No reading of `data/corpus.jsonl` — the environment is fully synthetic.
- No `rewrite` action — three actions only (`retrieve` / `answer` / `stop`).

## Components

### 1. `src/training/search_environment.py` — `SearchEnvironment`

Replaces the missing `game_environment.TreasureHuntGame`. The world is baked
into code, mirroring the treasure-hunt's self-contained design.

**World**
- `topics`: ~6 topic ids, each mapping to exactly one fact token.
- `questions`: ~4 questions, each with a `required_facts` subset (1–3 topics)
  and an `answer` id.

**Constructor**
```python
SearchEnvironment(stochastic: bool = False, max_steps: int = 20, seed: int | None = None)
```
`stochastic` enables retrieval failures (see below). `max_steps` bounds an
episode. `seed` is optional for reproducibility.

**State attributes** (read by the agent's state hash)
- `current_question`: the active question id (chosen on `reset()`).
- `gathered`: `set[str]` of facts collected so far.
- `steps`: steps taken this episode.
- `game_over`: bool.
- `victory`: bool (set True only on a correct `answer`).

**Interface** (matches what `QLearningAgent` already calls)
- `reset()` → picks the next question (round-robin over `questions` for
  determinism; index advances each reset), clears `gathered`, resets counters.
- `get_available_actions() -> list[str]` → `["retrieve:<topic>" for each topic]`
  plus `["answer", "stop"]`. Available actions do not depend on hidden state, so
  the agent sees a stable action set per question.
- `execute_action(action) -> tuple[str, float, bool]` → `(feedback, reward, done)`.

**Reward shaping**

| action                              | reward         | effect                          |
|-------------------------------------|----------------|---------------------------------|
| `retrieve:<topic>` relevant & new   | `+2` then `-1` step cost (net `+1`) | add fact to `gathered` |
| `retrieve:<topic>` irrelevant/dup   | `-1` step cost | wasted move                     |
| `answer` with `gathered ⊇ required` | `+50`          | `victory=True`, `done`          |
| `answer` early                      | `-10`          | `done` (answered too early)     |
| `stop`                              | `-5`           | `done` (gave up)                |
| step budget exceeded (`steps >= max_steps`) | `-10`  | `done`                          |

`game_over` is set whenever `done` is returned.

**Stochastic mode** (mirrors the sample's `stochastic` flag): when
`stochastic=True`, a `retrieve:<topic>` for a *relevant* topic fails with
probability `p` (default `0.2`) — the fact is not added and only the `-1` step
cost applies. This forces the policy to be robust to retrieval noise. Randomness
uses the environment's own RNG (seeded when `seed` is provided).

### 2. `src/training/rl_agent.py` — adapt `QLearningAgent`

Edit the existing file:
- Add `from __future__ import annotations`.
- Replace `from game_environment import TreasureHuntGame` with
  `from .search_environment import SearchEnvironment`.
- Replace all `TreasureHuntGame` type hints with `SearchEnvironment`.
- Rewrite `_get_state_hash` to produce a small discrete key:
  `(current_question, frozenset(gathered), steps_remaining_bucket)`, where
  `steps_remaining_bucket` coarsely buckets `max_steps - steps`.
- Fix the victory heuristic in `train()`: the sample counts `reward > 50` as a
  win; switch to counting the env's actual `victory` flag per episode (tracked
  through `train_episode`, which already returns `game.victory`).
- Keep `choose_action`, `update_q_value`, `train_episode`, `train`, `evaluate`,
  `save`, `load` behavior — they are environment-agnostic and only call the
  interface methods above.
- **Delete `DQNAgent`** entirely (dead placeholder: prints a "requires neural
  network library" note and acts randomly; never learns).

The `_get_state_hash` no longer references treasure-hunt fields
(`current_room`, `inventory`, `locked_exits`, `has_guard`); those are replaced by
the search-environment fields.

### 3. `examples/run_rl_search_demo.py`

Runnable as `python3 -m examples.run_rl_search_demo`, matching
`run_agentic_search.py` style. Trains the agent, evaluates it, and prints
victory rate / average reward / Q-table size.

CLI flags:
- `--episodes` (default `2000`)
- `--eval-episodes` (default `200`)
- `--stochastic` (flag)
- `--seed` (default `0`)

### 4. `tests/unit/test_rl_search_agent.py`

Seeded (`random.seed` / `np.random.seed` / env `seed`) for reproducibility.

**Environment**
- Answering when `gathered ⊇ required` → `reward == +50`, `victory` True, `done`.
- Answering early (missing a required fact) → negative reward, `victory` False, `done`.
- `stop` → `-5`, `done`.
- Step-budget termination sets `done`/`game_over` after `max_steps`.
- Retrieving a relevant new topic adds the fact and yields net positive reward.

**Agent**
- One-transition `update_q_value` matches the Bellman update
  `Q += α (r + γ max_a' Q(s',a') − Q)` on a hand-computed example.
- `epsilon` decays toward `epsilon_min` across episodes.
- On a trivial single-fact question, victory rate exceeds a high threshold
  (e.g. > 0.9) after a bounded number of seeded training episodes.
- `save` then `load` round-trips the Q-table and hyperparameters.

### 5. `src/training/__init__.py`

Add to the existing `try:` import block:
```python
from .rl_agent import QLearningAgent as QLearningAgent
from .search_environment import SearchEnvironment as SearchEnvironment
```

## Success criteria

1. `python3 -m examples.run_rl_search_demo` trains and prints a rising victory
   rate; the trained agent's eval victory rate is high (deterministic mode).
2. `pytest tests/unit/test_rl_search_agent.py` passes.
3. `ruff check` / `ruff format` clean.
4. No import of `game_environment`; no `DQNAgent`; no serving code touched.
