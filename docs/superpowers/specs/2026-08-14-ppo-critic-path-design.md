# Resolve the PPO-with-critic path that no trainer uses

## Status

Not started. Small, and the kind of thing that gets more expensive to decide the
longer it sits.

## Problem

`src/training/ppo/core_algos.py` implements `compute_value_loss` and
`compute_gae_advantages` — the PPO-with-critic path. Neither is called by any
trainer:

```
compute_value_loss(...)      -> called only in tests/unit/test_llm_agent_generation.py
compute_gae_advantages(...)  -> called only in tests/unit/test_reward_shapes.py
```

Everything else is a re-export: `src/model/generation.py` has
`compute_value_loss as compute_value_loss` (an explicit pass-through), and
`src/training/ppo/__init__.py` lists both in its export surface.

Training here is **critic-free GRPO** — group-relative advantages, no value
model. The training doc already says these helpers "exist for parity/tests only".

## Why this is worth deciding rather than leaving

It is not causing a bug, and that is precisely why it needs a decision rather
than drifting:

- The functions are **exported from two public surfaces**, so they read as
  supported API. Someone wiring a critic would reasonably assume this path is
  maintained.
- They are covered by tests, which makes the coverage number claim the path is
  exercised. It is exercised; it is not *used*.
- There is no value/critic model anywhere in the repo to feed
  `compute_gae_advantages` real values, so the tests necessarily pass synthetic
  ones. The tests verify shapes and arithmetic, not that the path works
  end-to-end — because there is no end to end.

## The two options

**Remove it.** Delete both functions, their re-exports, and their tests. Smaller
public surface, honest coverage, one fewer paradigm implied. Recoverable from git
if a critic is ever wanted.

**Keep it and say so loudly.** Leave the code, but mark it explicitly as
unreachable-by-design in the module docstring and both export lists, so nobody
mistakes it for a live path. The training doc already says this; the code does
not.

**This spec does not pre-judge which**, because the answer depends on something
not recorded anywhere: whether a critic-based trainer is actually wanted. That is
a question for whoever owns the training direction, and the spec exists to force
it to be asked rather than answered by neglect.

## Acceptance

Either:

- both helpers, their re-exports, and their tests are gone, with the full suite
  green and no import breaks (`src/training/__init__.py` and
  `src/training/ppo/__init__.py` both list them); **or**
- they remain with an explicit "not wired into any trainer; kept for X" note at
  each export site, naming the concrete reason.

Either way: the training doc's claim and the code stop disagreeing about how
supported this path is.

## Out of scope

`compute_ppo_policy_loss_core`, which **is** live — the GRPO trainers use it as
their clipped surrogate with a group-relative advantage in place of GAE. Do not
remove it while removing its neighbours; that is the obvious way for this
cleanup to break training.
