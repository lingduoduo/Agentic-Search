## Summary

Reduces `src/model/post_training/grpo` from seven implementation modules to
three, moves the shared causal-LM log-prob helper to a neutral home, and
optimizes the reward and online-training hot paths against recorded benchmarks.

No public name, call signature, reward value, advantage value, loss reduction,
rollout ordering, or checkpoint key changes.

**Spec:** `docs/superpowers/specs/2026-08-27-grpo-minimal-architecture-design.md`
**Plan:** `docs/superpowers/plans/2026-08-27-grpo-minimal-architecture.md`

## Architecture

| Before (7 modules) | After (3 modules) |
| --- | --- |
| `core_algos.py`, `rollouts.py`, `judge.py` | `algorithms.py` |
| `generation.py` | `generation.py` |
| `trainers.py`, `training.py` | `training.py` |
| `plot_rollouts.py` | → `examples/plot_grpo_rollouts.py` |

Dependencies are now acyclic — `training → generation → algorithms`, enforced by
an AST test. The controller/generation cycle is gone: the step mechanics live in
generation and `LocalGRPOController` delegates to them.

`get_response_log_probs` moved to `src/model/post_training/log_probs.py`. DPO no
longer imports a GRPO trainer to get its token alignment.

## Optimization evidence

Full tables in `docs/benchmarks/grpo-optimization-baseline.md`. Medians of three
runs, 128 rollouts, same host and fixtures.

| case | change |
| --- | ---: |
| `reward_components_sparse` | **−75%** |
| `log_prob_row_collection` | **−81%** |
| `group_advantages` | **−29%** |
| `group_advantages_normalized` | **−17%** |
| `left_pad_prompt_rows` | **−13.5%** time, **−99%** peak Python allocation |
| everything else | ±3% (noise) |

What changed:
- One canonical `compute_group_relative_advantages` kernel in `reward.py`
  replaces two near-identical grouping passes.
- The citation dimension resolves an answer's citations once instead of up to
  three times.
- A preset whose weights are all zero no longer walks the search context or
  evaluates shaping arithmetic. `sparse_final_only` — the preset the docs
  recommend for a first agent-RL phase — is 4× faster.
- Left-padding writes into a preallocated tensor instead of building a nested
  Python list.

### Rejected candidates

Recorded so they are not re-proposed:

- **`torch.inference_mode()` for the frozen reference.** Correct, but 7–12%
  *slower* at realistic hidden/vocab sizes; it wins only on toy models.
- **`pad_sequence` with row reversal for left-padding.** 18% slower than the
  nested list it would have replaced.
- An intermediate `all(getattr(cfg, name) == 0.0 ...)` form of the zero-weight
  guards cost the shaped path a repeatable +1.9%; rewritten as a
  short-circuiting `and` chain.
- **A hand-rolled accumulator in place of `sum()`** in the advantage kernel. It
  benchmarked 16 points better (−45%/−39% rather than −29%/−17%) and was
  reverted: on CPython 3.12+ `sum()` applies Neumaier compensation over floats,
  so the manual loop moved 8,745/20,000 random groups on centering and
  10,634/20,000 on the normalized path. The divergence is Python-version
  dependent, so "identical" would not have been a stable property either.

The harness itself had a measurement bug — it timed samples with `tracemalloc`
running, inflating every median roughly threefold. Timing and allocation are now
separate passes and the first baseline was discarded.

## Corrections to the plan

1. `compute_grpo_outcome_advantage` existed **twice** with different signatures,
   exported from different places (package = tensor form, root = list form).
   Both are public API, so neither could be renamed. `algorithms.py` now owns one
   dispatching entry point with a typed overload per call shape.
2. `plot_rollouts.py` had been **deleted without being relocated** by an earlier
   commit on this branch. Restored, with its real `--jsonl/--out` interface.
3. `torch.inference_mode()` was proposed as an optimization; it does not pay.

## Compatibility

- Package and root lazy exports keep their names and identity; nothing eagerly
  imports torch (verified in a torch-blocked subprocess).
- No compatibility shims for the deleted module paths, by design.
- Preset breakdowns are pinned against a golden file captured from the previous
  implementation (`tests/unit/reward_breakdown_baseline.json`).

## Review

Reviewed before merge; findings applied in `3e67ac4`. Three were substantive:

- The `sum()` regression above, plus the fact that the two tests meant to police
  it asserted only that the wrapper delegates to the kernel — both sides moved
  together, so any arithmetic change stayed green. Replaced with differential
  oracles against the pre-refactor implementations, mutation-checked.
- The log-prob row change had shipped with **no measurement**; the benchmark
  case named `training_batch_assembly` only ever exercised the padding helper.
  Split into `left_pad_prompt_rows` and `log_prob_row_collection`, both measured.
- A 1,088-line scratch file was swept into a commit by `git add -A` in a shared
  worktree. Removed.

One known cosmetic divergence, deliberately accepted: where a whole dimension is
zero-weighted and a metric is negative, the old path produced `-0.0` and the
constant dict now yields `0.0`. `==`, `approx`, and every sum are unaffected;
only `repr`/JSON text differ.

## Verification

- `pytest` — 3547 passed, 3 skipped, no new warnings
- `ruff check` / `ruff format --check` / `git diff --check` — clean
- No live import or doc references a deleted module; the package contains
  exactly `__init__.py`, `algorithms.py`, `generation.py`, `training.py`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
