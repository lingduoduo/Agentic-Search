# GRPO Advantage Dedupe and Cycle Untangle — Design

**Status:** approved
**Date:** 2026-08-24
**Scope:** `src/model/post_training/grpo/`, plus the two delegating methods in
`src/model/post_training/reward.py`

## Problem

PR #553 consolidated the GRPO package's *file layout*. It did not touch the
duplication inside those files. Three defects remain.

### 1. Group-relative advantage math is implemented eight times

Six of the eight operate on `list[float]` and compute the **same formula**:
mean-centering, optionally divided by the population standard deviation with
`ε = 1e-8`, with single-sample groups yielding `0.0`.

| Site | Formula |
| --- | --- |
| `grpo/rollouts.py::compute_grpo_outcome_advantage` | `r − mean` |
| `grpo/generation.py::_grpo_advantages` | `r − mean`, optionally `/(std + 1e-8)` |
| `grpo/training.py::LocalGRPOController.assign_group_advantages` | `(r − mean)/(std + 1e-8)` |
| `reward.py::SearchRewardFunction.compute_grpo_outcome_advantages` | `r − mean`, partitioned by `group_id` |
| `reward.py::SearchRewardFunction.compute_batch_advantages` | `(r − mean)/(std + 1e-8)`, partitioned by `group_id` |
| `grpo/generation.py::assign_group_relative_advantages` | wraps `_grpo_advantages` |

The remaining two are torch functions with genuinely different shapes and
epsilons — `grpo/core_algos.py::compute_grpo_outcome_advantage` (token-expanded,
`ε = 1e-6`, `std(unbiased=False)`) and `grpo/trainers.py::compute_group_advantages`
(2-D grouped with a validity mask, `sqrt(var + 1e-8)`). **They are out of scope.**
Unifying them would change numerics, which is not a refactor.

### 2. One public name resolves to two different functions

`compute_grpo_outcome_advantage` is defined in both `grpo/core_algos.py` (torch
tensors) and `grpo/rollouts.py` (`list[float]`). Which one a caller gets depends
on the import path:

```python
from src import compute_grpo_outcome_advantage                      # rollouts (scalar)
from src.model.post_training import compute_grpo_outcome_advantage  # rollouts (scalar)
from src.model.post_training.grpo import compute_grpo_outcome_advantage  # core_algos (torch)
```

The two have incompatible signatures, so a mistaken import fails at call time
rather than import time.

### 3. `generation.py` and `training.py` import each other

The cycle is worked around with seven function-local imports
(`training.py` lines 78, 91, 92, 123, 175, 252; `generation.py` lines 1344, 3680).
One of them, `training.py::_resolve_configs`, reaches back out through the
repo-root package (`from src import GRPOAdvantageConfig`) to obtain a symbol that
lives one module away in the same package.

Separately, `generation.py` line 32 imports `compute_grpo_outcome_advantage` from
`.core_algos` purely to re-export it. The symbol is never used in that
4,453-line module.

## Goals

- One implementation of the scalar group-relative advantage formula.
- No public name that resolves to two different functions.
- No import cycle inside the `grpo` package.
- No behavior change. Every numeric output identical, every public name that
  callers use still importable from the same place.

## Non-goals

- Merging or renumbering the two torch advantage functions.
- Splitting `LLMGenerationManager` (2,577 lines, ~70 methods). That is the
  package's largest complexity, and it is deliberately deferred: it increases
  file count, which is the opposite of what was asked for here.
- Removing `LLMGenerationManager.run_grpo_training_step`. It is a compatibility
  shim, but deleting it is a behavior change.
- Changing the lazy-export strategy in `grpo/__init__.py` or `src/__init__.py`.
  The eager-import ban stays: this package has broken torch-less CI four times.

## Design

### The shared primitive — and where it must live

`rollouts.py` and `reward.py` contain **no torch reference at all** today.
`grpo/core_algos.py` and `ppo/core_algos.py` both `import torch` at module
scope. So the obvious home for a shared advantage helper — `core_algos.py` — is
wrong: importing it from `rollouts.py` or `reward.py` would put torch in front
of two modules that do not need it. That is the exact failure mode this repo has
shipped four times (#356, #418, #517, #536).

The primitive therefore lives in `src/model/post_training/reward.py`, as two
module-level functions placed immediately above the `SearchRewardFunction`
methods that will delegate to them:

```python
def group_relative_advantages(
    rewards: Sequence[float], *, normalize: bool = False
) -> list[float]:
    """Mean-center one group's rewards; optionally divide by population std."""

def grouped_relative_advantages(
    rewards: Sequence[float], group_ids: Sequence[str], *, normalize: bool = False
) -> list[float]:
    """Partition by group id, apply the above per group, scatter back in order."""
```

`reward.py` earns this for four reasons, not just by elimination:

- It is torch-free and stays torch-free.
- It already owns two of the six duplicate implementations, and
  `SearchRewardFunction` already exposes advantage transforms — the module's
  remit already covers "a reward and the transform applied to it".
- It is the lowest node in the dependency graph among the six sites:
  `rollouts.py`, `judge.py`, `generation.py`, and `trainers.py` all import it
  already, and its own imports (`agents.core.base`, `context.search`) reach
  nothing in `grpo`. No new edge, no cycle.
- It adds no file. The instruction was to combine, not to split.

The trade-off, stated plainly: the shared helper for a `grpo/` cleanup ends up
outside `grpo/`. The torch-free constraint forces it, and no alternative
placement satisfies all of "torch-free", "no cycle", and "no new file".

Semantics, fixed by the six existing implementations:

- `len(rewards) == 0` → `[]`
- `len(rewards) == 1` → `[0.0]`
- `normalize=False` → `r − mean`
- `normalize=True` → `(r − mean) / (sqrt(Σ(r−mean)²/n) + 1e-8)`
- `grouped_*` raises `ValueError` when `rewards` and `group_ids` differ in length
- `grouped_*` gives a single-member group `0.0`, and preserves input order

### Call-site changes

| Site | Becomes |
| --- | --- |
| `reward.compute_grpo_outcome_advantages` | `return grouped_relative_advantages(rewards, group_ids)` |
| `reward.compute_batch_advantages` | `return grouped_relative_advantages(rewards, group_ids, normalize=True)` |
| `rollouts.compute_grpo_outcome_advantage` | `return group_relative_advantages(rewards)` |
| `generation._grpo_advantages` | deleted; `assign_group_relative_advantages` calls the primitive directly |
| `training.LocalGRPOController.assign_group_advantages` | assigns from `group_relative_advantages(..., normalize=True)` |

`rollouts.py` already imports from `..reward`, so its change is one added name on
an existing import line. `generation.py` and `training.py` import torch already,
so importing `..reward` costs them nothing new.

All five keep their existing names, signatures, docstring contracts, and error
messages. `reward.py`'s two methods stay methods — they are public API consumed
by `dpo` and `eval`.

### Name collision

`grpo/core_algos.py::compute_grpo_outcome_advantage` is renamed to
`compute_grpo_token_advantages`. The name states what it returns: advantages
expanded over response tokens. The scalar function keeps the shared name, which
is what `src/__init__.py` and `src/model/post_training/__init__.py` already
export under it — so the widest public surface is unchanged.

Updated references: `grpo/__init__.py` lazy map, `grpo/trainers.py` import,
`tests/unit/test_reward_shapes.py`. No compatibility alias is left behind,
consistent with the #553 precedent.

### Cycle

`async_run_grpo_training_step` moves from `generation.py` to `training.py`. It
is a pure delegator to `LocalGRPOController.async_training_step`, so it belongs
with the controller. `src/__init__.py`'s lazy entry repoints to
`.model.post_training.grpo.training`.

With that gone, `generation.py` imports nothing from `training.py` except inside
`LLMGenerationManager.run_grpo_training_step`, which is the retained compat
shim. `training.py` may then import `generation` at module scope, collapsing
five function-local imports into one module-level import block and letting
`TYPE_CHECKING` drop.

`training.py::_resolve_configs` switches `from src import GRPOAdvantageConfig`
to `from .rollouts import GRPOAdvantageConfig`.

### Dead code

- `generation.py::BatchRetriever` — a `Protocol` referenced only from a
  docstring in `Retriever`. Deleted, and the referring sentence with it.
- `generation.py` line 32 — the unused `compute_grpo_outcome_advantage`
  re-export. Deleted.

## Testing

The change is behavior-preserving, so the existing suite is the primary oracle.
On top of it:

1. **Characterization first.** Before any implementation edit, add tests that
   pin each of the six current implementations' outputs on a shared table of
   inputs — including `[]`, `[x]`, all-equal rewards (std 0, exercising the ε
   floor), and a multi-group case with interleaved group ids. These must pass
   against the *current* code.
2. **Then replace**, and the same tests must still pass unchanged.
3. **Mutation-check.** Delete the primitive's body and confirm the
   characterization tests go red. A test that stays green is not testing the
   thing it names.
4. Existing `tests/unit/test_grpo_module_layout.py` gains assertions that
   `async_run_grpo_training_step` resolves to the `training` module and that
   `grpo.core_algos` no longer exports `compute_grpo_outcome_advantage`.
5. **Torch-free guard.** `tests/unit/test_grpo_module_layout.py` already blocks
   torch via a `sys.meta_path` shim in a subprocess, but only to re-run its own
   filesystem-layout checks. Extend that same pattern to assert that
   `src.model.post_training.reward` and `src.model.post_training.grpo.rollouts`
   still *import* with torch blocked. That is the property this change puts at
   risk and the one the existing guard does not cover.
6. `ruff check . --fix && ruff format .` and the full `pytest` run.

## Risks

- **ε and variance convention drift.** The six implementations were verified
  identical by reading; the characterization tests exist to prove it rather than
  assert it. If any input reveals a divergence, that site is excluded from the
  dedupe and the divergence is documented, not silently normalized.
- **Lazy-export regressions.** Every `__init__.py` edit risks putting torch in
  front of a torch-less import. `tests/unit/test_grpo_module_layout.py` and the
  repo's existing torch-less guard cover this; the CI-torch-gap failure mode has
  recurred four times and is checked explicitly.
- **Renaming without a shim.** Any missed reference to
  `core_algos.compute_grpo_outcome_advantage` becomes an `ImportError`. Loud and
  immediate, which is the intent.

## Expected outcome

- 8 advantage implementations → 3 (one scalar, two torch).
- 1 ambiguous public name → 0.
- 7 function-local imports in the `generation`/`training` pair → 2.
- Roughly −180 LOC, and no new file: the shared primitive lands in a module all
  six sites already depend on.
- `rollouts.py` and `reward.py` remain torch-free, verifiably so.
