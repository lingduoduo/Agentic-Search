# GRPO Advantage Dedupe and Cycle Untangle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse six duplicate implementations of the scalar group-relative advantage formula into one, end a public name that resolves to two different functions, and break the `generation` ↔ `training` import cycle — with no behavior change.

**Architecture:** One pure-Python primitive pair lands in `src/model/post_training/reward.py` (the only module that is torch-free, already imported by every duplicate site, and free of any path back into `grpo`). Five call sites delegate to it while keeping their own names and signatures. The torch-tensor `compute_grpo_outcome_advantage` in `grpo/core_algos.py` is renamed to `compute_grpo_token_advantages` so the shared name is unambiguous. `async_run_grpo_training_step` moves from `generation.py` to `training.py`, letting `training.py` import `generation` at module scope.

**Tech Stack:** Python 3.10+, PyTorch (in the trainer/generation layers only), pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-24-grpo-advantage-dedupe-design.md`

## Global Constraints

- **No behavior change.** Every numeric output identical; every public name callers use stays importable from the same place it is importable from today.
- **`src/model/post_training/reward.py` and `src/model/post_training/grpo/rollouts.py` must remain torch-free.** Neither contains a single `torch` reference today. Adding one reintroduces a CI failure this repo has shipped four times (#356, #418, #517, #536). Never import `grpo/core_algos.py` or `ppo/core_algos.py` from either — both `import torch` at module scope.
- **No eager imports in `grpo/__init__.py` or `src/__init__.py`.** Both use lazy `__getattr__` registries. Keep it that way.
- **No compatibility shims or aliases** for renamed symbols, matching the #553 precedent. A missed reference must fail loudly.
- The population-variance convention and `ε = 1e-8` are load-bearing: `advantage_i = (r_i − mean) / (sqrt(Σ(r−mean)²/n) + 1e-8)`. Do not switch to sample variance (`n−1`) or a different epsilon.
- Empty input → `[]`; single-sample group → `[0.0]`.
- Do NOT touch `grpo/trainers.py::compute_group_advantages` or the renamed `compute_grpo_token_advantages` beyond the rename itself. They are torch functions with different shapes and epsilons; unifying them changes numerics.
- Do NOT split `LLMGenerationManager` and do NOT delete `LLMGenerationManager.run_grpo_training_step`. Both are explicit spec non-goals.
- Run `ruff check . --fix && ruff format .` before every commit. A pre-commit hook running ruff-format will abort the commit otherwise.
- Work on branch `refactor/grpo-advantage-dedupe`. Never commit to `main`.

---

### Task 1: The shared primitive, proven equivalent to all six existing implementations

The equivalence tests are the safety net for Tasks 2–5. They are written and green *before* any call site is touched, so any later divergence is caught immediately.

**Files:**
- Modify: `src/model/post_training/reward.py` (add two **module-level** functions immediately above the `class SearchRewardFunction` declaration — they are functions, not methods, so they must sit outside the class body)
- Test: `tests/unit/test_group_relative_advantages.py` (create)
- Modify: `tests/unit/test_grpo_module_layout.py` (extend the torch-blocked guard)

**Interfaces:**
- Produces: `src.model.post_training.reward.group_relative_advantages(rewards: Sequence[float], *, normalize: bool = False) -> list[float]` and `src.model.post_training.reward.grouped_relative_advantages(rewards: Sequence[float], group_ids: Sequence[str], *, normalize: bool = False) -> list[float]`. Tasks 2–5 consume both.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_group_relative_advantages.py`:

```python
"""The shared scalar advantage primitive, pinned against every implementation it replaces.

These equivalence tests run BEFORE the call sites are repointed. They are what
makes the dedupe safe: if any of the six implementations turns out to differ
from the primitive on any input, that shows up here rather than in a training run.
"""

from __future__ import annotations

import pytest

from src.model.post_training.reward import (
    SearchRewardConfig,
    SearchRewardFunction,
    group_relative_advantages,
    grouped_relative_advantages,
)

# Shared input table. Covers: empty, singleton, all-equal (std == 0, exercising
# the 1e-8 floor), a typical spread, a two-element group, and a zero-heavy group.
CASES: list[list[float]] = [
    [],
    [1.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.7, 0.0, 0.0],
    [-2.5, 3.5],
    [0.0, 0.0, 1.0],
]

GROUPED_REWARDS = [1.0, 0.0, 0.5, 1.0, 0.25]
GROUPED_IDS = ["a", "b", "a", "b", "a"]


@pytest.mark.parametrize("rewards", CASES)
def test_mean_centering_matches_the_documented_formula(rewards: list[float]):
    result = group_relative_advantages(rewards)
    if len(rewards) <= 1:
        assert result == [0.0] * len(rewards)
        return
    mean = sum(rewards) / len(rewards)
    assert result == pytest.approx([r - mean for r in rewards])


@pytest.mark.parametrize("rewards", CASES)
def test_normalized_divides_by_population_std_with_epsilon(rewards: list[float]):
    result = group_relative_advantages(rewards, normalize=True)
    if len(rewards) <= 1:
        assert result == [0.0] * len(rewards)
        return
    n = len(rewards)
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    std = (sum(c * c for c in centered) / n) ** 0.5
    assert result == pytest.approx([c / (std + 1e-8) for c in centered])


def test_all_equal_rewards_give_zero_advantage_not_nan():
    # std == 0; the epsilon floor is what keeps this finite.
    result = group_relative_advantages([2.0, 2.0, 2.0], normalize=True)
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_grouped_partitions_by_id_and_preserves_input_order():
    result = grouped_relative_advantages(GROUPED_REWARDS, GROUPED_IDS)
    # group "a" = indices 0, 2, 4 -> rewards 1.0, 0.5, 0.25, mean 0.583333...
    # group "b" = indices 1, 3    -> rewards 0.0, 1.0, mean 0.5
    mean_a = (1.0 + 0.5 + 0.25) / 3
    assert result == pytest.approx(
        [1.0 - mean_a, -0.5, 0.5 - mean_a, 0.5, 0.25 - mean_a]
    )


def test_grouped_gives_a_lone_member_zero():
    assert grouped_relative_advantages([5.0, 1.0, 2.0], ["solo", "x", "x"]) == (
        pytest.approx([0.0, -0.5, 0.5])
    )


def test_grouped_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        grouped_relative_advantages([1.0, 2.0], ["a"])


# --------------------------------------------------------------------------
# Equivalence with the implementations this primitive replaces.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rewards", CASES)
def test_matches_rollouts_compute_grpo_outcome_advantage(rewards: list[float]):
    from src.model.post_training.grpo.rollouts import compute_grpo_outcome_advantage

    assert group_relative_advantages(rewards) == pytest.approx(
        compute_grpo_outcome_advantage(list(rewards))
    )


def test_matches_reward_function_outcome_advantages():
    fn = SearchRewardFunction(SearchRewardConfig.second_pass())
    assert grouped_relative_advantages(GROUPED_REWARDS, GROUPED_IDS) == pytest.approx(
        fn.compute_grpo_outcome_advantages(list(GROUPED_REWARDS), list(GROUPED_IDS))
    )


def test_matches_reward_function_batch_advantages():
    fn = SearchRewardFunction(SearchRewardConfig.second_pass())
    assert grouped_relative_advantages(
        GROUPED_REWARDS, GROUPED_IDS, normalize=True
    ) == pytest.approx(
        fn.compute_batch_advantages(list(GROUPED_REWARDS), list(GROUPED_IDS))
    )


@pytest.mark.parametrize("rewards", CASES)
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_generation_grpo_advantages(rewards: list[float], normalize: bool):
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.generation import _grpo_advantages

    assert group_relative_advantages(rewards, normalize=normalize) == pytest.approx(
        _grpo_advantages(list(rewards), normalize=normalize)
    )


@pytest.mark.parametrize("rewards", CASES)
def test_matches_controller_assign_group_advantages(rewards: list[float]):
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.training import (
        LocalGRPOController,
        RolloutResult,
    )

    group = [
        RolloutResult(prompt_id=0, rollout_id=i, trajectory=None, reward=r)
        for i, r in enumerate(rewards)
    ]
    LocalGRPOController.assign_group_advantages(group)
    assert [item.advantage for item in group] == pytest.approx(
        group_relative_advantages(rewards, normalize=True)
    )
```

Note on `test_matches_generation_grpo_advantages`: it imports the private `_grpo_advantages`, which Task 2 deletes. Task 2 deletes this test with it — its whole job is to prove equivalence at the moment of replacement.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/unit/test_group_relative_advantages.py -v`

Expected: collection fails with `ImportError: cannot import name 'group_relative_advantages' from 'src.model.post_training.reward'`.

- [ ] **Step 3: Add the primitive to `reward.py`**

In `src/model/post_training/reward.py`, add `Sequence` to the imports. The file currently has `from typing import Callable` on line 8 and no `collections.abc` import; add a new line above it:

```python
from collections.abc import Sequence
from typing import Callable
```

Then insert these two functions immediately above the `class SearchRewardFunction` definition (module level, not methods):

```python
def group_relative_advantages(
    rewards: Sequence[float], *, normalize: bool = False
) -> list[float]:
    """Mean-center one prompt group's rewards; optionally divide by population std.

    This is the critic-free core signal GRPO trains on:

        advantage_i = reward_i - mean(group_rewards)

    With ``normalize=True`` the centered values are divided by the group's
    population standard deviation, which is what keeps the objective stable
    across groups whose reward scales differ:

        advantage_i = (reward_i - mean) / (std + 1e-8)

    A single-sample group has no relative comparison, so it gets ``0.0``.
    All-equal rewards give a zero std; the epsilon is what keeps the result
    finite rather than NaN.
    """
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    mean = sum(rewards) / n
    centered = [float(r) - mean for r in rewards]
    if not normalize:
        return centered
    std = math.sqrt(sum(c * c for c in centered) / n)
    return [c / (std + 1e-8) for c in centered]


def grouped_relative_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
    *,
    normalize: bool = False,
) -> list[float]:
    """Apply :func:`group_relative_advantages` per prompt group, in input order.

    Rollouts sharing a ``group_id`` were sampled from the same prompt and are
    the only ones compared against each other -- there is no cross-prompt
    mixing. The returned list is aligned with *rewards*.
    """
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length.")

    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)

    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        group_advantages = group_relative_advantages(
            [rewards[i] for i in indices], normalize=normalize
        )
        for index, advantage in zip(indices, group_advantages):
            advantages[index] = advantage
    return advantages
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest tests/unit/test_group_relative_advantages.py -v`

Expected: all pass. If any equivalence test fails, STOP — an implementation you were told is identical is not. Report the exact input and both outputs; do not "fix" it by changing the primitive.

- [ ] **Step 5: Mutation-check the equivalence tests**

Temporarily replace the body of `group_relative_advantages` with `return [0.0] * len(rewards)`.

Run: `pytest tests/unit/test_group_relative_advantages.py -v`

Expected: RED, and specifically every one of the four `test_matches_*` equivalence tests must fail on at least one parametrized case. If an equivalence test stays green, it is not testing what its name claims — fix the test, not the primitive. Then restore the real body and confirm green again.

- [ ] **Step 6: Extend the torch-blocked import guard**

`tests/unit/test_grpo_module_layout.py` already has `test_filesystem_layout_checks_run_without_torch`, which blocks torch via a `sys.meta_path` shim in a subprocess. Add a second, separate test beside it that asserts the two torch-free modules still import:

```python
def test_reward_and_rollouts_import_without_torch():
    """reward.py and rollouts.py must never acquire a torch dependency.

    They are the torch-free half of the post-training package, and the shared
    advantage primitive lives in reward.py precisely so that stays true.
    """
    import subprocess
    import sys

    program = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named %r (blocked)" % name)
        return None

sys.meta_path.insert(0, _Blocker())

import src.model.post_training.reward as reward
import src.model.post_training.grpo.rollouts as rollouts

assert reward.group_relative_advantages([1.0, 0.0]) == [0.5, -0.5]
assert rollouts.compute_grpo_outcome_advantage([1.0, 0.0]) == [0.5, -0.5]
print("torch-free OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "torch-free OK" in result.stdout
```

Do NOT put "replaced" or "approved_implementation_modules" in this test's name. `test_filesystem_layout_checks_run_without_torch` selects tests with `-k "replaced or approved_implementation_modules"` and asserts `"7 passed" in result.stdout`; a name that matches that filter breaks the count.

- [ ] **Step 7: Run the guard and the full layout suite**

Run: `pytest tests/unit/test_grpo_module_layout.py -v`

Expected: all pass, including the still-`7 passed` assertion inside `test_filesystem_layout_checks_run_without_torch`.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/post_training/reward.py tests/unit/test_group_relative_advantages.py tests/unit/test_grpo_module_layout.py
git commit -m "refactor(grpo): add shared group-relative advantage primitive

Pinned against all six existing implementations before any call site moves."
```

---

### Task 2: Repoint the five duplicate call sites

**Files:**
- Modify: `src/model/post_training/reward.py` (bodies of `compute_grpo_outcome_advantages` ~line 911 and `compute_batch_advantages` ~line 945)
- Modify: `src/model/post_training/grpo/rollouts.py:358-373` (`compute_grpo_outcome_advantage`)
- Modify: `src/model/post_training/grpo/generation.py:845-859` (delete `_grpo_advantages`) and its caller in `assign_group_relative_advantages` (~line 918)
- Modify: `src/model/post_training/grpo/training.py:64-75` (`LocalGRPOController.assign_group_advantages`)
- Modify: `tests/unit/test_group_relative_advantages.py` (delete the now-dangling `_grpo_advantages` equivalence test)

**Interfaces:**
- Consumes: `group_relative_advantages` and `grouped_relative_advantages` from Task 1.
- Produces: no new names. Every touched function keeps its existing name, signature, docstring contract, and `ValueError` messages.

- [ ] **Step 1: Repoint `reward.py`'s two methods**

Replace the body of `SearchRewardFunction.compute_grpo_outcome_advantages` (keep the `def` line, the parameter list, and the full docstring exactly as they are) with:

```python
        return grouped_relative_advantages(rewards, group_ids)
```

Replace the body of `SearchRewardFunction.compute_batch_advantages` (again keeping signature and docstring) with:

```python
        return grouped_relative_advantages(rewards, group_ids, normalize=True)
```

Both docstrings mention the `ValueError` on mismatched lengths; `grouped_relative_advantages` raises it with the identical message, so the contract holds. `compute_batch_advantages`'s docstring closes with a paragraph beginning "Single-pass: mean and std are computed together in one traversal" — that described the old body and is no longer true. Delete that paragraph and nothing else.

- [ ] **Step 2: Repoint `rollouts.py`**

Add `group_relative_advantages` to the existing import on line 13, keeping alphabetical order:

```python
from ..reward import (
    BatchJudgeFn,
    JudgeFn,
    SearchRewardFunction,
    _score_answers,
    group_relative_advantages,
)
```

Replace the body of `compute_grpo_outcome_advantage` (keep its `def` line and docstring) with:

```python
    return group_relative_advantages(rewards)
```

- [ ] **Step 3: Repoint `generation.py`**

Delete the whole `_grpo_advantages` function (lines 845-859). Do NOT remove `import math` on line 6 — it is still used at line 4133.

Add the import near the other relative imports at the top of the file:

```python
from ..reward import group_relative_advantages
```

In `assign_group_relative_advantages`, replace:

```python
    advantages = _grpo_advantages(list(rewards), normalize=normalize)
```

with:

```python
    advantages = group_relative_advantages(list(rewards), normalize=normalize)
```

- [ ] **Step 4: Repoint `training.py`**

Replace `LocalGRPOController.assign_group_advantages` (keep the `@staticmethod` decorator, the `def` line, and the docstring) with:

```python
    @staticmethod
    def assign_group_advantages(group: list[RolloutResult]) -> list[RolloutResult]:
        """Assign std-normalized advantages to a simple rollout group."""
        if not group:
            return group
        advantages = group_relative_advantages(
            [float(item.reward) for item in group], normalize=True
        )
        for item, advantage in zip(group, advantages):
            item.advantage = advantage
        return group
```

Add to the top-level imports of `training.py`:

```python
from ..reward import group_relative_advantages
```

`training.py` already imports torch-dependent modules, so importing `..reward` costs it nothing new and creates no cycle (`reward.py` reaches only `agents.core.base` and `context.search`, neither of which touches `grpo`).

- [ ] **Step 5: Delete the dangling equivalence test**

In `tests/unit/test_group_relative_advantages.py`, delete `test_matches_generation_grpo_advantages` in full, including its two `@pytest.mark.parametrize` decorators. `_grpo_advantages` no longer exists, so the test would error on import. The other three `test_matches_*` tests stay — the functions they compare against still exist as delegating wrappers, so they now assert the delegation is wired correctly.

- [ ] **Step 6: Run the full test suite and verify GREEN**

Run: `pytest -q`

Expected: all pass, with the same count as `main` minus one (the deleted test). Pay attention to `tests/unit/test_reward_shapes.py`, `tests/unit/test_llm_agent_generation.py`, `tests/unit/test_train_loop.py`, and `tests/unit/test_simulated_judge.py` — those exercise the four repointed sites.

- [ ] **Step 7: Mutation-check the delegation**

Temporarily change `grouped_relative_advantages`'s `normalize` handling so it ignores the keyword — make the `compute_batch_advantages` path return un-normalized values.

Run: `pytest tests/unit/test_group_relative_advantages.py tests/unit/test_reward_shapes.py -q`

Expected: RED. If green, the delegation is not actually covered; add a test that pins `compute_batch_advantages` to a known normalized output before continuing. Restore afterwards and confirm green.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(grpo): route all scalar advantage math through one primitive

Six implementations of (r - mean) / (std + 1e-8) collapse to one."
```

---

### Task 3: End the `compute_grpo_outcome_advantage` name collision

Today the name resolves to a torch function via `src.model.post_training.grpo` and to a `list[float]` function via `src` and `src.model.post_training`. `tests/unit/test_reward_shapes.py` already aliases the torch one `as compute_grpo_outcome_advantage_tensor` to work around this.

**Files:**
- Modify: `src/model/post_training/grpo/core_algos.py:25` (the `def` line and its docstring)
- Modify: `src/model/post_training/grpo/__init__.py` (lazy export map)
- Modify: `src/model/post_training/grpo/trainers.py:21` (import)
- Modify: `src/model/post_training/grpo/generation.py:32-34` (delete the unused re-export)
- Modify: `tests/unit/test_reward_shapes.py:17-19, 484, 493`
- Modify: `tests/unit/test_grpo_module_layout.py` (add a resolution assertion)

**Interfaces:**
- Produces: `src.model.post_training.grpo.core_algos.compute_grpo_token_advantages` with the signature unchanged: `(token_level_rewards, eos_mask, index, epsilon=1e-6, clip_advantages=None) -> tuple[torch.Tensor, torch.Tensor]`.
- The name `compute_grpo_outcome_advantage` now resolves to exactly one function — the `list[float]` one in `grpo/rollouts.py` — from every import path.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_grpo_module_layout.py`:

```python
def test_outcome_advantage_name_resolves_to_exactly_one_function():
    """One public name, one function, whatever the import path.

    Before this, `from ...grpo import compute_grpo_outcome_advantage` gave a
    torch tensor function while `from src import` gave a list[float] one, with
    incompatible signatures -- so a wrong import failed at call time, not
    import time.
    """
    pytest.importorskip("torch", exc_type=ImportError)

    import src as root
    from src.model.post_training import grpo
    from src.model.post_training.grpo import core_algos, rollouts

    assert not hasattr(core_algos, "compute_grpo_outcome_advantage")
    assert hasattr(core_algos, "compute_grpo_token_advantages")

    assert (
        grpo.compute_grpo_outcome_advantage
        is rollouts.compute_grpo_outcome_advantage
    )
    assert (
        root.compute_grpo_outcome_advantage
        is rollouts.compute_grpo_outcome_advantage
    )
```

- [ ] **Step 2: Run it and verify RED**

Run: `pytest tests/unit/test_grpo_module_layout.py::test_outcome_advantage_name_resolves_to_exactly_one_function -v`

Expected: FAIL on `assert not hasattr(core_algos, "compute_grpo_outcome_advantage")`.

- [ ] **Step 3: Rename in `core_algos.py`**

Change the `def` on line 25 from `compute_grpo_outcome_advantage` to `compute_grpo_token_advantages`. Leave the parameters, defaults, body, and return unchanged. Update the first docstring line to name what it returns:

```python
    """Group-normalized outcome advantages expanded over response tokens.

    Named for its shape: unlike the scalar ``compute_grpo_outcome_advantage``
    in :mod:`src.model.post_training.grpo.rollouts`, this returns a
    ``(batch, seq_len)`` tensor with each rollout's advantage broadcast across
    its response tokens.
    """
```

Keep the rest of the existing docstring (the `Args:` block) exactly as it is.

- [ ] **Step 4: Update the two importers**

In `src/model/post_training/grpo/__init__.py`, change the `_LAZY_EXPORTS` key `"compute_grpo_outcome_advantage": "core_algos"` to `"compute_grpo_token_advantages": "core_algos"`. Leave every other entry alone. `__all__` is derived from the dict, so it updates itself.

In `src/model/post_training/grpo/trainers.py` line 21, change:

```python
from .core_algos import compute_grpo_token_advantages
```

and update its use sites in that file to match (grep for `compute_grpo_outcome_advantage` in `trainers.py`).

- [ ] **Step 5: Delete the unused re-export in `generation.py`**

Remove lines 32-34 entirely:

```python
from .core_algos import (
    compute_grpo_outcome_advantage as compute_grpo_outcome_advantage,
)
```

The symbol is imported only to be re-exported and is never called in that 4,453-line module — verified by grep: the only occurrence is the import itself. Do not replace it with the new name.

- [ ] **Step 6: Update `test_reward_shapes.py`**

Replace lines 17-19:

```python
from src.model.post_training.grpo.core_algos import compute_grpo_token_advantages
```

The `as compute_grpo_outcome_advantage_tensor` alias existed only to dodge the collision and is no longer needed. Update the two call sites (lines ~484 and ~493) and the section comment on line 475 to use `compute_grpo_token_advantages`.

- [ ] **Step 7: Verify nothing references the old name**

Run: `grep -rn --include='*.py' --exclude-dir=.worktrees 'compute_grpo_outcome_advantage' .`

Expected: every remaining hit refers to the `list[float]` function in `grpo/rollouts.py` — its definition, `src/__init__.py`, `src/model/post_training/__init__.py`, `grpo/__init__.py`, the delegating body, and tests. No hit should be in `core_algos.py`, `trainers.py`, or `generation.py`.

- [ ] **Step 8: Run the tests and verify GREEN**

Run: `pytest tests/unit/test_grpo_module_layout.py tests/unit/test_reward_shapes.py -v`

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(grpo): rename tensor advantage fn to compute_grpo_token_advantages

One public name no longer resolves to two incompatible functions."
```

---

### Task 4: Break the `generation` ↔ `training` import cycle

**Files:**
- Modify: `src/model/post_training/grpo/generation.py:1316-1367` (delete `async_run_grpo_training_step`)
- Modify: `src/model/post_training/grpo/training.py` (add it; hoist the function-local imports)
- Modify: `src/__init__.py:255-258` (repoint the lazy entry)
- Modify: `tests/unit/test_llm_agent_generation.py:4548, 4642` (import path)
- Modify: `tests/unit/test_grpo_module_layout.py` (add an ownership assertion)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `src.model.post_training.grpo.training.async_run_grpo_training_step`, with the signature byte-identical to the one it replaces (21 parameters; see Step 3).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_grpo_module_layout.py`:

```python
def test_training_owns_the_async_step_and_generation_does_not_import_training():
    """The controller entrypoint lives with the controller.

    It was a pure delegator sitting in generation.py, which forced a
    generation <-> training cycle papered over with function-local imports.
    """
    pytest.importorskip("torch", exc_type=ImportError)

    from src.model.post_training.grpo import generation, training

    assert (
        training.async_run_grpo_training_step.__module__
        == "src.model.post_training.grpo.training"
    )
    assert not hasattr(generation, "async_run_grpo_training_step")

    import src as root

    assert (
        root.async_run_grpo_training_step is training.async_run_grpo_training_step
    )
```

- [ ] **Step 2: Run it and verify RED**

Run: `pytest tests/unit/test_grpo_module_layout.py::test_training_owns_the_async_step_and_generation_does_not_import_training -v`

Expected: FAIL with `AttributeError: module '...training' has no attribute 'async_run_grpo_training_step'`.

- [ ] **Step 3: Move the function into `training.py`**

Cut `async_run_grpo_training_step` from `generation.py` (lines 1316-1367) and paste it at module level in `training.py`, below the `LocalGRPOController` class. Drop its now-pointless function-local `from .training import LocalGRPOController` and call the class directly. The signature is unchanged:

```python
async def async_run_grpo_training_step(
    manager: LLMGenerationManager,
    prompt_batch: Any,
    *,
    search_mode: str,
    sampling_params: dict[str, Any],
    judge_fn: Callable[[str, str], float],
    num_rollouts: int = 4,
    reward_fn: Any = None,
    advantage_config: Any = None,
    batch_judge_fn: Any = None,
    old_backend: Any = None,
    new_backend: Any = None,
    ref_backend: Any = None,
    loss_config: PPOPolicyLossConfig | None = None,
    safety_config: Any = None,
    optimizer: Any = None,
    base_seed: int | None = None,
    current_step: int = 0,
    total_steps: int = 1,
    max_workers: int | None = None,
) -> Any:
    """Run one GRPO trainer step with concurrent rollout collection.

    Delegates to ``LocalGRPOController.async_training_step`` which runs all
    ``N_prompts × N_rollouts`` trajectories in parallel, overlapping HTTP
    search I/O, then performs one learner-side update.
    """
    return await LocalGRPOController(
        manager, num_rollouts=num_rollouts, max_workers=max_workers
    ).async_training_step(
        prompt_batch,
        search_mode=search_mode,
        sampling_params=sampling_params,
        judge_fn=judge_fn,
        num_rollouts=num_rollouts,
        reward_fn=reward_fn,
        advantage_config=advantage_config,
        batch_judge_fn=batch_judge_fn,
        old_backend=old_backend,
        new_backend=new_backend,
        ref_backend=ref_backend,
        loss_config=loss_config,
        safety_config=safety_config,
        optimizer=optimizer,
        base_seed=base_seed,
        current_step=current_step,
        total_steps=total_steps,
    )
```

The `LogProbCapable`, `GRPORolloutSafetyConfig`, and `GRPOTrainingStepResult` annotations become `Any` because those types live in `generation.py`; Step 4 makes them importable, so restore the precise annotations there rather than leaving `Any`.

- [ ] **Step 4: Hoist `training.py`'s function-local imports to module scope**

`generation.py` now imports nothing from `training.py` at module scope, so `training.py` can import it normally. Replace the `if TYPE_CHECKING:` block (lines 18-19) with a real import block near the other relative imports:

```python
from .generation import (
    GRPOPromptGroupResult,
    GRPORolloutSafetyConfig,
    GRPOTrainingStepResult,
    LLMGenerationManager,
    LogProbCapable,
    _single_prompt_batch,
    apply_safety_penalties_to_scored_rollouts,
    async_run_prompt_rollout_group,
    score_group_rollout,
)
from .rollouts import GRPOAdvantageConfig
```

Then delete the five function-local import statements at (original) lines 78, 91-92, 123-127, 175, and 252, and restore the precise type annotations on `async_run_grpo_training_step` now that `LogProbCapable`, `GRPORolloutSafetyConfig`, and `GRPOTrainingStepResult` are in scope.

Note the `from .rollouts import GRPOAdvantageConfig`: `_resolve_configs` currently does `from src import GRPOAdvantageConfig`, reaching back out through the repo-root lazy package for a symbol one module away in the same package. Use the direct relative import.

Remove `TYPE_CHECKING` from the `typing` import on line 16 if nothing else uses it.

- [ ] **Step 5: Confirm exactly one deferred import remains**

Run: `grep -n "^\s\+from \.\|^\s\+import " src/model/post_training/grpo/training.py src/model/post_training/grpo/generation.py | grep -E "from \.(generation|training)"`

Expected: exactly one hit — the `from .training import LocalGRPOController` inside `LLMGenerationManager.run_grpo_training_step` in `generation.py`. That one stays: it is the retained compatibility shim, and it is a method on a class `training.py` imports, so hoisting it would recreate the cycle.

- [ ] **Step 6: Repoint the lazy export and the tests**

In `src/__init__.py` lines 255-258, change the module path:

```python
    "async_run_grpo_training_step": (
        ".model.post_training.grpo.training",
        "async_run_grpo_training_step",
    ),
```

Leave `"async_run_prompt_rollout_group"` pointing at `generation` — that function did not move.

In `tests/unit/test_llm_agent_generation.py` line 4548, change the import to `from src.model.post_training.grpo.training import async_run_grpo_training_step`. Check line 4642's call site still resolves.

- [ ] **Step 7: Run the tests and verify GREEN**

Run: `pytest tests/unit/test_grpo_module_layout.py tests/unit/test_llm_agent_generation.py tests/unit/test_train_loop.py -v`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(grpo): break the generation<->training import cycle

async_run_grpo_training_step moves to the controller module it delegates to."
```

---

### Task 5: Remove dead code, verify, and document

**Files:**
- Modify: `src/model/post_training/grpo/generation.py:1517-1531` (delete `BatchRetriever`) and the `Retriever` docstring above it
- Modify: `src/model/post_training/grpo/__init__.py` (module docstring accuracy)
- Modify: `CLAUDE.md` (the `grpo/` bullet under **Post-training**)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: no new names.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_grpo_module_layout.py`:

```python
def test_unused_batch_retriever_protocol_is_gone():
    """BatchRetriever was referenced nowhere outside its own definition --
    nothing implemented it, nothing isinstance-checked it, and no other
    docstring named it."""
    pytest.importorskip("torch", exc_type=ImportError)

    from src.model.post_training.grpo import generation

    assert not hasattr(generation, "BatchRetriever")
```

- [ ] **Step 2: Run it and verify RED**

Run: `pytest tests/unit/test_grpo_module_layout.py::test_unused_batch_retriever_protocol_is_gone -v`

Expected: FAIL — the attribute still exists.

- [ ] **Step 3: Delete `BatchRetriever`**

Remove the `@runtime_checkable` decorator, the `class BatchRetriever(Protocol):` block, and its `retrieve_batch` method (lines 1517-1531) in full. Keep `runtime_checkable` in the `typing` import — `Fetcher` still uses it.

The `Retriever` docstring immediately above lists implementations; it does not mention `BatchRetriever`, so leave it alone. Grep for any other prose reference:

Run: `grep -rn --include='*.py' --include='*.md' --exclude-dir=.worktrees 'BatchRetriever' .`

Expected: no hits after the deletion.

- [ ] **Step 4: Run it and verify GREEN**

Run: `pytest tests/unit/test_grpo_module_layout.py -v`

Expected: all pass.

- [ ] **Step 5: Update the `grpo/__init__.py` docstring**

The module docstring's closing paragraph currently says `rollouts` is not re-exported because it pulls in the agent loop, and describes the lazy-export rationale. Both still hold — do not rewrite them. Add one sentence to the paragraph about what is not GRPO, recording where the shared advantage primitive went and why:

```
The scalar group-relative advantage formula that every trainer here needs lives
in ``src.model.post_training.reward`` rather than in this package: ``reward`` and
``rollouts`` are the torch-free half of post-training, and importing
``core_algos`` from them would put ``import torch`` in front of both.
```

- [ ] **Step 6: Update `CLAUDE.md`**

In the **Post-training** section, the `grpo/` bullet lists `core_algos.py` as holding "GRPO advantage + the REINFORCE losses". Change that phrase to name the rename and the new home:

```
  `core_algos.py` (the token-level GRPO advantage `compute_grpo_token_advantages`
  + the REINFORCE losses, kept as the ancestor policy-gradient algorithm; the
  scalar group-relative advantage primitive shared by every trainer lives in
  `post_training/reward.py`, which is torch-free)
```

Leave the rest of the bullet unchanged.

- [ ] **Step 7: Full verification**

```bash
ruff check . && ruff format --check .
cd web && npm run typecheck && cd ..
pytest -q
```

Expected: ruff clean, typecheck clean (untouched, but confirms nothing collateral), full suite green. Record the pass count and compare against `main`'s: it should be `main`'s count, plus the new tests from Tasks 1/3/4/5, minus the one deleted in Task 2.

- [ ] **Step 8: Confirm the spec's claimed outcome**

```bash
grep -rn --include='*.py' --exclude-dir=.worktrees -E 'def (compute_grpo_outcome_advantage|compute_grpo_token_advantages|compute_group_advantages|group_relative_advantages|grouped_relative_advantages|assign_group_advantages|_grpo_advantages|compute_batch_advantages|compute_grpo_outcome_advantages)\b' src/
```

Expected: exactly eight definitions, and `_grpo_advantages` must NOT be among them.

| Definition | Role |
| --- | --- |
| `reward.group_relative_advantages` | **the** scalar implementation |
| `reward.grouped_relative_advantages` | partitions by group id, delegates |
| `reward.compute_grpo_outcome_advantages` | delegating wrapper (public API) |
| `reward.compute_batch_advantages` | delegating wrapper (public API) |
| `rollouts.compute_grpo_outcome_advantage` | delegating wrapper (public API) |
| `training.assign_group_advantages` | delegating wrapper |
| `trainers.compute_group_advantages` | torch, 2-D masked — untouched |
| `core_algos.compute_grpo_token_advantages` | torch, token-expanded — renamed only |

Eight definitions, but **three real implementations**: one scalar and two
torch, down from eight. The four wrappers are one line each and exist solely
to preserve the public surface.

Then confirm the diff size:

```bash
git diff --stat main...HEAD
```

Expected (measured after the fact): `src/` diffs to 166 insertions / 170
deletions, a net −4 lines, not the roughly 180 originally estimated here.
Each duplicate body was only ~10-15 lines and its one-line replacement keeps
the full docstring, while the new primitive adds ~60 documented lines of its
own -- the real reduction is structural (8 implementations → 3, one
ambiguous public name → 0, 7 deferred imports → 1), not in line count.

- [ ] **Step 9: Commit and open the PR**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(grpo): drop unused BatchRetriever protocol and sync docs"
git push -u origin refactor/grpo-advantage-dedupe
gh pr create --title "refactor(grpo): collapse six duplicate advantage implementations and break the generation/training cycle" --body "$(cat <<'BODY'
## Summary

Six implementations of the same scalar group-relative advantage formula
collapse into one primitive. No behavior change.

- `group_relative_advantages` / `grouped_relative_advantages` land in
  `post_training/reward.py` -- the only home that is torch-free, already
  imported by every duplicate site, and free of a path back into `grpo`.
  Putting them in `core_algos.py` would have dragged torch in front of
  `reward.py` and `rollouts.py`, which is the CI failure this repo has
  shipped four times.
- `compute_grpo_outcome_advantage` no longer resolves to two different
  functions depending on import path. The torch one is now
  `compute_grpo_token_advantages`.
- `async_run_grpo_training_step` moves to `training.py`, breaking the
  `generation` <-> `training` cycle. Seven function-local imports become one
  (the `from .training import LocalGRPOController` inside the retained
  `LLMGenerationManager.run_grpo_training_step` compat shim; `training.py`
  now has zero function-local imports of `generation`).
- Drops the unused `BatchRetriever` protocol and an unused re-export.

Explicit non-goals: splitting `LLMGenerationManager`, and unifying the two
torch advantage functions (different shapes and epsilons -- merging them
would change numerics).

## Verification

Every replaced implementation was pinned by equivalence tests *before* the
call sites moved, and those tests were mutation-checked. Full suite green.

Spec: `docs/superpowers/specs/2026-08-24-grpo-advantage-dedupe-design.md`
Plan: `docs/superpowers/plans/2026-08-24-grpo-advantage-dedupe.md`
BODY
)"
```

Use a specific PR title; do not reuse a title from a previous GRPO refactor PR.
