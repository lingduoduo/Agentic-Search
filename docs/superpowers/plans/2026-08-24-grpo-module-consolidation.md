# GRPO Module Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the GRPO package by four implementation modules while preserving behavior and package-level public symbols.

**Architecture:** Merge the three trainer implementations into `trainers.py`, merge controller and durable-loop orchestration into `training.py`, and move the generation-only tensor helpers into `generation.py`. Delete the six superseded modules and update every repository import; retain lazy package and root exports to avoid eager PyTorch and agent-loop imports.

**Tech Stack:** Python 3.10+, PyTorch, pytest, Ruff, Git lazy-import registries.

**Spec:** `docs/superpowers/specs/2026-08-24-grpo-module-consolidation-design.md`

## Global Constraints

- Preserve all existing class/function names, signatures, runtime behavior, exception semantics, and checkpoint formats.
- Preserve imports from `src.model.post_training.grpo` and lazy exports from `src`.
- Remove old direct module paths without compatibility shim files.
- Keep `core_algos.py`, `rollouts.py`, `judge.py`, and `plot_rollouts.py` separate.
- Do not move trainer or orchestration behavior into the 4,000-line `generation.py`.
- Do not introduce eager imports in `grpo/__init__.py` or `src/__init__.py`.

---

### Task 1: Consolidate the Trainer Hierarchy

**Files:**
- Create: `src/model/post_training/grpo/trainers.py`
- Delete: `src/model/post_training/grpo/grpo_trainer.py`
- Delete: `src/model/post_training/grpo/llm_grpo_trainer.py`
- Delete: `src/model/post_training/grpo/search_agent_grpo_trainer.py`
- Modify: `src/model/post_training/grpo/__init__.py`
- Modify: `src/model/post_training/dpo/trainer.py`
- Modify: `examples/_grpo_common.py`
- Modify: `examples/run_bamboogle_grpo_train.py`
- Modify: `examples/run_retriever_aware_grpo.py`
- Modify: `tests/unit/test_grpo_trainer.py`
- Modify: `tests/unit/test_llm_grpo_trainer.py`
- Modify: `tests/unit/test_search_agent_grpo_trainer.py`
- Modify: `tests/unit/test_grpo_common.py`
- Create: `tests/unit/test_grpo_module_layout.py`

**Interfaces:**
- Consumes: `compute_grpo_outcome_advantage` from `.core_algos`; rollout types from `.rollouts`; PPO helpers from `..ppo.core_algos`.
- Produces: `Policy`, `GRPOTrainer`, `make_grpo_trainer`, `compute_group_advantages`, `grpo_clipped_policy_loss`, `reverse_kl_penalty`, `LLMGRPOConfig`, `LLMRolloutResult`, `get_response_log_probs`, `LLMGRPOTrainer`, and `SearchAgentGRPOTrainer` from `.trainers`.

- [ ] **Step 1: Write the failing trainer layout contract**

Create `tests/unit/test_grpo_module_layout.py` with literal module expectations:

```python
from __future__ import annotations

from importlib.util import find_spec

import pytest

pytest.importorskip("torch")


def test_consolidated_trainer_module_owns_the_full_hierarchy():
    from src.model.post_training.grpo.trainers import (
        GRPOTrainer,
        LLMGRPOTrainer,
        SearchAgentGRPOTrainer,
    )

    assert LLMGRPOTrainer in SearchAgentGRPOTrainer.__mro__
    assert GRPOTrainer.__module__ == "src.model.post_training.grpo.trainers"
    assert LLMGRPOTrainer.__module__ == "src.model.post_training.grpo.trainers"
    assert SearchAgentGRPOTrainer.__module__ == "src.model.post_training.grpo.trainers"


@pytest.mark.parametrize(
    "module_name",
    [
        "src.model.post_training.grpo.grpo_trainer",
        "src.model.post_training.grpo.llm_grpo_trainer",
        "src.model.post_training.grpo.search_agent_grpo_trainer",
    ],
)
def test_replaced_trainer_modules_are_removed(module_name: str):
    assert find_spec(module_name) is None
```

- [ ] **Step 2: Run the layout contract and verify RED**

Run: `pytest tests/unit/test_grpo_module_layout.py -v`

Expected: collection fails because `src.model.post_training.grpo.trainers` does not exist. If the import test is temporarily split to allow collection, the old-module assertions must also fail because all three old specs still exist.

- [ ] **Step 3: Build `trainers.py` without behavior changes**

Use `apply_patch` to create one module in this order:

1. shared imports (`asyncio`, `copy`, `torch`, `torch.nn`, `torch.optim`);
2. bandit tensor utilities and `Policy`/`GRPOTrainer`;
3. `LLMGRPOConfig`, `LLMRolloutResult`, log-prob helpers, and `LLMGRPOTrainer`;
4. `_resolve_max_concurrent` and `SearchAgentGRPOTrainer`.

Copy the existing definitions exactly. Deduplicate imports, replace the old
`from .llm_grpo_trainer import ...` with direct references to definitions above,
and retain the `.rollouts`, `.core_algos`, and PPO imports.

- [ ] **Step 4: Redirect all trainer imports and lazy exports**

Change the three trainer groups in `grpo/__init__.py` so every symbol maps to
`"trainers"`. Replace direct imports in the listed examples and tests with:

```python
from src.model.post_training.grpo.trainers import (
    LLMGRPOConfig,
    LLMGRPOTrainer,
    SearchAgentGRPOTrainer,
)
```

Import only the names each call site uses. Update the DPO trainer docstring to
name `src.model.post_training.grpo.trainers`. Update the monkeypatch target in
`test_grpo_common.py` to
`src.model.post_training.grpo.trainers.SearchAgentGRPOTrainer.from_pretrained`.

- [ ] **Step 5: Delete the replaced trainer modules**

Use `apply_patch` to delete the three old files. Do not leave re-export stubs.

- [ ] **Step 6: Run focused trainer and import tests**

Run:

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_grpo_trainer.py \
  tests/unit/test_llm_grpo_trainer.py \
  tests/unit/test_search_agent_grpo_trainer.py \
  tests/unit/test_grpo_common.py \
  tests/unit/test_run_bamboogle_grpo_train.py
```

Expected: all selected tests pass and none import a deleted trainer module.

- [ ] **Step 7: Check formatting and commit**

Run: `ruff check src/model/post_training/grpo/trainers.py tests/unit/test_grpo_module_layout.py && ruff format --check src/model/post_training/grpo/trainers.py tests/unit/test_grpo_module_layout.py && git diff --check`

Commit:

```bash
git add src/model/post_training/grpo src/model/post_training/dpo/trainer.py examples tests/unit
git commit -m "refactor(grpo): consolidate trainer hierarchy"
```

---

### Task 2: Consolidate Training Orchestration

**Files:**
- Create: `src/model/post_training/grpo/training.py`
- Delete: `src/model/post_training/grpo/controller.py`
- Delete: `src/model/post_training/grpo/train_loop.py`
- Modify: `src/model/post_training/grpo/__init__.py`
- Modify: `src/__init__.py`
- Modify: `examples/run_retriever_aware_grpo.py`
- Modify: `tests/unit/test_train_loop.py`
- Modify: `tests/unit/test_grpo_module_layout.py`

**Interfaces:**
- Consumes: `PPOPolicyLossConfig`, trainer objects exposing `policy`, `optimizer`, `step`, `state_dict`, and `load_state_dict`, plus caller-supplied rollout callbacks.
- Produces: `RolloutResult`, `LocalGRPOController`, `TrainLoopConfig`, `save_checkpoint(trainer: Any, path: str, step: int) -> None`, `load_checkpoint(trainer: Any, path: str) -> int`, and `train_loop(...)` from `.training`.

- [ ] **Step 1: Extend the layout contract for orchestration**

Append:

```python
def test_consolidated_training_module_owns_orchestration():
    from src.model.post_training.grpo.training import (
        LocalGRPOController,
        TrainLoopConfig,
        train_loop,
    )

    assert LocalGRPOController.__module__ == "src.model.post_training.grpo.training"
    assert TrainLoopConfig.__module__ == "src.model.post_training.grpo.training"
    assert train_loop.__module__ == "src.model.post_training.grpo.training"


@pytest.mark.parametrize(
    "module_name",
    [
        "src.model.post_training.grpo.controller",
        "src.model.post_training.grpo.train_loop",
    ],
)
def test_replaced_training_modules_are_removed(module_name: str):
    assert find_spec(module_name) is None
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `pytest tests/unit/test_grpo_module_layout.py -v`

Expected: the new orchestration tests fail because `.training` is missing and the two old modules still exist.

- [ ] **Step 3: Build `training.py` and redirect consumers**

Create `training.py` with controller definitions first, followed by
`TrainLoopConfig`, checkpoint helpers, `train_loop`, and `_append_jsonl`. Preserve
the original implementations and imports. Update:

```python
# examples/run_retriever_aware_grpo.py and tests/unit/test_train_loop.py
from src.model.post_training.grpo.training import TrainLoopConfig, train_loop
```

Map `LocalGRPOController` and `RolloutResult` to `"training"` in
`grpo/__init__.py`. In `src/__init__.py`, point their lazy export module paths to
`.model.post_training.grpo.training` and change the section comment accordingly.

- [ ] **Step 4: Delete old orchestration modules**

Use `apply_patch` to delete `controller.py` and `train_loop.py`; do not retain
shim imports.

- [ ] **Step 5: Run focused orchestration and export tests**

Run:

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_train_loop.py \
  tests/unit/test_training_exports.py \
  tests/unit/test_grpo.py
```

Expected: all selected tests pass, including lazy package/root export checks.

- [ ] **Step 6: Check formatting and commit**

Run: `ruff check src/model/post_training/grpo/training.py tests/unit/test_grpo_module_layout.py && ruff format --check src/model/post_training/grpo/training.py tests/unit/test_grpo_module_layout.py && git diff --check`

Commit:

```bash
git add src/model/post_training/grpo src/__init__.py examples/run_retriever_aware_grpo.py tests/unit
git commit -m "refactor(grpo): consolidate training orchestration"
```

---

### Task 3: Fold Tensor Helpers Into Generation

**Files:**
- Modify: `src/model/post_training/grpo/generation.py`
- Delete: `src/model/post_training/grpo/tensor_helper.py`
- Modify: `src/__init__.py`
- Modify: `tests/unit/test_llm_agent_tensor_helper.py`
- Modify: `tests/unit/test_grpo_module_layout.py`

**Interfaces:**
- Consumes: PyTorch tensor primitives already imported by `generation.py`.
- Produces: `TensorConfig` and `TensorHelper` from `.generation`; root-level `src.TensorConfig` and `src.TensorHelper` remain lazy exports.

- [ ] **Step 1: Extend the layout contract for tensor ownership**

Append:

```python
def test_generation_owns_its_tensor_helpers():
    from src.model.post_training.grpo.generation import TensorConfig, TensorHelper

    assert TensorConfig.__module__ == "src.model.post_training.grpo.generation"
    assert TensorHelper.__module__ == "src.model.post_training.grpo.generation"
    assert find_spec("src.model.post_training.grpo.tensor_helper") is None
```

Change imports in `tests/unit/test_llm_agent_tensor_helper.py` from
`.tensor_helper` to `.generation`.

- [ ] **Step 2: Run the tensor tests and verify RED**

Run: `pytest -q tests/unit/test_grpo_module_layout.py tests/unit/test_llm_agent_tensor_helper.py`

Expected: failures because the classes still report the old module and the old
module spec exists.

- [ ] **Step 3: Move the two helper classes into `generation.py`**

Insert `TensorConfig` and `TensorHelper` after the imports and before their first
consumer. Copy their definitions exactly, remove:

```python
from .tensor_helper import TensorConfig, TensorHelper
```

and do not change generation behavior.

- [ ] **Step 4: Redirect root exports and delete the old helper module**

In `src/__init__.py`, map both helper names to
`.model.post_training.grpo.generation` and update the section comment. Delete
`tensor_helper.py` with `apply_patch`; do not add a shim.

- [ ] **Step 5: Run focused generation and export tests**

Run:

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_llm_agent_tensor_helper.py \
  tests/unit/test_llm_agent_generation.py \
  tests/unit/test_training_exports.py \
  tests/unit/test_qlearning.py
```

Expected: all selected tests pass; `test_qlearning.py` continues to prove that
package initialization does not introduce eager optional-dependency imports.

- [ ] **Step 6: Check formatting and commit**

Run: `ruff check src/model/post_training/grpo/generation.py tests/unit/test_grpo_module_layout.py tests/unit/test_llm_agent_tensor_helper.py && ruff format --check src/model/post_training/grpo/generation.py tests/unit/test_grpo_module_layout.py tests/unit/test_llm_agent_tensor_helper.py && git diff --check`

Commit:

```bash
git add src/model/post_training/grpo src/__init__.py tests/unit
git commit -m "refactor(grpo): colocate generation tensor helpers"
```

---

### Task 4: Enforce the Final Package Boundary

**Files:**
- Modify: `src/model/post_training/grpo/__init__.py`
- Modify: `tests/unit/test_grpo_module_layout.py`
- Modify: any live docstring or example still identified by the exact stale-path search below.

**Interfaces:**
- Consumes: consolidated `.trainers`, `.training`, and `.generation` modules from Tasks 1–3.
- Produces: a seven-module GRPO implementation surface plus lazy package/root exports with unchanged public names.

- [ ] **Step 1: Add the final literal module inventory test**

Append:

```python
from pathlib import Path


def test_grpo_package_has_only_the_approved_implementation_modules():
    package_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "model"
        / "post_training"
        / "grpo"
    )
    actual = {path.name for path in package_dir.glob("*.py")}
    assert actual == {
        "__init__.py",
        "core_algos.py",
        "generation.py",
        "judge.py",
        "plot_rollouts.py",
        "rollouts.py",
        "trainers.py",
        "training.py",
    }
```

- [ ] **Step 2: Run the inventory test and confirm its behavior**

Run: `pytest tests/unit/test_grpo_module_layout.py::test_grpo_package_has_only_the_approved_implementation_modules -v`

Expected: PASS only if Tasks 1–3 deleted all superseded modules and created both consolidated modules. To prove the test detects drift, temporarily restore one deleted filename, run and observe FAIL, then remove it and rerun to PASS.

- [ ] **Step 3: Remove all stale repository references**

Run:

```bash
rg -n 'grpo\.(grpo_trainer|llm_grpo_trainer|search_agent_grpo_trainer|controller|train_loop|tensor_helper)' src tests examples README.md docs
```

Expected: no matches except historical prose that explicitly documents the
completed migration. Update live docstrings/imports rather than suppressing
matches. Historical design and plan artifacts may describe old filenames but
must not instruct readers to import them.

- [ ] **Step 4: Verify lazy import behavior**

Run:

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_training_exports.py \
  tests/unit/test_qlearning.py \
  tests/unit/test_grpo.py
```

Expected: all tests pass and GRPO package import remains lazy.

- [ ] **Step 5: Run the complete GRPO-focused suite**

Run:

```bash
pytest -q \
  tests/unit/test_grpo.py \
  tests/unit/test_grpo_trainer.py \
  tests/unit/test_llm_grpo_trainer.py \
  tests/unit/test_search_agent_grpo_trainer.py \
  tests/unit/test_train_loop.py \
  tests/unit/test_reward_shapes.py \
  tests/unit/test_reward_human_signal.py \
  tests/unit/test_simulated_judge.py \
  tests/unit/test_gold_aware_judge.py \
  tests/unit/test_llm_agent_tensor_helper.py \
  tests/unit/test_llm_agent_generation.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Run repository verification**

Run:

```bash
ruff check src tests examples
ruff format --check src tests examples
git diff --check
pytest
```

Expected: Ruff and formatting exit 0, `git diff --check` emits no output, and
the full default suite reports zero failures and no new warnings caused by this
refactor. The isolated baseline was 3,332 passed, 3 skipped, and 73 warnings.

- [ ] **Step 7: Commit final boundary checks**

```bash
git add src tests examples README.md docs
git commit -m "test(grpo): enforce consolidated module layout"
```

- [ ] **Step 8: Request review before integration**

Use `superpowers:requesting-code-review` against the spec and the complete
commit range beginning at `de70f24`. Fix every Critical or Important finding,
rerun the relevant focused suite, and repeat the full verification command
before claiming completion.
