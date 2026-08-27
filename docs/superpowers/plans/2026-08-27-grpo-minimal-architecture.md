# GRPO Minimal Architecture and Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GRPO to three coherent implementation modules, optimize its shared reward and online-training hot paths with measured evidence, and preserve numerical, API, checkpoint, and deterministic behavior.

**Architecture:** Consolidate loss/reward-flow code in `algorithms.py`, environment interaction in `generation.py`, and all optimization/orchestration in `training.py`; relocate the plotting CLI outside the library. Move response log-prob extraction to a neutral post-training utility used by DPO and GRPO, and keep dependencies acyclic as `training → generation → algorithms` with both higher layers allowed to consume the shared reward and log-prob utilities.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face model/tokenizer interfaces, asyncio, pytest, Ruff, `time.perf_counter_ns`, lazy import registries.

**Spec:** `docs/superpowers/specs/2026-08-27-grpo-minimal-architecture-design.md`

## Global Constraints

- Preserve every public name and call signature exported from `src` and `src.model.post_training.grpo`.
- Preserve reward component keys, weights, scalar totals, advantage values, sparse-token placement, loss reduction, rollout ordering, RNG restoration, and fixed-seed results.
- Preserve checkpoint keys, values, and load behavior.
- Keep `grpo/__init__.py` and `src/__init__.py` lazy; importing lightweight post-training code must not eagerly load the agent loop or optional model dependencies.
- Delete superseded direct modules without compatibility shim files.
- Final GRPO inventory is exactly `__init__.py`, `algorithms.py`, `generation.py`, and `training.py`.
- DPO and GRPO must import `get_response_log_probs` from `src.model.post_training/log_probs.py`.
- `algorithms.py` must not import `generation.py` or `training.py`; `generation.py` must not import `training.py`.
- Do not change the mathematical GRPO objective, hyperparameter defaults, judge prompts, sampling defaults, external interfaces, or reward policy.
- Accept runtime or memory optimizations only with repeatable before/after evidence and exact equivalence tests.
- Use `apply_patch` for source edits and run the listed RED test before each implementation step.

---

### Task 1: Establish the Final Boundary and Dependency Contracts

**Files:**
- Modify: `tests/unit/test_grpo_module_layout.py`
- Create: `tests/unit/test_grpo_dependency_direction.py`

**Interfaces:**
- Consumes: current package paths and lazy registries.
- Produces: literal contracts for the final four-file inventory, symbol ownership, removed module paths, and acyclic internal imports.

- [ ] **Step 1: Replace the current inventory expectation with the final target**

Add this contract while retaining the existing historical removed-module assertions:

```python
def test_grpo_package_has_only_the_minimal_implementation_modules():
    package_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "model"
        / "post_training"
        / "grpo"
    )
    assert {path.name for path in package_dir.glob("*.py")} == {
        "__init__.py",
        "algorithms.py",
        "generation.py",
        "training.py",
    }
```

- [ ] **Step 2: Add ownership and removed-path assertions**

```python
@pytest.mark.parametrize(
    "module_name",
    [
        "src.model.post_training.grpo.core_algos",
        "src.model.post_training.grpo.rollouts",
        "src.model.post_training.grpo.judge",
        "src.model.post_training.grpo.trainers",
        "src.model.post_training.grpo.plot_rollouts",
    ],
)
def test_second_stage_replaced_modules_are_removed(module_name: str):
    assert find_spec(module_name) is None


def test_representative_symbols_have_final_owners():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.algorithms import (
        GRPOAdvantageConfig,
        LLMJudge,
        compute_grpo_policy_loss,
        score_prompt_group,
    )
    from src.model.post_training.grpo.training import (
        LLMGRPOTrainer,
        LocalGRPOController,
        SearchAgentGRPOTrainer,
    )

    for value in (
        GRPOAdvantageConfig,
        LLMJudge,
        compute_grpo_policy_loss,
        score_prompt_group,
    ):
        assert value.__module__ == "src.model.post_training.grpo.algorithms"
    for value in (LLMGRPOTrainer, LocalGRPOController, SearchAgentGRPOTrainer):
        assert value.__module__ == "src.model.post_training.grpo.training"
```

- [ ] **Step 3: Add an AST dependency-direction test**

Create `tests/unit/test_grpo_dependency_direction.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / "src/model/post_training/grpo"


def _relative_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_grpo_dependency_direction_is_acyclic():
    assert "generation" not in _relative_imports("algorithms")
    assert "training" not in _relative_imports("algorithms")
    assert "training" not in _relative_imports("generation")
```

- [ ] **Step 4: Run the boundary tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_grpo_module_layout.py tests/unit/test_grpo_dependency_direction.py
```

Expected: FAIL because `algorithms.py` is absent, five superseded modules remain, and `generation.py` imports `.training`.

- [ ] **Step 5: Commit the RED contracts**

```bash
git add tests/unit/test_grpo_module_layout.py tests/unit/test_grpo_dependency_direction.py
git commit -m "test(grpo): define minimal architecture contracts"
```

---

### Task 2: Extract Shared Causal-LM Log-Probability Arithmetic

**Files:**
- Create: `src/model/post_training/log_probs.py`
- Modify: `src/model/post_training/dpo/trainer.py`
- Modify: `src/model/post_training/grpo/trainers.py`
- Modify: `src/model/post_training/grpo/__init__.py`
- Modify: `tests/unit/test_llm_grpo_trainer.py`
- Modify: `tests/unit/dpo/test_trainer.py`
- Create: `tests/unit/test_post_training_log_probs.py`

**Interfaces:**
- Consumes: causal-LM `model(input_ids).logits`, `(B, P+T)` token IDs, integer prompt length, and `(B, T)` response mask.
- Produces: `get_response_log_probs(model: nn.Module, full_ids: torch.Tensor, prompt_len: int, response_mask: torch.Tensor) -> torch.Tensor` in `src.model.post_training.log_probs`.

- [ ] **Step 1: Write token-alignment and shared-identity tests**

Create a deterministic next-token model in `tests/unit/test_post_training_log_probs.py` and assert the exact response slice:

```python
def test_response_log_probs_align_logits_with_the_tokens_they_predict():
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.log_probs import get_response_log_probs

    model = PositionLogitModel(vocab_size=7)
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    mask = torch.tensor([[1, 1, 0]])
    actual = get_response_log_probs(model, ids, prompt_len=2, response_mask=mask)
    expected = torch.log_softmax(model(ids).logits[:, 1:-1], dim=-1)
    expected = expected.gather(-1, ids[:, 2:].unsqueeze(-1)).squeeze(-1) * mask
    torch.testing.assert_close(actual, expected)


def test_dpo_and_grpo_use_the_shared_helper():
    from src.model.post_training import log_probs
    from src.model.post_training.dpo import trainer as dpo_trainer
    from src.model.post_training.grpo import trainers as grpo_trainers

    assert dpo_trainer.get_response_log_probs is log_probs.get_response_log_probs
    assert grpo_trainers.get_response_log_probs is log_probs.get_response_log_probs
```

Define `PositionLogitModel` in the test as an `nn.Module` returning a namespace whose deterministic `(B, L, V)` logits vary by position and vocabulary index.

- [ ] **Step 2: Run the shared-helper tests and verify RED**

Run: `pytest -q tests/unit/test_post_training_log_probs.py`

Expected: FAIL with `ModuleNotFoundError: src.model.post_training.log_probs`.

- [ ] **Step 3: Move the helper without changing arithmetic**

Create `log_probs.py` with the current implementation from `trainers.py`, including its input validation and exact shift:

```python
def get_response_log_probs(
    model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=full_ids).logits
    response_logits = logits[:, prompt_len - 1 : -1]
    response_ids = full_ids[:, prompt_len:]
    log_probs = torch.log_softmax(response_logits, dim=-1)
    selected = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return selected * response_mask.to(dtype=selected.dtype)
```

If the current helper has additional validation, copy it exactly before the forward pass.

- [ ] **Step 4: Redirect DPO and GRPO while retaining the lazy GRPO export**

Use this import in both trainers:

```python
from ..log_probs import get_response_log_probs
```

Map the GRPO lazy export to the neutral module without eagerly importing it:

```python
"get_response_log_probs": "..log_probs",
```

Update `grpo.__getattr__` so entries beginning with `..` resolve relative to the package, or use a `(module, symbol)` lazy-registry tuple consistently. Assert `from src.model.post_training.grpo import get_response_log_probs` preserves identity.

- [ ] **Step 5: Run shared, DPO, and GRPO log-prob tests**

Run:

```bash
pytest -q \
  tests/unit/test_post_training_log_probs.py \
  tests/unit/dpo/test_trainer.py \
  tests/unit/test_llm_grpo_trainer.py \
  tests/unit/test_training_exports.py
```

Expected: PASS.

- [ ] **Step 6: Check and commit**

```bash
ruff check src/model/post_training/log_probs.py src/model/post_training/dpo/trainer.py src/model/post_training/grpo tests/unit/test_post_training_log_probs.py
ruff format --check src/model/post_training/log_probs.py src/model/post_training/dpo/trainer.py src/model/post_training/grpo tests/unit/test_post_training_log_probs.py
git diff --check
git add src/model/post_training tests/unit
git commit -m "refactor(training): share response log-prob arithmetic"
```

---

### Task 3: Consolidate Algorithms, Rollouts, and Judges

**Files:**
- Create: `src/model/post_training/grpo/algorithms.py`
- Delete: `src/model/post_training/grpo/core_algos.py`
- Delete: `src/model/post_training/grpo/rollouts.py`
- Delete: `src/model/post_training/grpo/judge.py`
- Modify: `src/model/post_training/grpo/generation.py`
- Modify: `src/model/post_training/grpo/trainers.py`
- Modify: `src/model/post_training/grpo/__init__.py`
- Modify: `src/model/post_training/__init__.py`
- Modify: `src/__init__.py`
- Modify: `examples/_grpo_common.py`
- Modify: `examples/run_bamboogle_grpo_train.py`
- Modify: `examples/run_bamboogle_synthetic_grpo.py`
- Modify: `examples/run_retriever_aware_grpo.py`
- Modify: `tests/unit/test_gold_aware_judge.py`
- Modify: `tests/unit/test_grpo_common.py`
- Modify: `tests/unit/test_grpo_trainer.py`
- Modify: `tests/unit/test_llm_grpo_trainer.py`
- Modify: `tests/unit/test_reward_human_signal.py`
- Modify: `tests/unit/test_reward_shapes.py`
- Modify: `tests/unit/test_run_bamboogle_grpo_train.py`
- Modify: `tests/unit/test_search_agent_grpo_trainer.py`
- Modify: `tests/unit/test_simulated_judge.py`
- Modify: `tests/unit/test_grpo_module_layout.py`

**Interfaces:**
- Consumes: PPO loss primitives, `SearchRewardFunction`, `PromptBatch`, and `AgentLoopBase`.
- Produces: all existing public symbols from the three source modules at `src.model.post_training.grpo.algorithms` with unchanged signatures.

- [ ] **Step 1: Extend lazy-export identity tests for every moved public symbol**

Parameterize the existing layout test with the current public names from the three modules and assert each package/root export is identical to `algorithms.<name>`. Include at minimum:

```python
ALGORITHM_EXPORTS = [
    "compute_grpo_outcome_advantage",
    "compute_grpo_policy_loss",
    "compute_reinforce_policy_loss",
    "GRPOAdvantageConfig",
    "GRPORolloutSample",
    "ScoredGRPORollout",
    "sample_prompt_group",
    "score_prompt_group",
    "assemble_on_policy_batch",
    "LLMJudge",
    "SimulatedPreferenceJudge",
    "GoldAgreementJudge",
]
```

- [ ] **Step 2: Run ownership tests and verify RED**

Run: `pytest -q tests/unit/test_grpo_module_layout.py`

Expected: FAIL because `algorithms.py` does not exist and old owners remain.

- [ ] **Step 3: Build `algorithms.py` in dependency order**

Move definitions without behavior changes in this order:

```text
imports and type aliases
loss/advantage functions from core_algos.py
rollout dataclasses and sampling configuration
sampling functions
judge classes and parsing helpers
rollout scoring and advantage selection
on-policy filtering, assembly, and statistics
```

Replace former cross-module imports with direct references. Keep imports from `reward.py`, PPO, data, and agent base explicit; do not import generation or training.

- [ ] **Step 4: Redirect imports and lazy registries**

Change imports in every file listed for this task from `.core_algos`,
`.rollouts`, and `.judge` to `.algorithms`. Point all corresponding entries in
`grpo/__init__.py` and `src/__init__.py` to `algorithms` while preserving lazy
identity.

- [ ] **Step 5: Delete the three superseded modules**

Delete `core_algos.py`, `rollouts.py`, and `judge.py` with `apply_patch`. Do not leave re-export stubs.

- [ ] **Step 6: Run the complete algorithm/reward/judge suite**

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_grpo_dependency_direction.py \
  tests/unit/test_grpo.py \
  tests/unit/test_reward.py \
  tests/unit/test_reward_shapes.py \
  tests/unit/test_reward_human_signal.py \
  tests/unit/test_simulated_judge.py \
  tests/unit/test_gold_aware_judge.py
```

Expected: PASS.

- [ ] **Step 7: Check stale paths and commit**

```bash
rg -n 'grpo\.(core_algos|rollouts|judge)' src tests examples README.md docs/training-and-evaluation.md
ruff check src tests examples
ruff format --check src tests examples
git diff --check
git add src tests examples docs/training-and-evaluation.md
git commit -m "refactor(grpo): consolidate algorithms and scoring"
```

Expected stale-path search: no live import or current documentation matches.

---

### Task 4: Consolidate Trainers and Remove the Generation Cycle

**Files:**
- Modify: `src/model/post_training/grpo/training.py`
- Modify: `src/model/post_training/grpo/generation.py`
- Delete: `src/model/post_training/grpo/trainers.py`
- Modify: `src/model/post_training/grpo/__init__.py`
- Modify: `src/__init__.py`
- Modify: `src/model/post_training/dpo/trainer.py`
- Modify: live imports and patch targets under `tests/` and `examples/`
- Modify: `tests/unit/test_grpo_module_layout.py`
- Modify: `tests/unit/test_grpo_dependency_direction.py`

**Interfaces:**
- Consumes: `algorithms.py`, `generation.py`, shared log-prob helper, PPO primitives, and caller-injected policy/tokenizer/optimizer.
- Produces: `Policy`, `GRPOTrainer`, `LLMGRPOConfig`, `LLMGRPOTrainer`, `SearchAgentGRPOTrainer`, `LocalGRPOController`, `TrainLoopConfig`, checkpoints, and `train_loop` from `.training`.

- [ ] **Step 1: Add final trainer-ownership and no-cycle tests**

Assert every symbol formerly owned by `trainers.py` resolves by identity from `training.py`. Retain the AST assertion that `generation.py` has no relative import of `training`, including imports inside functions and methods.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_grpo_dependency_direction.py
```

Expected: FAIL because trainer symbols still belong to `trainers.py` and two deferred `.training` imports remain in `generation.py`.

- [ ] **Step 3: Move the trainer hierarchy into `training.py`**

Order the consolidated module as:

```text
imports and shared tensor helpers
Policy and bandit GRPOTrainer
LLMGRPOConfig, LLMRolloutResult, and LLMGRPOTrainer
SearchAgentGRPOTrainer
RolloutResult and LocalGRPOController
TrainLoopConfig, checkpoint helpers, train_loop
```

Deduplicate imports and point all algorithm symbols to `.algorithms` and log probabilities to `..log_probs`.

- [ ] **Step 4: Invert the controller dependency without duplicating orchestration**

Extract the controller-independent body used by `LocalGRPOController.training_step` and `async_training_step` into generation-owned lower-level functions accepting all backends and optimizer explicitly. Make the controller delegate to those functions. Update `LLMGenerationManager.run_grpo_training_step` and `async_run_grpo_training_step` to call the same lower-level functions directly, preserving signatures and results.

The resulting direction must be:

```python
# training.py
from .generation import _run_grpo_training_step_core

# generation.py: no import from .training anywhere
```

- [ ] **Step 5: Redirect consumers and delete `trainers.py`**

Change DPO to retain its neutral `..log_probs` import. Move all GRPO trainer imports and lazy mappings to `.training`, update monkeypatch strings, then delete `trainers.py` without a shim.

- [ ] **Step 6: Run trainer, controller, generation, DPO, and cycle tests**

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_grpo_dependency_direction.py \
  tests/unit/test_grpo_trainer.py \
  tests/unit/test_llm_grpo_trainer.py \
  tests/unit/test_search_agent_grpo_trainer.py \
  tests/unit/test_train_loop.py \
  tests/unit/test_llm_agent_generation.py \
  tests/unit/dpo/test_trainer.py \
  tests/unit/test_training_exports.py
```

Expected: PASS.

- [ ] **Step 7: Check and commit**

```bash
rg -n 'grpo\.trainers|from \.training import' src/model/post_training/grpo src tests examples
ruff check src tests examples
ruff format --check src tests examples
git diff --check
git add src tests examples
git commit -m "refactor(grpo): consolidate training and orchestration"
```

Expected: no `grpo.trainers` match and no `.training` import in `generation.py`.

---

### Task 5: Relocate the Rollout Plotting CLI

**Files:**
- Create: `examples/plot_grpo_rollouts.py`
- Delete: `src/model/post_training/grpo/plot_rollouts.py`
- Create: `tests/unit/test_plot_grpo_rollouts.py`
- Modify: `docs/training-and-evaluation.md`
- Modify: `tests/unit/test_grpo_module_layout.py`

**Interfaces:**
- Consumes: rollout JSONL paths and output HTML path.
- Produces: `python -m examples.plot_grpo_rollouts INPUT --output OUTPUT` with unchanged HTML rendering.

- [ ] **Step 1: Add relocated-module and CLI-output tests**

Update plotting tests to import `examples.plot_grpo_rollouts`. Add a subprocess/parser-level assertion that a temporary one-record JSONL produces HTML containing the escaped question, answer, reward, and trajectory fields.

- [ ] **Step 2: Run the plotting and inventory tests and verify RED**

Run: `pytest -q tests/unit/test_grpo_module_layout.py tests/unit/test_plot_grpo_rollouts.py`

Expected: FAIL because the example module is absent and the package module remains.

- [ ] **Step 3: Move the CLI implementation unchanged**

Copy the implementation into `examples/plot_grpo_rollouts.py`, retaining `main()` and the `if __name__ == "__main__": main()` entrypoint. Redirect documentation commands.

- [ ] **Step 4: Delete the package CLI and run tests**

Delete `src/model/post_training/grpo/plot_rollouts.py`, then run:

```bash
pytest -q tests/unit/test_grpo_module_layout.py tests/unit/test_plot_grpo_rollouts.py
```

Expected: PASS and the literal package inventory equals four files.

- [ ] **Step 5: Commit**

```bash
git add src/model/post_training/grpo examples tests docs README.md
git commit -m "refactor(grpo): move rollout plotting to examples"
```

---

### Task 6: Establish Reproducible Performance Baselines

**Files:**
- Create: `examples/benchmark_grpo_optimization.py`
- Create: `tests/unit/test_benchmark_grpo_optimization.py`
- Create: `docs/benchmarks/grpo-optimization-baseline.md`

**Interfaces:**
- Consumes: deterministic fixture sizes and callable benchmark cases.
- Produces: `measure_case(name: str, fn: Callable[[], object], warmup: int, iterations: int) -> BenchmarkResult` plus JSON/Markdown-friendly median and allocation metrics.

- [ ] **Step 1: Write failing benchmark-harness tests**

```python
def test_measure_case_runs_warmups_outside_recorded_samples():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1

    result = measure_case("operation", operation, warmup=2, iterations=5)
    assert calls == 7
    assert result.name == "operation"
    assert len(result.samples_ns) == 5
    assert result.median_ns > 0


def test_cli_json_contains_reproducibility_metadata(tmp_path):
    payload = run_smoke_benchmarks(iterations=3)
    assert payload["python_version"]
    assert payload["torch_version"]
    assert payload["iterations"] == 3
    assert {case["name"] for case in payload["cases"]} >= {
        "group_advantages",
        "reward_batch",
        "training_batch_assembly",
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/unit/test_benchmark_grpo_optimization.py`

Expected: FAIL because the benchmark module does not exist.

- [ ] **Step 3: Implement a deterministic diagnostic harness**

Use `time.perf_counter_ns()` for repeated wall-clock samples and `tracemalloc` for Python peak allocations. Seed Python and Torch, use stub models/judges/retrievers, and include cases for group advantages, token-F1/composite reward, batch reward/sparse advantages, on-policy assembly, log-prob evaluation, generation batch assembly, and async rollout collection. Do not assert absolute time in tests.

- [ ] **Step 4: Record the pre-optimization baseline**

Run:

```bash
python -m examples.benchmark_grpo_optimization --warmup 5 --iterations 25 --output docs/benchmarks/grpo-optimization-baseline.md
```

Record commit hash, hardware/device, Python/Torch versions, fixture dimensions, median time, and peak Python allocation for every case.

- [ ] **Step 5: Verify and commit**

```bash
pytest -q tests/unit/test_benchmark_grpo_optimization.py
git diff --check
git add examples/benchmark_grpo_optimization.py tests/unit/test_benchmark_grpo_optimization.py docs/benchmarks/grpo-optimization-baseline.md
git commit -m "perf(grpo): establish optimization baselines"
```

---

### Task 7: Optimize Reward and Advantage Computation

**Files:**
- Modify: `src/model/post_training/reward.py`
- Modify: `src/model/post_training/grpo/algorithms.py`
- Modify: `tests/unit/test_reward.py`
- Modify: `tests/unit/test_reward_shapes.py`
- Create: `tests/unit/test_reward_performance_contracts.py`
- Modify: `docs/benchmarks/grpo-optimization-baseline.md`

**Interfaces:**
- Consumes: answer/gold pairs, optional batch judge, agent outputs, group IDs, and `SearchRewardConfig`.
- Produces: one dependency-light `compute_group_relative_advantages(rewards: Sequence[float], group_ids: Sequence[Hashable], *, normalize: bool, epsilon: float = 1e-8, clip_range: tuple[float, float] | None = None) -> list[float]` in `reward.py`, plus existing public reward methods as compatible adapters.

- [ ] **Step 1: Write equivalence and invocation-count tests**

Cover all reward presets and assert:

```python
def test_batch_scoring_invokes_judge_once_per_answer():
    calls = 0

    def judge(answer: str, gold: str) -> float:
        nonlocal calls
        calls += 1
        return float(answer == gold)

    scored = score_prompt_group(samples, ground_truth="gold", judge_fn=judge)
    assert calls == len(samples)
    assert [item.reward for item in scored] == expected_rewards


def test_disabled_evidence_dimensions_do_not_traverse_context(monkeypatch):
    reward = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    monkeypatch.setattr(reward, "_evidence_components", forbidden)
    assert reward.compute(output, "gold", judge) == expected_terminal_reward
```

Also parameterize scalar totals, exact breakdown dictionaries, last-action sparse placement, group isolation, zero variance, clipping, and output order.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/unit/test_reward_performance_contracts.py`

Expected: at least the disabled-component fast-path and canonical-kernel ownership assertions FAIL.

- [ ] **Step 3: Add the canonical group kernel and redirect adapters**

Implement one grouping pass in `reward.py`: validate equal lengths, collect `(index, float_reward)` by group, compute mean and optional population standard deviation once per group, apply epsilon and optional clipping, and write results into a preallocated output list. Make `SearchRewardFunction.compute_grpo_outcome_advantages`, `compute_batch_advantages`, and `algorithms.py` list adapters delegate to it.

- [ ] **Step 4: Consolidate correctness and sparse-vector work**

Add private helpers that accept pre-scored correctness and build terminal vectors from `(length, last_action_index, scalar)`. Route batch sparse rewards and outcome token advantages through them so judges and group statistics are not repeated.

- [ ] **Step 5: Skip provably unused scalar-path components**

Split scalar computation from full breakdown construction. `reward_components()` must still return every current key; `compute()` and sparse terminal paths may bypass evidence/citation/search calculations whose configured weights are zero. Reuse answer text, URL sets, evidence deltas, and dimension grouping inside one breakdown call.

- [ ] **Step 6: Run reward and GRPO numerical suites**

```bash
pytest -q \
  tests/unit/test_reward.py \
  tests/unit/test_reward_shapes.py \
  tests/unit/test_reward_human_signal.py \
  tests/unit/test_reward_performance_contracts.py \
  tests/unit/test_grpo.py \
  tests/unit/test_simulated_judge.py \
  tests/unit/test_gold_aware_judge.py
```

Expected: PASS with exact expected dictionaries and numerical values.

- [ ] **Step 7: Rerun benchmarks and retain only evidenced improvements**

Run the same command and fixture sizes as Task 6, append an `After reward optimization` table, and calculate percentage changes. Revert any optimization that shows no repeatable improvement across three runs unless it materially reduces duplication and has no regression.

- [ ] **Step 8: Commit**

```bash
ruff check src/model/post_training/reward.py src/model/post_training/grpo/algorithms.py tests/unit/test_reward_performance_contracts.py
ruff format --check src/model/post_training/reward.py src/model/post_training/grpo/algorithms.py tests/unit/test_reward_performance_contracts.py
git diff --check
git add src/model/post_training/reward.py src/model/post_training/grpo/algorithms.py tests/unit docs/benchmarks/grpo-optimization-baseline.md
git commit -m "perf(grpo): streamline reward and advantage computation"
```

---

### Task 8: Optimize Online Rollout and Policy-Update Hot Paths

**Files:**
- Modify: `src/model/post_training/grpo/generation.py`
- Modify: `src/model/post_training/grpo/training.py`
- Modify: `src/model/post_training/log_probs.py`
- Modify: GRPO trainer/generation tests under `tests/unit/`
- Create: `tests/unit/test_grpo_performance_contracts.py`
- Modify: `docs/benchmarks/grpo-optimization-baseline.md`

**Interfaces:**
- Consumes: current-policy rollouts, optional reference policy/backend, scored trajectories, and optimizer.
- Produces: unchanged rollout/training results with fewer model forwards, conversions, allocations, or coroutine objects where benchmarks support the change.

- [ ] **Step 1: Add model-forward, inference-mode, and tensor-equivalence contracts**

Use counting stub models to assert reference inference is absent when
KL/reference outputs are not required and occurs exactly once when required.
The reference stub records `torch.is_inference_mode_enabled()` during its
forward and the test asserts `[True]`. Assert optimized batch assembly tensors
equal fixtures for prompts, responses, attention mask, info mask, advantages,
old log probabilities, reference log probabilities, group IDs, and rollout
indices.

- [ ] **Step 2: Add concurrency and ordering contracts**

Use an async stub loop with controlled completion order. Assert returned groups remain prompt order then rollout-index order, maximum simultaneous calls never exceeds the configured bound, and empty/singleton batches do not create unnecessary worker pools.

- [ ] **Step 3: Run the new contracts and verify RED**

Run: `pytest -q tests/unit/test_grpo_performance_contracts.py`

Expected: FAIL because current reference evaluation uses `torch.no_grad()` and
the recording stub observes inference mode as disabled.

- [ ] **Step 4: Optimize measured policy/reference inference**

Wrap frozen reference work in `torch.inference_mode()`, retain conditional reference evaluation, and reuse full IDs, attention/response masks, and old log probabilities within a step. Do not cache tensors across optimizer steps.

- [ ] **Step 5: Optimize measured batch assembly**

Replace repeated nested list padding and list-to-tensor conversions with `torch.nn.utils.rnn.pad_sequence`, preallocated tensors, or shared masks only where the equivalence fixtures remain exact. Preserve left-padding for prompts and right-padding for responses.

- [ ] **Step 6: Optimize measured rollout scheduling**

Fast-path empty and singleton inputs, preserve the semaphore bound, avoid nested group-level oversubscription, and retain explicit sorting by rollout index after concurrent completion.

- [ ] **Step 7: Run focused behavioral suites**

```bash
pytest -q \
  tests/unit/test_grpo_performance_contracts.py \
  tests/unit/test_llm_grpo_trainer.py \
  tests/unit/test_search_agent_grpo_trainer.py \
  tests/unit/test_llm_agent_generation.py \
  tests/unit/test_train_loop.py \
  tests/unit/test_post_training_log_probs.py
```

Expected: PASS, including fixed-seed and ordering assertions.

- [ ] **Step 8: Rerun benchmarks and retain only evidenced improvements**

Append `After online-path optimization` results using the exact Task 6 environment and fixture sizes. State model-forward-count changes separately from timing. Revert changes with a repeatable regression unless required for the acyclic architecture.

- [ ] **Step 9: Commit**

```bash
ruff check src/model/post_training/grpo src/model/post_training/log_probs.py tests/unit/test_grpo_performance_contracts.py
ruff format --check src/model/post_training/grpo src/model/post_training/log_probs.py tests/unit/test_grpo_performance_contracts.py
git diff --check
git add src/model/post_training/grpo src/model/post_training/log_probs.py tests/unit docs/benchmarks/grpo-optimization-baseline.md
git commit -m "perf(grpo): optimize online rollout and update paths"
```

---

### Task 9: Documentation, Full Verification, Review, and PR

**Files:**
- Modify: `docs/training-and-evaluation.md`
- Modify: `examples/_grpo_common.py`
- Modify: `examples/run_bamboogle_grpo_train.py`
- Modify: `examples/run_bamboogle_synthetic_grpo.py`
- Modify: `examples/run_retriever_aware_grpo.py`
- Modify: `src/model/post_training/dpo/trainer.py`
- Modify: `src/model/post_training/grpo/generation.py`
- Modify: `src/model/post_training/grpo/training.py`
- Modify: `docs/benchmarks/grpo-optimization-baseline.md`
- Modify: `tests/unit/test_grpo_module_layout.py`

**Interfaces:**
- Consumes: completed three-module GRPO architecture and benchmark evidence.
- Produces: current documentation, a verified commit range, review findings resolved, pushed branch, and pull request.

- [ ] **Step 1: Update architecture and usage documentation**

Document:

```text
algorithms.py — GRPO math, grouped rollout scoring, judges, on-policy batches
generation.py — live environment interaction and trajectory/tensor assembly
training.py — trainers, controller, checkpointing, durable loop
reward.py — shared shaped reward computation optimized for batched GRPO use
log_probs.py — shared causal-LM response-token alignment for DPO and GRPO
examples/plot_grpo_rollouts.py — visualization CLI
```

Include one offline-prompt/on-policy-generation example and one live-search online example. Explain that GRPO uses the current policy for fresh rollouts, rollout-time old log probabilities for clipping, and a frozen reference for optional KL regularization.

- [ ] **Step 2: Run exact stale-path and inventory checks**

```bash
rg -n 'grpo\.(core_algos|rollouts|judge|trainers|plot_rollouts)' src tests examples README.md docs/training-and-evaluation.md
find src/model/post_training/grpo -maxdepth 1 -type f -name '*.py' -print | sort
```

Expected: no live stale-path matches and exactly four Python files in the package.

- [ ] **Step 3: Run focused verification**

```bash
pytest -q \
  tests/unit/test_grpo_module_layout.py \
  tests/unit/test_grpo_dependency_direction.py \
  tests/unit/test_post_training_log_probs.py \
  tests/unit/test_grpo.py \
  tests/unit/test_grpo_trainer.py \
  tests/unit/test_llm_grpo_trainer.py \
  tests/unit/test_search_agent_grpo_trainer.py \
  tests/unit/test_train_loop.py \
  tests/unit/test_reward.py \
  tests/unit/test_reward_shapes.py \
  tests/unit/test_reward_human_signal.py \
  tests/unit/test_reward_performance_contracts.py \
  tests/unit/test_grpo_performance_contracts.py \
  tests/unit/test_simulated_judge.py \
  tests/unit/test_gold_aware_judge.py \
  tests/unit/test_llm_agent_generation.py \
  tests/unit/dpo/test_trainer.py \
  tests/unit/test_training_exports.py \
  tests/unit/test_plot_grpo_rollouts.py \
  tests/unit/test_benchmark_grpo_optimization.py
```

Expected: PASS.

- [ ] **Step 4: Run repository verification**

```bash
ruff check src tests examples
ruff format --check src tests examples
git diff --check
pytest
```

Expected: all commands exit zero with no new warning introduced by this work.

- [ ] **Step 5: Commit documentation and final contracts**

```bash
git add src tests examples docs README.md
git commit -m "docs(grpo): document minimal optimized architecture"
```

- [ ] **Step 6: Request a code review against the complete range**

Use `superpowers:requesting-code-review` with base commit `29720fd`, the current HEAD, and the design acceptance criteria. Fix every Critical or Important finding, rerun the focused suite, and rerun the full verification commands before continuing.

- [ ] **Step 7: Create the delivery branch and PR**

If still on `main`, create a non-destructive feature branch before pushing:

```bash
git switch -c refactor/grpo-minimal-architecture
git push -u origin refactor/grpo-minimal-architecture
gh pr create \
  --title "refactor(grpo): consolidate and optimize online training" \
  --body-file docs/superpowers/context-packs/grpo-minimal-architecture-pr.md
```

Create the PR body file with summary, exact architecture changes, reward/online optimization evidence, compatibility guarantees, focused/full test results, and benchmark tables copied from `docs/benchmarks/grpo-optimization-baseline.md`. Report the PR URL.
