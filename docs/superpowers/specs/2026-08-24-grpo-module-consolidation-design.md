# GRPO Module Consolidation Design

## Goal

Reduce the number of narrowly split modules in `src/model/post_training/grpo`
without combining unrelated responsibilities or making the already large
`generation.py` harder to maintain. Preserve the package-level GRPO API while
removing obsolete direct module paths rather than retaining compatibility
shims.

## Current Structure

The package contains eleven implementation modules. Three trainer modules form
one inheritance hierarchy, two modules jointly implement training orchestration,
and `tensor_helper.py` is used only by `generation.py`. The remaining modules
have distinct responsibilities:

- `core_algos.py`: GRPO and REINFORCE loss functions.
- `rollouts.py`: prompt-group sampling, scoring, and on-policy batch assembly.
- `judge.py`: simulated, deterministic, and LLM judges.
- `generation.py`: search-agent generation and trajectory handling.
- `plot_rollouts.py`: standalone rollout visualization command.

`generation.py` is already more than 4,000 lines, so consolidation must not add
unrelated trainer or orchestration code to it.

## Target Structure

The package will contain these implementation files:

| Target | Responsibility | Sources |
| --- | --- | --- |
| `trainers.py` | Bandit, causal-LM, and search-agent GRPO trainers | `grpo_trainer.py`, `llm_grpo_trainer.py`, `search_agent_grpo_trainer.py` |
| `training.py` | Local controller, durable train loop, and checkpoint helpers | `controller.py`, `train_loop.py` |
| `generation.py` | Existing generation system plus its private tensor helper types | `generation.py`, `tensor_helper.py` |
| `core_algos.py` | GRPO/REINFORCE algorithm functions | unchanged |
| `rollouts.py` | Grouped rollout sampling and batch assembly | unchanged |
| `judge.py` | Reward judges | unchanged |
| `plot_rollouts.py` | Rollout visualization CLI | unchanged |
| `__init__.py` | Lazy package-level public exports | updated mappings |

The six replaced source files will be deleted. No deprecated shim modules will
remain.

## Import and API Policy

Package-level imports such as the following remain supported:

```python
from src.model.post_training.grpo import LLMGRPOTrainer
```

Direct imports through deleted modules intentionally break and all repository
call sites will move to the consolidated paths:

```python
from src.model.post_training.grpo.trainers import LLMGRPOTrainer
from src.model.post_training.grpo.training import TrainLoopConfig, train_loop
```

The root `src` lazy-export table, examples, tests, documentation, docstrings,
and patch targets will be updated so the repository contains no references to
deleted module paths. Class and function names, signatures, behavior, and
package-level export names remain unchanged.

## Internal Dependencies

`SearchAgentGRPOTrainer` will reference `LLMGRPOTrainer`, `LLMGRPOConfig`, and
`LLMRolloutResult` directly within `trainers.py`; the old cross-module import is
removed. Shared imports will be deduplicated at the top of the consolidated
file.

`training.py` will retain its dependencies on rollout and PPO types. The train
loop will use the colocated `LocalGRPOController` without a compatibility
import.

`TensorConfig` and `TensorHelper` will move near the beginning of
`generation.py`, before their first consumer. They remain importable from
`generation.py` but will not be advertised as package-level GRPO exports unless
they already were. Root lazy exports currently pointing to `tensor_helper` will
point to `generation` so existing `from src import TensorHelper` usage remains
stable.

Lazy loading in `grpo/__init__.py` remains load-bearing. Consolidation will not
replace it with eager imports because importing the package must not force
PyTorch or agent-loop dependencies into unrelated callers.

## Error Handling and Behavior

This is a structural refactor. Runtime behavior and error semantics must not
change. Consolidated code will keep the existing validation, exception types,
checkpoint format, rollout behavior, and trainer update semantics. Circular
imports introduced by consolidation are treated as defects and must be removed
rather than hidden behind runtime imports unless an existing lazy boundary
requires one.

## Testing

Implementation follows test-driven development:

1. Add import-contract tests for the new `trainers`, `training`, and
   `generation` locations and update existing tests to import those paths.
2. Verify the tests fail before the new modules exist or old paths are removed.
3. Consolidate one responsibility group at a time and run its focused tests.
4. Assert package-level and root-level lazy exports still resolve without eager
   dependency regressions.
5. Search the repository for references to each deleted module path.
6. Run Ruff, formatting checks, `git diff --check`, all GRPO-related tests, and
   the full default Pytest suite.

## Acceptance Criteria

- `grpo/` has four fewer implementation modules overall: six old modules are
  deleted and two consolidated modules are added, while tensor helpers move
  into the existing generation module.
- No compatibility shim files remain.
- No repository import or documentation reference targets a deleted module.
- Package-level GRPO exports and root `src` exports retain their current names.
- Trainer, rollout, judge, controller, train-loop, generation, and export tests
  pass.
- The full default test suite passes with no new failures or warnings caused by
  the refactor.

## Out of Scope

- Splitting or behaviorally redesigning `generation.py`.
- Changing GRPO algorithms, trainer behavior, checkpoint formats, or public
  symbol names.
- Moving GRPO functionality into PPO or other post-training packages.
- Providing deprecation shims for deleted direct module paths.
