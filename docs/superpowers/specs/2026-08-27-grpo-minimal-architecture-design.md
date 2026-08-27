# GRPO Minimal Architecture and Optimization Design

## Goal

Reduce `src/model/post_training/grpo` from seven implementation modules to the
smallest practical architecture while simplifying duplicated GRPO paths and
improving runtime and memory efficiency where benchmarks demonstrate a gain.
Preserve public package/root exports, training semantics, deterministic
behavior, and checkpoint compatibility.

## Current Structure

The package currently contains these implementation modules:

| Module | Lines | Responsibility |
| --- | ---: | --- |
| `core_algos.py` | 172 | GRPO and REINFORCE loss wrappers |
| `rollouts.py` | 579 | grouped sampling, scoring, advantages, and on-policy batches |
| `judge.py` | 283 | simulated, deterministic, and LLM judges |
| `generation.py` | 4,453 | agent interaction, retrieval, trajectories, and tensor assembly |
| `trainers.py` | 1,063 | bandit, causal-LM, and search-agent trainers |
| `training.py` | 519 | local controller, checkpointing, and durable train loop |
| `plot_rollouts.py` | 211 | standalone HTML visualization CLI |

Several boundaries are narrower than their responsibilities warrant:

- Algorithm and reward-flow logic is split across `core_algos.py`,
  `rollouts.py`, and `judge.py`.
- Trainer optimization and controller orchestration are split across
  `trainers.py` and `training.py`.
- `plot_rollouts.py` is a command-line utility rather than a library concern.
- Group-relative advantages and rollout scoring have overlapping entry points
  across the package.
- `generation.py` and `training.py` use deferred imports in both directions,
  obscuring the intended dependency graph.

## Target Structure

The GRPO package will contain three implementation modules:

| Target | Responsibility | Sources |
| --- | --- | --- |
| `algorithms.py` | losses, advantages, rollout sampling/scoring, judges, and on-policy batch assembly | `core_algos.py`, `rollouts.py`, `judge.py` |
| `generation.py` | agent interaction, retrieval, trajectories, rollout execution, and generation tensor assembly | existing `generation.py` |
| `training.py` | bandit/LM/search trainers, local controller, policy updates, checkpoints, and durable train loop | `trainers.py`, existing `training.py` |

`plot_rollouts.py` will move to `examples/plot_grpo_rollouts.py`. The package
will therefore contain only `__init__.py` plus the three implementation files.

The dependency direction is:

```text
training ───────→ generation ───────→ algorithms
    └───────────────────────────────→ algorithms
```

`algorithms.py` must not import either higher layer. `generation.py` must not
import `training.py`. The controller-facing generation interfaces currently
responsible for the reverse import will be expressed through caller-supplied
objects or definitions owned by the lower layer, without changing their public
signatures.

## Public API and Compatibility

Package-level imports and root lazy exports retain their existing names:

```python
from src.model.post_training.grpo import LLMGRPOTrainer
from src import GRPOAdvantageConfig, LocalGRPOController
```

The lazy registries in `grpo/__init__.py` and `src/__init__.py` will point to
the new owners. Lazy loading remains mandatory so lightweight consumers do not
eagerly import PyTorch, the agent loop, or optional model dependencies.

Repository imports will move to the consolidated paths:

```python
from src.model.post_training.grpo.algorithms import GRPOAdvantageConfig
from src.model.post_training.grpo.training import SearchAgentGRPOTrainer
```

The deleted `core_algos`, `rollouts`, `judge`, and `trainers` module paths will
not receive compatibility shims. The plotting command's supported path becomes
`python -m examples.plot_grpo_rollouts`.

The refactor preserves:

- public names and call signatures;
- exception classes and validation semantics;
- reward values and advantage normalization;
- loss reduction and gradient scaling;
- rollout ordering and RNG restoration;
- checkpoint keys, values, and load behavior;
- deterministic results for fixed inputs and seeds.

## Algorithm and Scoring Simplification

`algorithms.py` will own one canonical group-relative advantage kernel. Thin
adapters may accept Python sequences or tensors, but there will be one source
of truth for centering, optional standard-deviation normalization, epsilon,
clipping, DAPO behavior, and zero-variance behavior.

Rollout scoring will follow one explicit flow:

1. Validate group shape and identifiers.
2. Evaluate batched judge/correctness signals when available.
3. Compute reward components once per rollout.
4. Select the configured scalar reward component.
5. Compute group advantages through the canonical kernel.
6. Return scored immutable records in rollout order.

Generation-specific records will adapt to this pipeline instead of
reimplementing group scoring. Existing public convenience functions remain as
thin, tested entry points.

## Training Simplification

`training.py` will place the trainer hierarchy before controller and durable
loop definitions. Causal-LM and search-agent trainers will share internal
helpers for behavior that is already identical, including response masks,
sparse terminal reward placement, log-prob extraction, reference-policy
handling, and policy-update bookkeeping.

The local controller will consume generation results through stable data
interfaces and call the shared policy-update path. Checkpoint helpers remain at
the orchestration boundary and keep the current serialization format.

No attempt will be made to force bandit and language-model optimization through
one abstraction where tensor shapes or optimizer semantics genuinely differ.

## Performance and Memory Optimization

Optimization is evidence-driven. Each changed hot path requires a focused
benchmark or profiler measurement before and after the change.

Candidate improvements are:

- Keep reference-policy inference conditional and execute it under
  `torch.inference_mode()`.
- Reuse tokenization, response masks, attention masks, and padded tensors
  within one optimization step.
- Avoid repeated Python-list-to-tensor conversions during generation batch
  assembly when vectorized padding produces equivalent values.
- Batch correctness/judge work and reward-component calculation where the
  existing interfaces permit it.
- Avoid rebuilding rollout-index maps and group statistics more than once per
  scoring pass.
- Improve action-mask construction only if profiling shows token decoding and
  regular-expression offset mapping is material.
- Preserve bounded concurrency while avoiding unnecessary coroutine/task
  construction for empty or singleton batches.

An optimization will not land solely because it appears faster. It must show a
repeatable improvement in wall-clock time, peak allocation, model-forward
count, or tensor-construction count and pass equivalence tests.

Representative benchmarks cover:

- list and tensor group-advantage calculation;
- prompt-group scoring and on-policy batch assembly;
- causal-LM policy/reference log-prob evaluation;
- generation training-batch tensor assembly;
- concurrent prompt rollout collection with stubbed I/O.

Benchmarks will use warmup plus repeated samples and report inputs, iteration
count, median time, and the observed improvement. They are diagnostic and will
not introduce brittle absolute-time assertions into the default test suite.

## Validation and Error Handling

Public validation behavior remains compatible. Internal boundaries will fail
early for malformed inputs, including mismatched group lengths, rollout-index
collisions, invalid advantage modes, and empty data where the existing public
operation cannot produce a meaningful result.

Existing public exception types and message-stable tests remain authoritative.
New internal validation may use `ValueError` with actionable context, but it
must not replace a documented public exception type.

## Migration Sequence

1. Add a literal package-inventory contract for the three-module target and
   ownership tests for representative public symbols.
2. Create `algorithms.py`, redirect imports/exports, and delete
   `core_algos.py`, `rollouts.py`, and `judge.py`.
3. Merge `trainers.py` into `training.py`, remove the reverse
   generation/training dependency, redirect imports/exports, and delete
   `trainers.py`.
4. Move the plotting utility to `examples/plot_grpo_rollouts.py` and delete the
   package copy.
5. Establish focused performance baselines.
6. Simplify duplicate advantage, scoring, update, and tensor-assembly paths one
   at a time with equivalence tests.
7. Apply only benchmark-supported runtime or memory improvements.
8. Update documentation, examples, monkeypatch targets, and lazy registries.
9. Search for stale module paths and run focused and full verification.

## Testing

Implementation follows test-driven development:

- Write module-layout and ownership tests before moving implementations.
- Run focused tests after each deleted module and redirected dependency.
- Add numerical equivalence tests for advantages, rewards, losses, masks, and
  batch tensors.
- Round-trip existing and newly saved checkpoints through the consolidated
  trainer/controller paths.
- Assert rollout order and fixed-seed behavior.
- Test lazy package and root exports in processes where optional dependencies
  are intentionally unavailable.
- Test the relocated plotting CLI and its HTML output.
- Record benchmark evidence for each accepted optimization.
- Run Ruff, formatting, `git diff --check`, the complete GRPO-focused suite,
  and the full default Pytest suite.

## Acceptance Criteria

- `src/model/post_training/grpo` contains only `__init__.py`,
  `algorithms.py`, `generation.py`, and `training.py`.
- `plot_rollouts.py` is available from `examples/plot_grpo_rollouts.py`.
- No live repository import or documentation targets a deleted module.
- Package-level and root exports preserve their current names and laziness.
- There is one canonical group-advantage implementation and one rollout
  scoring pipeline.
- The generation/training dependency is acyclic.
- Checkpoint, numerical, ordering, seed, and error-semantics tests pass.
- Every performance claim is backed by recorded before/after evidence.
- Focused and full repository verification complete without new failures or
  warnings caused by the consolidation.

## Out of Scope

- Changing the mathematical GRPO objective or default hyperparameters.
- Changing reward weights, judge prompts, sampling defaults, or rollout policy.
- Redesigning external agent, retriever, model, or storage interfaces.
- Altering checkpoint formats or providing compatibility shim modules.
- Splitting `generation.py` into a new subpackage; the objective is the minimum
  practical module count.
