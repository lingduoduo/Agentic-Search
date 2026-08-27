# GRPO Minimal Architecture and Optimization Design

## Goal

Reduce `src/model/post_training/grpo` from seven implementation modules to the
smallest practical architecture while simplifying duplicated GRPO paths and
improving runtime and memory efficiency where benchmarks demonstrate a gain.
The optimization scope includes the shared
`src/model/post_training/reward.py` layer used by GRPO. Preserve public
package/root exports, reward outputs, training semantics, deterministic
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
The shared `src/model/post_training/reward.py` module remains outside the GRPO
package because evaluation and other post-training methods also consume it; it
is nevertheless part of this refactor's profiling and optimization scope.
The causal-LM response log-probability helper currently owned by GRPO will move
to `src/model/post_training/log_probs.py`. DPO and GRPO will both consume this
neutral utility; DPO must not depend on a GRPO trainer module for shared token
alignment arithmetic.

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

Shared causal-LM code will use:

```python
from src.model.post_training.log_probs import get_response_log_probs
```

The deleted `core_algos`, `rollouts`, `judge`, and `trainers` module paths will
not receive compatibility shims. The plotting command's supported path becomes
`python -m examples.plot_grpo_rollouts`, keeping its existing `--jsonl` /
`--out` / `--max_records` / `--title` flags.

### The two `compute_grpo_outcome_advantage` functions

`core_algos.py` and `rollouts.py` each export a public function under this name,
and they are not duplicates: the first takes token-level reward tensors with an
eos mask and returns tensors; the second takes `list[float]` and returns scalar
per-rollout advantages. They are also exported from *different* places —
`src.model.post_training.grpo` re-exports the tensor form, while the `src` root
re-exports the list form — so both names are load-bearing public API and neither
may be renamed.

Merging both owners into one `algorithms.py` therefore requires a decision the
migration cannot avoid. `algorithms.py` keeps a single public
`compute_grpo_outcome_advantage` that dispatches on whether its first argument
is a tensor, with a typed overload for each call shape, delegating to two
private implementations. Both existing call contracts keep working unchanged.
The list form delegates to the canonical kernel described below; the tensor form
retains its vectorized path and is pinned to the same numerical contract.

The refactor preserves:

- public names and call signatures;
- exception classes and validation semantics;
- reward values and advantage normalization;
- loss reduction and gradient scaling;
- rollout ordering and RNG restoration;
- checkpoint keys, values, and load behavior;
- deterministic results for fixed inputs and seeds.

## Algorithm and Scoring Simplification

`reward.py` will expose one dependency-light canonical group-relative advantage
kernel because both GRPO and non-GRPO reward consumers already depend on that
layer. `algorithms.py` and `SearchRewardFunction` methods will use thin adapters
around it. There will be one source of truth for centering, optional
standard-deviation normalization, epsilon, clipping, DAPO behavior, and
zero-variance behavior. Tensor-native loss paths may retain vectorized tensor
arithmetic, but equivalence tests will pin them to the same numerical contract.

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

`get_response_log_probs` itself will live in the shared post-training utility
module. Its logits/target shift and response-mask contract remain unchanged,
and its DPO and GRPO callers will share focused off-by-one alignment tests.

The local controller will consume generation results through stable data
interfaces and call the shared policy-update path. Checkpoint helpers remain at
the orchestration boundary and keep the current serialization format.

No attempt will be made to force bandit and language-model optimization through
one abstraction where tensor shapes or optimizer semantics genuinely differ.

## Performance and Memory Optimization

Optimization is evidence-driven. Each changed hot path requires a focused
benchmark or profiler measurement before and after the change.

Candidate improvements are:

- Keep reference-policy inference conditional. Executing it under
  `torch.inference_mode()` rather than `torch.no_grad()` is a *candidate*, not a
  conclusion: it must clear the same benchmark bar as everything else here, and
  the KL term consuming the reference output must still backpropagate into the
  policy.
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

Two consequences of taking that rule literally:

- The benchmark harness must not time samples while `tracemalloc` is running.
  Doing so inflates every median several-fold and measures the profiler.
- A candidate that fails its benchmark is reverted and the measurement is
  recorded, so the next reader does not re-propose it. Rejected candidates are
  as much a deliverable as accepted ones.

### Reward-function optimization

The 1,088-line shared reward module is explicitly in scope. Optimization will
preserve every public `SearchRewardConfig` preset and the exact component keys,
weights, totals, sparse-token placement, group isolation, and output ordering.
The work will target:

- Normalize and tokenize each answer/gold string no more than once per scoring
  pass when token-F1 or multiple composite judges reuse the same text.
- Score correctness once, preferably through `BatchJudgeFn`, and pass the
  resulting scalar into reward-component arithmetic without invoking the judge
  again.
- Build group membership and group statistics once, then reuse them for scalar
  and sparse token-level advantages.
- Share sparse terminal-vector construction between batch reward and advantage
  APIs while retaining the current last-action-token fallback rules.
- Avoid computing disabled reward dimensions and expensive evidence/citation
  features when their configured weights are zero and their values are not
  requested for a public breakdown.
- Reuse extracted answer text, cited/fetched URL sets, evidence deltas, and
  grouped dimension totals within one `reward_components` call.
- Validate batch lengths before invoking a remote or expensive judge so invalid
  input cannot consume external work.

Any memoization will be bounded to a single scoring call unless profiling and
correctness tests prove that a longer-lived cache is safe. Mutable agent output,
configuration changes, and judge results must never be cached across calls.

Representative benchmarks cover:

- list and tensor group-advantage calculation;
- token-F1 and composite reward calculation;
- scalar, breakdown, batched, and sparse-token reward paths;
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
2. Move `get_response_log_probs` to the neutral post-training utility, redirect
   DPO and GRPO callers, and preserve its root/package export behavior.
3. Create `algorithms.py`, redirect imports/exports, and delete
   `core_algos.py`, `rollouts.py`, and `judge.py`.
4. Merge `trainers.py` into `training.py`, remove the reverse
   generation/training dependency, redirect imports/exports, and delete
   `trainers.py`.
5. Move the plotting utility to `examples/plot_grpo_rollouts.py` and delete the
   package copy.
6. Establish focused GRPO and reward-function performance baselines.
7. Consolidate the reward layer's correctness scoring, group statistics,
   sparse-vector construction, and enabled-component calculation with
   equivalence tests.
8. Simplify duplicate advantage, rollout scoring, policy-update, and
   tensor-assembly paths one at a time with equivalence tests.
9. Apply only benchmark-supported runtime or memory improvements.
10. Update documentation, examples, monkeypatch targets, and lazy registries.
11. Search for stale module paths and run focused and full verification.

## Testing

Implementation follows test-driven development:

- Write module-layout and ownership tests before moving implementations.
- Run focused tests after each deleted module and redirected dependency.
- Add numerical equivalence tests for advantages, rewards, losses, masks, and
  batch tensors.
- Assert reward scalar totals, breakdown dictionaries, batch results, sparse
  token placement, and judge invocation counts across every configuration
  preset and disabled-component fast path.
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

- `src/model/post_training/grpo` contains only `__init__.py`, `algorithms.py`,
  `core_algos.py`, `generation.py`, and `training.py`. **This is four
  implementation modules, not the three targeted.** `core_algos.py` is split
  back out because it is the package's torch boundary: everything in it needs
  torch and everything in `algorithms.py` — grouped sampling, scoring, the
  judges — does not. Merging them made `algorithms.py` unimportable without
  torch, and the CI unit-test job installs no torch, so 17 judge tests were
  silently skipped rather than run. The torch-free seam is load-bearing (the
  repo has shipped that CI failure five times) and outranks the module count.
- DPO and GRPO import `get_response_log_probs` from the neutral post-training
  utility and pass the shared token-alignment contract tests.
- `plot_rollouts.py` is available from `examples/plot_grpo_rollouts.py`.
- No live repository import or documentation targets a deleted module.
- Package-level and root exports preserve their current names and laziness.
- There is one canonical group-advantage implementation and one rollout
  scoring pipeline.
- Reward computation invokes each judge no more than once per answer, skips
  provably unused component work, and preserves scalar and breakdown outputs.
- The generation/training dependency is acyclic.
- Checkpoint, numerical, ordering, seed, and error-semantics tests pass.
- Every performance claim is backed by recorded before/after evidence, and every
  rejected candidate is recorded with the measurement that rejected it.
- Focused and full repository verification complete without new failures or
  warnings caused by the consolidation.

## Out of Scope

- Changing the mathematical GRPO objective or default hyperparameters.
- Changing reward weights, judge prompts, sampling defaults, or rollout policy.
- Redesigning external agent, retriever, model, or storage interfaces.
- Altering checkpoint formats or providing compatibility shim modules.
- Splitting `generation.py` into a new subpackage; the objective is the minimum
  practical module count.
