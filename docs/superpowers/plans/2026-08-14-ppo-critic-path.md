# Plan: remove the unreachable PPO-with-critic path

Spec: `docs/superpowers/specs/2026-08-14-ppo-critic-path-design.md`

## The decision the spec deferred

The spec deliberately did not choose between removing the helpers and marking
them, because the answer depended on something recorded nowhere: whether a
critic-based trainer is wanted. That was asked and answered — **remove**.

The evidence supporting it, gathered before asking:

- **No critic exists.** There is no value model, value head, or critic anywhere
  in `src/`. Every occurrence of the word describes its *absence* —
  "critic-free final-outcome GRPO", "no separate value model, no token-level
  critic", "compute GRPO outcome advantages without a critic value model".
- **Nothing produces the input.** `compute_gae_advantages` consumes per-token
  `values` from a critic head. Nothing emits them, so the path cannot be
  exercised end to end and its tests necessarily pass synthetic tensors — they
  check arithmetic, not a working path.
- **69 lines across two functions**, exported from three public surfaces.

## Steps

### 1. Remove both functions from `core_algos.py`

→ verify: 470 → 401 lines, module still parses, no residual references.

### 2. Remove the re-exports

`src/training/__init__.py`, `src/training/ppo/__init__.py` (both the import and
the `__all__` entry), and `src/model/generation.py`'s explicit pass-through.

→ verify: `import src.training, src.training.ppo, src.model.generation` all
succeed, and no reference survives in `src/`.

### 3. Tests — split rather than delete where a live property shares the test

`test_value_loss_and_kl_penalty_variants` covered **both** the removed
`compute_value_loss` *and* the live `kl_penalty` variants. Deleting it wholesale
would have taken real coverage with it. It is renamed
`test_kl_penalty_variants` with the value-loss half removed and the KL half
intact.

`TestComputeGAEAdvantages` covered nothing else and goes entirely.

→ verify: both files pass; `kl_penalty` coverage unchanged.

### 4. Docs

`training-and-evaluation.md` described the path as existing "for parity/tests
only". That paragraph now records the removal and the reason.

## The trap this had to avoid

`compute_ppo_policy_loss_core` sits in the same module, is exported alongside,
and is **live** — it is the clipped surrogate both GRPO trainers use, with a
group-relative advantage in place of GAE. A cleanup that pattern-matched on
"PPO" would have taken it and broken training silently in the sense that the
tests for it would simply vanish too.

The spec called this out in its Out-of-scope section for exactly that reason,
and it is why removal was done by explicit function name rather than by any
broader sweep.

## Verification

Full suite: **3169 passed**, 1 deselected (the hardware-sensitive latency bar,
unrelated). Ruff clean.
