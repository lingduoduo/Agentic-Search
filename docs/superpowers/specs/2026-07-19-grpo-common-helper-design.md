# Consolidate feedback-GRPO duplication into examples/_grpo_common.py

**Date:** 2026-07-19
**Status:** Approved

## Problem

`examples/run_feedback_grpo.py` and Phase 2 of `examples/run_sft_grpo.py` contain
an identical ~30-line block: load feedback examples → build reward_fn (human
feedback, correctness_weight=0.0) → build `SearchAgentGRPOTrainer.from_pretrained`
→ `step_async` → save checkpoint. `run_feedback_grpo.py` is exactly
`run_sft_grpo.py`'s Phase 2 with no SFT phase. Both are documented CLIs
(docs/training-and-evaluation.md), so neither can be deleted.

## Design

Extract the shared block into `examples/_grpo_common.py`:

```python
async def run_feedback_grpo_step(
    *, model_path, db_path, output_dir, min_ratings,
    human_feedback_weight, num_rollouts, search_url, device,
) -> dict:
    """Load feedback examples, run one GRPO step from model_path, save a
    checkpoint to output_dir, and return the step metrics."""
```

Both scripts keep their own argparse/`main` and documented flags; their `_train`
bodies call this helper:
- `run_feedback_grpo.py._train` → one call with `model_path=args.model`,
  `output_dir=args.output_dir`.
- `run_sft_grpo.py._train` → keeps Phase 1 (SFT) unchanged, then calls the helper
  for Phase 2 with `model_path=grpo_model_path`, `output_dir=args.grpo_output_dir`.

## Scope / non-goals

- No CLI/flag changes; both documented entry points keep working identically.
- No change to the SFT phase, trainers, reward, or data loaders.
- Behavior-preserving refactor only.

## Verification

- New `tests/unit/test_grpo_common.py` mocks `SearchAgentGRPOTrainer.from_pretrained`
  and `load_feedback_examples`, asserting the helper passes prompts/ground_truths/
  metadata to `step_async`, saves policy+tokenizer to `output_dir`, and returns metrics.
- Both scripts still import and `_parse_args()` works.
