# A DPO trainer for `src/training/dpo/`

## Problem

`src/training/` implements SFT and a GRPO stack, and nothing else. Direct
Preference Optimization is the obvious missing third method: it trains directly
on preference pairs without a reward model, a critic, or online sampling, which
makes it the cheapest way to use preference data this repo can already collect.

Until now the top-level `src/training/__init__` docstring stated that no DPO
trainer exists and named `dpo/` as where one would go. This is that package, and
those two statements (there and in `.claude/CLAUDE.md`) are corrected as part of
the change — a docstring asserting the absence of a package sitting next to that
package is a defect, not documentation.

## Where the pairs come from

DPO needs `(prompt, chosen, rejected)`. The repo's `retrieval_feedback` table
records one thumbs-up/thumbs-down **per session** and has no notion of two
competing answers to the same prompt, so it cannot yield pairs directly — only
by matching question text across sessions with opposite signals, which is sparse
and fragile. The existing `SimulatedPreferenceJudge` could label on-policy
samples, but it is a deterministic heuristic placeholder, so pairs would be
exactly as good as that heuristic.

So the ingest for this version is a **JSONL file** of
`{"prompt", "chosen", "rejected"}` — the standard DPO format. DPO is an offline
method over a fixed preference set, so a file is the honest interface, and it
decouples the trainer from the unresolved judge-quality question. The trainer
itself takes `list[PreferenceExample]` and never reads a file, so a judge-backed
or feedback-backed loader can be added later without touching it.

## Design

Two modules, mirroring how `sft/` and `rl/` are already shaped.

### `dpo/data.py`

```python
@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str

def load_preference_pairs(path: Path) -> list[PreferenceExample]
```

Validation is strict, in the style of `load_canonical_examples`: malformed JSON,
a non-object line, a missing or blank key, or an empty file all raise. This data
*is* the training signal — a silently skipped row changes what the model learns
with no later symptom.

One rejection is worth naming: `chosen == rejected` raises. Such a pair
contributes exactly `log 2` of constant loss and zero gradient forever, so it is
never a useful training example, and it is a common artifact of a
pair-construction bug upstream.

### `dpo/trainer.py`

The loss, for one pair, with `y_w` chosen and `y_l` rejected:

```
L = -log σ( β · [ (log π_θ(y_w|x) − log π_ref(y_w|x))
                − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )
```

`log π(y|x)` is the **sum** of per-token log probabilities over response tokens
only; prompt tokens are excluded, the same masking `SFTTrainer` performs with
`-100`. No length normalization — that is SimPO, a different method.

Per-token log probs come from `get_response_log_probs` in
`src.training.rl.llm_grpo_trainer` rather than a second implementation. That
function already owns the `logits[:, prompt_len - 1 : -1]` shift that aligns
logits with the tokens they predict — an off-by-one there is silent and
catastrophic, so it must exist in exactly one place.

The reference forward runs under `torch.no_grad()`.

`DPOTrainer` mirrors `SFTTrainer`: a frozen `DPOConfig`
(`beta=0.1`, `epochs`, `lr`, `batch_size`, `max_length`, `grad_clip`),
construction as `DPOTrainer(policy, tokenizer, optimizer, config, device)`, an
injectable optimizer, and `save()` writing HuggingFace format. The reference is
`deepcopy(policy)` frozen at init (`.eval()`, `requires_grad_(False)`), matching
`LLMGRPOTrainer`; callers wanting a distinct SFT checkpoint can inject one.

`train()` returns per-step records carrying `loss`, `margin` (the implicit
reward gap) and `accuracy` (the fraction of pairs where chosen outscores
rejected). Loss alone cannot distinguish learning from collapse; the margin and
accuracy can.

## Testing

A tiny stub causal-LM keeps every test offline — no model downloads, so these
run in the unit-test job, which installs no heavy ML packages beyond torch and
guards with `pytest.importorskip("torch")`.

The load-bearing test is a **known value**: when the policy still equals the
reference, every difference term is zero, so the loss is exactly `-log σ(0)`
= `log 2` ≈ 0.6931. That pins the formula itself rather than merely its shape —
a sign error or a missing `β` still produces a plausible-looking number, and only
a known value catches it.

The rest: gradient direction (chosen logprob rises relative to rejected after a
step), β monotonicity on a fixed positive margin, the reference staying
bit-identical after training with no grads attached, prompt tokens not affecting
the sequence sum, and the loader's five rejection cases.

## Out of scope

No CLI entry point, no IPO/cDPO/SimPO variants, no judge-backed or
feedback-backed loader. Each is a separate, reviewable change; none is needed to
make this package useful or testable.
