# Intent routing: swap MiniLM for e5-small-v2 — Design

**Status:** proposed
**Branch:** `feat/intent-encoder-e5`, stacked on `feat/intent-knn-routing` (PR #511)
**Depends on:** #511 — `intent_encoder.py`, the `top_k` parameter, and the evaluation instrument all land there
**Predecessor spec:** `docs/superpowers/specs/2026-08-13-intent-knn-routing-design.md`

> **Branch hazard, read first.** This branch is stacked on an open PR. A squash-merge of #511 will not carry these commits, and this repo has already lost a stacked commit that way (PR #440, re-landed as #441). After #511 merges, **rebase before opening the PR**:
> ```
> git fetch origin && git rebase --onto origin/main feat/intent-knn-routing feat/intent-encoder-e5
> ```
> Verify with `git log --oneline origin/main..HEAD` that only this branch's commits remain.

## Problem

PR #511 built a nearest-canonical-example intent router and measured it honestly: clean-slice route accuracy `0.6225`, combined `bulk-181` `0.6519` against a `0.75` floor. It reached a **hard stop** and ships dark.

Its documentation attributed that to the representation, then corrected itself: `TOP_K = 3` was an arbitrary constant, and sweeping it reaches `0.6887` clean on the same encoder and anchors. Both readings pointed at the same next lever — the sentence encoder — and probing it produced a result large enough to change the verdict.

**Measured, on #511's committed anchors and evaluation instrument:**

| encoder | k | clean-151 | bulk-181 | hard-40 | OOS AUC | Cohen's d | p95 latency |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 *(shipped)* | 3 | 0.6225 | 0.6519 | 0.6250 | 0.868 | 1.49 | 6.1ms |
| all-MiniLM-L6-v2 | 15 | 0.6887 | ~0.71 | 0.6500 | 0.871 | — | 6.1ms |
| **e5-small-v2** | 15 | **0.8146** | **0.8287** | 0.7500 | 0.871 | — | **12.0ms** |
| e5-base-v2 | 8 | 0.8212 | 0.8453 | 0.7500 | **0.927** | 2.02 | 32.2ms |

`e5-small-v2` clears the `0.80` promotion bar, stays inside the 25ms latency ceiling, keeps **384 dimensions** so the index format is unchanged, and is **already a repo dependency** — `intfloat/e5-small-v2` is in the dense-retrieval model list and `e5-base-v2` is already in the local HuggingFace cache. This is a configuration change and a rebuild, not new infrastructure.

**Every number above is fitted.** k was chosen after seeing these results, on the same queries used to report them. Correcting that is the substance of this spec.

## Approach

Swap the encoder, re-tune the three hyperparameters on data that is not the test set, and re-measure. Ship dark; promotion is a separate change.

### Rejected alternatives

**e5-base-v2 and raise the latency ceiling.** Better on every accuracy and separability measure, and the 25ms ceiling was an arbitrary number in the predecessor spec — routing precedes an LLM call taking seconds, so 32ms is plausibly irrelevant. Rejected for now because it doubles the index to 768 dimensions and renegotiates a stated budget, for `+0.017` accuracy. Its out-of-scope AUC advantage (`0.927` vs `0.871`) is real and makes it the obvious candidate if a later round needs better abstention.

**Keep MiniLM, bank only the k gain.** Free and fast, but tops out near `0.69` clean, never clears the bar, and the router stays dark forever.

**A learned head over frozen embeddings.** The other lever named in #511's docs. Deferred: it reintroduces a training run, a checkpoint, and promotion gates — precisely the machinery #511 removed — and the encoder swap alone appears to clear the bar.

## Design

### 1. The prefix contract

E5 models are trained with instruction prefixes and **silently degrade without them** — no error, just worse vectors. Every text gets `"query: "`, applied **symmetrically** to canonical anchors and to incoming queries. (E5 also defines `"passage: "` for asymmetric retrieval; this is symmetric short-text similarity, so both sides use `"query: "`.)

The prefix is a property of the encoder, so it lives with it:

- `intent_encoder.py` gains a prefix per model, applied inside `encode_texts` — callers never pass it, so no call site can forget.
- The index stores the prefix beside the encoder name, and `load_intent_index`'s encoder-match check is extended to compare both.

That check was added during #511's review as a guard against a same-dimension model swap producing garbage similarities. This change is exactly that scenario: e5-small-v2 is also 384-dimensional, so **an old MiniLM index would load and score without error**. The check is what turns that into one loud, cached failure.

### 2. Hyperparameter tuning — no fitting on the test set

Three hyperparameters need values: `top_k`, `min_confidence`, `min_margin`. None may be chosen on the reported test slice.

**Tuning set:** the contaminated `legacy-30` (already worthless as a gate — spending it here costs nothing that still has value) plus a fixed 40-query slice of the clean 151, sampled stratified by route with a recorded seed.

**Test set:** the remaining 111 clean queries plus `hard-40`, untouched until the final measurement.

**Leave-one-out is not used to select `k`.** It is a biased selector: measured on #511's index, LOO keeps climbing (`0.7464` at k=15, `0.7643` at k=25) while clean-slice accuracy has already turned down (`0.6887` → `0.6755`). Past k≈15 LOO measures the anchor set's internal cohesion rather than held-out behaviour. It stays a reported diagnostic.

The 111-query test slice gives a wider confidence interval than #511's 151. That is the honest cost of not fitting, and the report states it rather than hiding it.

### 3. The out-of-scope bar changes units

#511's bar is **raw margin ≥ 0.25** — mean in-scope confidence minus mean out-of-scope confidence, in cosine units. That bar is **not comparable across encoders**, and this change proves it: e5-base scores raw margin `0.0401` against MiniLM's `0.1188` while being clearly *better* separated (AUC `0.927` vs `0.868`, Cohen's d `2.02` vs `1.49`). E5 compresses cosines into a narrow high band, so a bar in raw units would reject the better model.

Replace it with scale-free measures: **AUC ≥ 0.90**, with Cohen's d reported alongside. Raw margin is still recorded, labelled as encoder-specific and not comparable across rows.

### 4. Migration and blast radius

Dimensions are unchanged at 384, so `index.npz`'s format, `IntentIndex`, the scoring rules, the taxonomy, the canonical set, the evaluation instrument, and the whole serving cascade are all untouched. What changes: `DEFAULT_ENCODER`, the prefix contract, three tuned defaults, the out-of-scope bar's units, the pinned floors, and the documentation.

`route_request` is not modified. Neither is any dispatcher or the frontend.

Any existing built index is invalidated — same dimensions, different model — and the encoder-match check rejects it with a message naming the rebuild command.

### 5. What ships

**Dark**, as #511 does. `intent_index_path` stays unset by default, so the encoder never loads in production. This branch establishes whether the bar is genuinely cleared on an untouched test set; wiring it live is a separate change with the final numbers in hand, reviewed on its own terms.

## Success criteria

| metric | shipped today | bar |
|---|---|---|
| route accuracy, 111-query test slice | 0.6225 (on clean-151) | **≥ 0.80** |
| out-of-scope AUC | 0.868 | **≥ 0.90** |
| p95 routing latency, end to end | 6.1ms | **≤ 25ms** |
| hard-40 accuracy | 0.6250 | reported, no bar |
| leave-one-out | 0.6750 | reported, no bar, not a selector |

**Hard stop:** below `0.75` on the test slice, report every number and stop. No further tuning — that is a new spec. This is the rule that made the whole `TOP_K` error visible, and it applies to its own successor.

## Risks

| risk | mitigation |
|---|---|
| **The headline number falls once k is not fitted.** `0.8287` was chosen after seeing the results; the honest expectation is lower. | The bar is applied to the properly-split number, not the fitted one. If it lands under `0.80` the router stays dark, and under `0.75` the arc stops. |
| A missing or wrong `"query: "` prefix degrades silently | Prefix applied inside `encode_texts` so no call site can omit it; stored in the index and verified on load |
| A stale MiniLM index loads without error (same 384 dimensions) | Encoder-match check extended to encoder name *and* prefix; failure is loud and cached |
| Thresholds carried over from MiniLM are meaningless | All three re-tuned; the units trap is documented as #511 documented softmax→cosine |
| 111 test queries is a small instrument | Stated with the result; the eval set remains the honest constraint on everything here |
| Squash-merge of #511 drops this branch's commits | Rebase instruction at the top of this document; verify before opening the PR |

## Out of scope

Promotion to live routing; plan-aware / multi-route output (blocked on per-step accuracy — a two-step plan needs both steps right, so joint accuracy is roughly the square of the per-step number); a learned head; e5-base-v2 and any latency-budget renegotiation; the CI eval-gate job that would stop the pinned floors from silently skipping.
