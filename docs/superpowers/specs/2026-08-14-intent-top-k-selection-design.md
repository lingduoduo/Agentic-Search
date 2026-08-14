# Choose `top_k` on the tuning/test split

## Status

Deferred through #511, #512, #513, #516, #518 — five PRs. Not started.

## Problem

`TOP_K = 3` in `src/model/intent_knn.py` is the number of per-route neighbours
averaged into a route score. It was picked arbitrarily when the kNN router was
written and **has never been chosen on data**. #511 established that this
matters: sweeping it moves accuracy by ~6.6 points, which is larger than most of
the deliberate changes made since.

The measured curve on the tuning slice under the serving encoder, from
`evaluation_report.json`'s `top_k_sweep`:

| `top_k` | tuning accuracy | leave-one-out | raw separation margin |
|---|---|---|---|
| **3 (shipped)** | 0.8286 | 0.7321 | 0.0416 |
| 5 | 0.8571 | 0.7893 | 0.0392 |
| 8 | 0.8714 | 0.8071 | 0.0363 |
| 15 | 0.8714 | 0.8393 | 0.0328 |
| 25 | 0.8714 | 0.8464 | 0.0303 |

Tuning accuracy plateaus at `0.8714` from `k=8`. Leave-one-out keeps climbing
monotonically all the way to `k=25`. Separation falls at every step.

## Why it was right to defer, and why that no longer holds

It was deferred because there was no honest instrument. Before #512 there was no
tuning/test split, so any `k` read off a curve was fitted to the queries it was
reported on — which is precisely the error #512 documented costing 3.6 points.

**That blocker is gone.** The split exists, is deterministic in seed `17`, and
the sweep above is already computed on the tuning slice only. The remaining
reason for `k=3` is inertia.

## The trade is real and must be decided, not optimized

Accuracy and abstention pull against each other here. Averaging more neighbours
lifts the score floor for *every* route, including routes the request has nothing
to do with, so separation degrades as `k` grows. There is no `k` in the table
that improves both.

This is therefore a **judgment about what the router is for**, not a search for a
maximum:

- If abstention is the safety property (the current stance — abstaining costs an
  LLM fallback, misrouting costs a wrong answer), a low `k` is defensible and
  `k=3` may survive on its merits.
- If coverage is the goal — e5 at `k=3` already defers 61 of 111 test queries,
  which is the standing cost objection to promotion — a higher `k` buys accuracy
  and the margin threshold can be re-tuned to pay for the lost separation.

The output of this work is a **decision with a stated rationale**, and `k=3`
surviving is a perfectly good outcome. What is not acceptable is `k=3` continuing
by default.

## Method

1. Sweep `k` **jointly with `min_margin`** on the tuning slice. They are coupled:
   raising `k` compresses margins, so a `k` evaluated at the current `0.015` is
   evaluated at the wrong threshold for itself.
2. Report the selected pair once on the test slice. One shot.
3. Preserve the invariant `test_the_threshold_sweep_never_chooses_top_k` guards
   today — if `k` becomes sweep-selected, that test must be deliberately replaced,
   not deleted, and the reported headline must be re-derived, because argmax
   accuracy is abstention-blind but **not** `k`-blind.

## Acceptance

- A recorded decision naming the chosen `k`, the rule that chose it, and the
  abstention cost accepted — in `docs/training-and-evaluation.md`.
- Test-slice numbers re-measured at the chosen `(k, min_margin)` pair and quoted
  as the new headline.
- If `k` changes: `AGENTIC_SEARCH_INTENT_TOP_K`'s default and `TOP_K` move
  together, and every pinned bar in `tests/unit/test_intent_index_eval.py` is
  re-pinned in the same commit.
- If `k` stays `3`: the reasoning is written down so the sixth PR does not
  re-open it.

## Note on the historical table

`docs/training-and-evaluation.md` still carries a MiniLM-era `k` sweep computed
over the queries it reported. It is labelled history and must not be used to
choose `k` — it is the exact fitting error this spec exists to avoid repeating.
