# Plan: choose `top_k` on the tuning/test split

Spec: `docs/superpowers/specs/2026-08-14-intent-top-k-selection-design.md`

## Pre-registered selection rule

**Committed before the joint sweep runs.**

> Select the `(top_k, min_margin)` pair with the highest **served accuracy on
> the tuning slice** at `coverage >= 0.60`. Break ties first toward higher
> out-of-scope deferral, then toward **lower `top_k`**.

The first clause and the first tie-break are the existing `_select_thresholds`
rule, unchanged — this extends that rule to a second dimension rather than
inventing one. The new tie-break goes to lower `k` because lower `k` measures
better out-of-scope separation, and abstention is this router's safety
property; when accuracy cannot tell two settings apart, the safer one wins and
the incumbent stays.

## The hazard, and why this is still honest

Pinning `k` has been load-bearing. The reported headline is **argmax** accuracy,
which is abstention-blind — it depends on `top_k` and on nothing else the sweep
chooses. That is what let the margin grid be re-derived in #512 *after* the
headline was known: with `k` fixed, no threshold the sweep picks can move
`test_slice.accuracy` by any amount. Letting the sweep choose `k` gives that up.

Three things keep this from becoming the fitting error the arc has already paid
for twice:

1. **The `k` grid is pre-existing.** `_SWEEP_TOP_K = (3, 5, 8, 15, 25)` was
   fixed in #511 as a report-only table. It is not being widened, and
   critically it is not being widened *after* seeing the headline it would now
   influence. Had I needed to change the grid, this work would have to stop and
   the grid be re-registered first.
2. **Selection touches the tuning slice only**, exactly as before.
3. **The test slice is read once**, at the selected pair, and reported as the
   result.

**If the sweep selects `k = 3`, nothing about the current guarantee changes** —
the incumbent survives on measured evidence rather than inertia, which is the
spec's stated goal, and `test_the_threshold_sweep_never_chooses_top_k` can stay
exactly as it is.

**If it selects a different `k`,** the headline must be re-derived at that `k`
and every pinned bar re-pinned in the same commit, and the guard test must be
*replaced* by one asserting the new property (selection is tuning-only) rather
than deleted. That is a bigger change and it will be called out as such.

## Steps

### 1. Extend the sweep to `(top_k, min_margin)`

`_select_thresholds` currently pins `top_k = TOP_K`. Sweep `_SWEEP_TOP_K`
against the existing margin grid, keeping `min_confidence` in the search as
today.

→ verify: the sweep has `len(_SWEEP_TOP_K) x len(_SWEEP_MIN_CONFIDENCES) x
len(_SWEEP_MIN_MARGINS)` rows and every row records its `top_k`.

### 2. Apply the pre-registered rule

→ verify: the selected row is the tuning-slice maximum under the rule, and the
tie-break demonstrably prefers lower `k` when served accuracy ties.

### 3. Read the test slice once

→ verify: report `test_slice.accuracy` and the served pair at the selected
`(k, min_margin)`. Compare against the incumbent `0.7928` at `k=3`.

### 4. Decide, and write the decision down

Either outcome is a result. Record in `docs/training-and-evaluation.md`:
the chosen `k`, the rule that chose it, the abstention cost accepted, and — if
`k` stays 3 — that it now survives on evidence, so a seventh PR need not
re-open it.

### 5. If and only if `k` changed

- `TOP_K` and `AGENTIC_SEARCH_INTENT_TOP_K` move together
- every pinned bar in `tests/unit/test_intent_index_eval.py` re-pinned
- `test_the_threshold_sweep_never_chooses_top_k` replaced, not deleted
- the whole "what it scores" table re-measured, since argmax is `k`-dependent

## Risk

Moderate, and concentrated in step 5. If `k` changes, this stops being a
config tweak and becomes a re-measurement of every number the router publishes.
The rule's tie-break toward lower `k` is deliberately conservative for that
reason: a marginal, noise-sized accuracy gain should not trigger that churn.
