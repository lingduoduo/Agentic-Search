# Broaden canonical coverage beyond this project's own vocabulary

## Status

Known since #511. Not started. This is the most likely single cause of the gap
to the promotion bar.

## Problem

The canonical set — 280 examples, which with kNN routing *is* the model — was
curated from this project's own example corpus. **47% of the examples carry
IR/ML vocabulary** (retrieval, embedding, index, rerank, corpus).

The measured consequence, from #511's off-domain probe: of 16 held-out in-scope
queries drawn from outside that vocabulary, **13 abstained** rather than routing,
and **9 had a wrong best-guess route underneath**. (The counts overlap; an
abstaining query still has a nearest route, it just does not clear the
thresholds.)

The failure is safe — abstention defers to the LLM classifier — but off-domain
traffic abstains far more often than it should, and real traffic is mostly
off-domain. A router curated on the vocabulary of the repository it lives in is
measuring itself.

## Why this is the highest-value remaining lever

The arc has now tried the alternatives:

| change | result |
|---|---|
| more training data / a trained head (#509, #510) | MLP `0.4768` |
| replace the head with kNN over curated anchors (#511) | `0.6225` |
| a stronger encoder (#512) | `0.7928`, **+0.17** |
| sweeping `top_k` (#511, still open) | ~+0.066 on the tuning curve |

The encoder swap was the big win and it landed. What remains untried is the
*content* of the anchor set, which is the one component nobody has changed since
it was first authored — and which #511 already identified as topically narrow.

Route accuracy sits at `0.7928` against a `0.80` promotion bar. This is the
cheapest plausible route to closing a 0.7-point gap.

## Constraints that make this harder than "add more examples"

1. **A bad anchor does not average out.** With kNN it becomes an attractor that
   pulls every nearby query onto its route. `test_intent_canonical_data.py`
   guards the size band, per-route balance, per-module support, and internal
   near-duplication — but nothing guards *quality*.
2. **The internal-similarity ceiling is now tight.** `_MAX_INTERNAL_COSINE` was
   tightened `0.95 → 0.94` in #516, a hair above the clean set's measured
   maximum of `0.9271`. New anchors must clear it, and that is deliberate.
3. **The current route balance is chat 99 / tool 94 / search 87.** `search` is
   both the smallest route and the weakest on the test slice (25/37, and it
   serves only 11 of those 37). New anchors should not worsen that skew.
4. **Every canonical edit invalidates the distribution statistics**, per #518 —
   Cohen's d, leave-one-out, raw margin, and the tuning-slice margin quantiles the
   threshold grid derives from all move. Re-measure all of them, not just accuracy.

## Approach

1. Build an off-domain in-scope probe set substantially larger than #511's 16, in
   ordinary business/personal vocabulary, route-labelled and held out from
   curation. **Author it before adding anchors**, so it cannot be curated against
   — that is the contamination #512 quantified at 3.6 points.
2. Add anchors in vocabulary the set currently lacks, keeping route balance and
   clearing the `0.94` internal-similarity bar.
3. Append → rebuild → re-measure, every time.

## Acceptance

- Off-domain probe abstention falls materially from 13/16, with the wrong
  best-guess count falling too — a probe that stops abstaining but stays wrong is
  a regression in disguise.
- Test-slice route accuracy does not fall; the goal is `≥ 0.80`.
- Out-of-scope AUC does not fall below `0.8551`; broader anchors risk pulling
  genuinely out-of-scope requests in, and this is the guard against that.
- The 47% IR/ML concentration figure is re-measured and quoted in
  `docs/training-and-evaluation.md`.
- Every distribution statistic in the doc is re-derived, per #518's lesson.

## Note

`data/` is gitignored, so `data/intent_canonical.json` and the eval files are
force-added. Any new probe file must be force-added the same way or it will not
land, and the CI `Intent Routing Gate` rebuilds from the committed set only.
