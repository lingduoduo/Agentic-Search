# A wider evaluation instrument for the router

## Status

Not started. This is what every recent intent result has been asking for, in the
form of caveats nobody can currently resolve.

## Problem

The router is judged on **111 held-out queries**. Every conclusion of the last
several changes turns on movements of one or two of them:

| claim | actual margin |
|---|---|
| clears the `0.80` promotion bar | `0.8108` = 90/111, against 88 for `0.80` — **2 queries** |
| `top_k=15` beats `k=3` | `0.8018` vs `0.7928` — **1 query** |
| `hard_40` regressed in #524 | `0.7250` vs `0.7500` — **1 query of 40** |
| out-of-scope AUC | 111 in-scope against **24** probes |

None of those is measurable at that resolution. The doc says so in several
places, and the honest phrasing it has settled on — "no longer clearly below the
bar" rather than "above it" — is a workaround for an instrument that cannot
support the sentence anyone actually wants to write.

The out-of-scope side is worse than the in-scope side: **24 probes** denominate
every AUC and Cohen's d figure quoted, and those same 24 also tie-break the
threshold sweep, so they are not fully held out from what they measure.

## Why this is now the blocking item

Every cheap lever has been pulled. Encoder (#512), `top_k` (#522), thresholds
(#520, #522), anchors (#524). The remaining questions are all of the form "is
this real or is it noise", and no amount of further tuning answers that — only a
bigger instrument does.

Concretely: the promotion decision cannot be made well on this instrument. Two
queries either way flips which band the result sits in.

## Approach

1. **Grow the clean eval set** well beyond the current 151 `bulk-` queries,
   route-stratified, authored without reference to the canonical set so it does
   not measure curation.
2. **Grow the out-of-scope probes** substantially past 24, and — the important
   part — **split them**, so the probes that tie-break threshold selection are
   disjoint from the probes that denominate reported separability. That closes
   the caveat the doc has carried since #512.
3. Re-run the split with the same seed discipline. The tuning/test proportions
   should be reconsidered rather than inherited: 40 clean queries were spent on
   tuning because the set was small, and a larger set may not need that ratio.
4. Report confidence intervals alongside point estimates, so a one-query
   difference stops reading as a result.

## Acceptance

- Test slice materially larger than 111, with the split still deterministic in
  its seed and an exact partition.
- Out-of-scope probes split into disjoint tuning and reporting sets, with the
  reported AUC computed only against the reporting half.
- Every pinned bar re-derived on the new instrument, in one commit, with the old
  and new values both recorded — the numbers will move simply because the
  instrument changed, and that must not read as a regression.
- Confidence intervals published for test-slice accuracy and AUC.
- **The canonical set is not touched.** Growing the instrument and changing the
  model in the same change makes neither measurable.

## Out of scope

Any tuning. This spec produces a better ruler; using it to re-open `top_k`,
thresholds, or promotion is separate work, and doing them together would fit the
new instrument to a decision made against the old one.

## Note

The eval files are force-added under a gitignored `data/`. A new file that is not
force-added silently will not land, and the CI gate rebuilds from the committed
set only.
