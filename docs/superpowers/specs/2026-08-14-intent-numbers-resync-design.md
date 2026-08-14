# Re-sync the intent routing numbers after #516

## Problem

`docs/training-and-evaluation.md` documents the canonical-example intent router in
detail, including six tables of measured numbers. #516 edited the canonical set —
it rewrote `canon-auth-011` to remove a near-duplicate pair — and updated the
*prose* about that change, but did not propagate the effect into the tables above
it.

The result is a document that contradicts itself: the "Known limitations" section
states leave-one-out fell `0.7393 → 0.7321`, while the headline table forty lines
earlier still prints `0.7393`.

The underlying reason is worth stating, because it will recur. A canonical edit
leaves *accuracy* bars alone — argmax accuracy, hard-40, and out-of-scope AUC were
genuinely unchanged, which is what #516 checked. What it moves is every statistic
that summarizes the *spread* of the score distribution: Cohen's d, raw separation
margin, leave-one-out, and the margin quantiles the threshold grid is derived
from. #516 checked the first group and assumed the second followed.

## Scope

Documentation only. No source, test, or data file changes. The shipped router,
its thresholds, and its promotion status are all untouched — `0.7928` still lands
in the middle band and the artifact stays dark.

## What is measured

Every number is re-derived from a freshly built index on the committed canonical
set (280 anchors), via the documented commands:

- `python -m src.model.intent_index_cli build` then `evaluate`
- `python -m examples.measure_intent_operating_point`
- direct `separability_report` / `leave_one_out_route_accuracy` calls for the
  MiniLM comparison column, which no CLI emits

## Corrections

| Location | Was | Now |
|---|---|---|
| headline table, Cohen's d | `1.4747` / `1.6208` | `1.4852` / `1.6365` |
| headline table, leave-one-out | `0.7393` / `0.6750` | `0.7321` / `0.6643` |
| headline table, raw margin | `0.0280` / `0.1188` (mismatched slice) | `0.0278` / `0.1210` (both test_111) |
| in-scope confidence span | `0.792`–`0.905` | `0.792`–`0.896` |
| operating-point tables, MiniLM | `0.6667` (74/111), `0.7162` | `0.6757` (75/111), `0.7200` |
| e5 `top_k` sweep, leave-one-out column | `0.7393`…`0.8500` | `0.7321`…`0.8464` |
| tuning-slice margin p75 | `0.0280` | `0.0274` |

Confirmed unchanged and left alone: test-slice accuracy `0.7928`, hard-40
`0.6750`, AUC `0.8551`, per-route `search 25/37 · chat 33/37 · tool 30/37`,
wrong-route counts `2` versus `21`, the selected thresholds, and every claim
#516 made about the duplicate pair itself (39,060 pairs, max internal cosine
`0.9271`, none at or above `0.93`).

## Two non-numeric changes

1. **A new measured finding.** Broken out by route on the test slice, the margin
   gate abstains hardest on `search` — the weak route at 25/37 argmax serves only
   11 of 37, while `chat` serves 18/37 with no errors. Abstention concentrates on
   the queries the index cannot place rather than spreading evenly, which
   strengthens the existing operating-point argument and was not recorded anywhere.

2. **A structural fix.** The guide's opening sentence and its `← Back to README`
   link were stranded in the middle of the file, below the intent section that had
   been inserted above them. The intro moves under the title; the back-link moves
   to the end.

## Why the MiniLM column moves at all

MiniLM is not the serving encoder, but it is scored against the *same* canonical
set, so a canonical edit moves its numbers too. Re-measuring it is what keeps the
before/after comparison a matched pair rather than one number from each of two
different anchor sets — the property the document already insists on for slices.

## Out of scope

- Re-deriving `min_module_score`, still dead at `0.45`. Unchanged by this work and
  still owed before promotion.
- The MiniLM-era ceiling section's historical table. It is relabelled as history
  with today's equivalents noted, not re-measured — its value is the reasoning.
- Latency (`9.73ms` / `11.47ms`). Re-measured at `9.70` / `10.07`, inside
  run-to-run noise on the same machine class; churning it would add diff without
  adding meaning.
