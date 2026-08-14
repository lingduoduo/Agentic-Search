# Plan: measure off-domain coverage, then close what the measurement shows

Spec: `docs/superpowers/specs/2026-08-14-intent-canonical-coverage-design.md`
(phase 2 of 2 — phase 1 authored the probes in #523)

## The measurement came first, and it changed the plan

The spec was written from #511's finding: 47% IR/ML vocabulary, **13 of 16**
off-domain probes abstaining. It named canonical coverage "the highest-value
untried lever" and expected a substantial anchor-writing effort.

Measured at the shipped settings before touching a single anchor:

| | value |
|---|---|
| argmax accuracy | `0.8222` (37/45) |
| abstained | 19/45 (42%) |
| served wrong | 2 |

Off-domain accuracy was **higher than the in-domain test slice** (`0.8018`). The
premise had largely dissolved — the encoder swap and the `top_k` selection did
it, not anchors. Writing 30+ anchors against a mostly-solved problem would have
been speculative work dressed up as spec compliance.

**What the measurement did show** is that the residual failure is a *route*, not
a vocabulary: all 6 search-route probe failures were `search` read as `tool` or
`chat`, against one failure each for the other two routes. `search` is also the
weak route in-domain (25/37). So the work became narrow and evidence-backed:
add `search` anchors covering business document and fact lookup.

## Steps

### 1. Author 24 search anchors, then prove they help before committing them

Built as a candidate set in the scratchpad, indexed separately, and compared
against the live index on the held-out sets. **Nothing was committed until the
comparison came back positive** — the alternative is shipping anchors because
they were written, which is how a canonical set silently accumulates noise.

→ verify: test-slice accuracy does not fall, out-of-scope AUC does not fall,
off-domain abstention falls. All three held.

### 2. Check every guard the canonical set has

→ verify: 304 examples (band 260–400); route shares `search 0.365 / chat 0.326 /
tool 0.309` (band 0.25–0.40); internal cosine max `0.9271` with nothing at or
above the `0.94` bar; no leakage into the eval, hard, or probe sets.

### 3. Re-derive everything downstream, per #518

A canonical edit moves the spread statistics and the computed grids.

→ verify: the joint sweep re-selects `k=15, min_margin=0.010` (stable), and
`min_module_score` re-derives `0.821` → `0.8216`.

### 4. Raise the floors

→ verify: test-slice floor `0.78` → `0.79`, AUC floor `0.84` → `0.85`.
Deliberately **no** floor on `hard_40` argmax — see below.

## The two rows that moved the wrong way

**`hard_40` argmax fell `0.7500` → `0.7250`.** One query. Its *served* accuracy
rose `0.8947` → `0.9048` on more coverage (19 → 21), so at the operating point
the adversarial set improved too. This is the third time in this arc that argmax
and the operating point have disagreed, and it is the reason no floor is pinned
to that argmax figure: one would have blocked a change that improves what
serving actually does.

**Leave-one-out fell `0.8393` → `0.8191`.** Expected, and arguably healthy. #518
established that leave-one-out measures the anchor set's self-consistency; 24
anchors in a vocabulary region the set did not previously cover legitimately
lower how well anchors recover their own route from their neighbours. A set that
covers more ground scores worse against itself.

## What this spends

The 45 probes are **no longer a clean instrument**. Their per-item failures were
read in order to decide which anchors to write, which makes them a development
set from here on. The genuinely held-out gates for this change were the sets
that had not been consulted for this purpose: test-slice accuracy, `hard_40`,
and out-of-scope AUC.

Recorded in the doc rather than glossed. A future off-domain figure that wants
to be a *result* rather than a development number needs a fresh probe set,
authored the same score-free way.
