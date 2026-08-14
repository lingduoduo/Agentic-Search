# Plan: author the off-domain probe set, before any anchor moves

Spec: `docs/superpowers/specs/2026-08-14-intent-canonical-coverage-design.md`
(phase 1 of 2 — this plan authors the instrument; a later change uses it)

## Why this is its own change

The spec's first instruction is to build the probe set **before** adding
anchors, "so it cannot be curated against — that is the contamination #512
quantified at 3.6 points."

Landing it as a separate commit, ahead of any anchor edit, turns that from a
claim into a fact anyone can check with `git log`. If the probes and the anchors
arrived together there would be no way to tell, from the outside, which was
written in response to which.

It is also why this change deliberately **reports no accuracy number**. Measuring
the probes needs the serving `top_k`, which #522 moves from 3 to 15; measuring
them here would either publish a figure at a `k` that is about to change, or
force this branch to stack on an open PR. Neither is worth it, and a probe set
with no score attached is precisely the artifact the spec asks for.

## The vocabulary decision

Generic **business/personal** — HR, finance, travel, facilities, scheduling,
procurement, personal admin. Chosen because it is the register #511's original
16 probes used (13 of which abstained), so the new numbers stay comparable to
that baseline, and because it is the widest guess at what non-IR traffic looks
like absent a record of what this deployment actually serves.

## Steps

### 1. Author 45 probes, 15 per route

In-scope (each has a correct route), route-labelled with taxonomy modules, in
ordinary English carrying none of the IR/ML vocabulary that saturates 47% of the
canonical set.

→ verify: schema valid against `INTENT_LABELS` / `modules_for_route`, ids
unique, routes balanced 15/15/15.

### 2. Check they are genuinely new

→ verify: no exact-text overlap with the canonical set or either eval set, and
no probe within `0.95` cosine of an anchor.

→ result: one probe failed this. `"forward the invoice to accounts payable"`
scored `0.9323` against the anchor `"forward the invoice to finance"` — the same
request one word apart, which would have measured the anchor rather than the
router. Replaced; the set's maximum is now `0.8867`, against
`"summarize what changed in the ranking config doc"`, which is a legitimate
generalization test (same verb pattern, entirely different subject).

### 3. Guard the properties, not the score

`tests/unit/test_intent_offdomain_probes.py` asserts size, schema, route
balance, absence of in-domain vocabulary, and non-duplication. It asserts
**nothing about accuracy** — that belongs in the evaluation report, and a test
pinning it here would quietly become a curation target.

Needs no encoder, so it runs in the fast CI job rather than skipping like the
pinned bars.

→ verify: 5 tests pass; `data/intent_offdomain_probes.json` is force-added,
since `data/` is gitignored and it would otherwise silently not land.

## What comes next, and what must not happen

Phase 2 measures the probes at the shipped `top_k` and adds anchors in the
vocabulary they show missing. Two rules carry over from the spec:

- **Re-measure every distribution statistic**, not just accuracy — #518's
  lesson: a canonical edit moves Cohen's d, leave-one-out, raw margin, and the
  tuning-slice margin quantiles the grids derive from.
- **Out-of-scope AUC must not fall.** Broader anchors risk pulling genuinely
  out-of-scope requests in, and that is the failure this router cannot afford.

And the rule this change exists to protect: **these probes are never edited in
response to their own score.** A probe that looks unfair after measurement is
evidence, not a defect.
