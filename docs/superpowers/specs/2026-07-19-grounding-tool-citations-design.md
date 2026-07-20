# Ground-check tool [Tx] citations, not just document [Dx]

**Date:** 2026-07-19
**Status:** Approved

## Problem

The optional runtime grounding verifier (`src/context/grounding.py`,
`GroundingVerifier`) is a secondary, sentence-level lexical check that runs when
`AnswerGenerationRequest.verify_grounding` is set. It flags dangling citations
(citing evidence not present) and strips them from the rendered answer.

Its citation regex is `\[(D\d+)\]` — **document evidence only**. Approved tool
results are cited as `[Tx]` (`EvidenceSource(provenance="tool")`, IDs `T1, T2…`),
and a supported claim citing tool evidence renders a `[Tx]` marker just like a
`[Dx]`. But the grounding verifier's regex never matches `[Tx]`, so:

- a tool citation is never lexically checked against its tool-result text, and
- a *dangling* `[Tx]` (no such tool evidence) is never flagged or stripped.

This is inconsistent with `extract_citations` (`src/context/utils.py`), which
**already** recognizes `[((?:D|T)\d+)]`. The grounding verifier is the only
citation-aware component that still lags. (This is a caveat already documented in
`docs/retrieval.md` under "Approved tool evidence".)

There is also a structural reason the verifier can't check `[Tx]` today: it maps
citations against `context.documents` (D* only). Tool evidence lives in a separate
`list[EvidenceSource]` computed in `generate_answer` and never reaches the
verifier.

## Goal

Make `GroundingVerifier` treat `[Tx]` citations as first-class: check tool
citations against their tool-result text, and flag/strip dangling ones — matching
how `[Dx]` is already handled. Opt-in path only (`verify_grounding=True`);
behavior when tool evidence is absent is unchanged.

## Design

**1. `GroundingVerifier.verify`** (`src/context/grounding.py`) — add an optional
`tool_evidence: list[EvidenceSource] | None = None` parameter. Build a unified
`id → text` map from `context.documents` (D*) plus each tool source (T*), then
run the existing lexical-overlap logic against that map instead of a doc-only map:

```python
text_by_id = {doc.id: doc.content for doc in context.documents}
for source in tool_evidence or ():
    text_by_id.setdefault(source.id, source.text)   # D* wins on any id clash
```

**2. Regex** — widen to cover both prefixes:
- `_CITATION_RE`: `\[(D\d+)\]` → `\[([DT]\d+)\]`
- `_split_sentences` negative-lookahead: `(?!\[D\d+\])` → `(?!\[[DT]\d+\])`

**3. Call site** (`src/context/pipeline.py`) — the `tool_evidence` local (already
derived as `[e for e in evidence if e.provenance == "tool"]`) is in scope at the
`verify_grounding` block; pass it through:

```python
report = GroundingVerifier().verify(answer, request.context, tool_evidence=tool_evidence)
```

**4. Docstrings** — `GroundingVerifier` and `CitationVerdict` updated to say
`[Dx]/[Tx]`.

## Scope / non-goals

- No change to the **mandatory** claim-support verifier (`safety.py`), which
  already validates `T*` IDs — only the optional secondary grounding pass.
- No signature change for existing callers: `tool_evidence` defaults to `None`,
  so `verify(answer, context)` behaves exactly as before.
- No new evidence plumbing beyond forwarding the already-computed `tool_evidence`.
- `D*` IDs keep precedence on the unlikely event of an id collision (`setdefault`).

## Verification

- New unit tests (`tests/unit/test_grounding.py`): a grounded `[T1]` citation
  (found + grounded + retained), a dangling `[T9]` (flagged + stripped), a mixed
  `[D1]`+`[T1]` answer, and a pipeline test asserting `generate_answer` forwards
  `tool_evidence` to `verify`.
- Regression: all existing `test_grounding.py` cases stay green (the widened
  regex must not change `[Dx]`-only behavior).
- Related suites green: `test_grounded_generation.py`, `test_context_pipeline.py`,
  `test_mcp_server.py`, `test_rag_tool_evidence.py`, `test_rag_safety.py`.
- `ruff check` clean.
