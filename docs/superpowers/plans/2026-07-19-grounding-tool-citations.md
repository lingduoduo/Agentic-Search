# Grounding Tool [Tx] Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the optional `GroundingVerifier` check tool `[Tx]` citations against their tool-result text (flagging/stripping dangling ones), matching how `[Dx]` document citations are already handled.

**Architecture:** Widen the citation regex to `[DT]\d+`, give `verify()` an optional `tool_evidence` list, build a unified `id → text` map (D* from `context.documents`, T* from tool evidence), and forward the already-computed `tool_evidence` local at the `pipeline.py` call site. Default `None` ⇒ existing `verify(answer, context)` behavior unchanged.

**Tech Stack:** Python, pytest.

## Global Constraints

- Opt-in path only (`verify_grounding=True`); no behavior change when tool evidence is absent.
- No change to the mandatory claim-support verifier (`safety.py`).
- `tool_evidence` defaults to `None` — no existing call site breaks.
- `D*` IDs take precedence over `T*` on id collision (`setdefault`).
- Never commit to `main`; work on branch `fix/grounding-verify-tool-citations` (already created).

---

### Task 1: Ground-check [Tx] citations (TDD)

**Files:**
- Modify: `src/context/grounding.py` (`_CITATION_RE`, `_split_sentences`, `GroundingVerifier.verify`)
- Modify: `src/context/pipeline.py` (`verify_grounding` call site, ~line 127)
- Modify: `src/context/models.py` (`CitationVerdict` docstring)
- Test: `tests/unit/test_grounding.py`

**Interfaces:**
- Consumes: `EvidenceSource` (existing), the `tool_evidence` local in `generate_answer` (existing).
- Produces: `GroundingVerifier.verify(answer, context, tool_evidence=None)` — new optional param; `CitationVerdict` for `[Tx]` citations.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_grounding.py`, add an `EvidenceSource` import + a `_tool_ev` helper, then:
- `test_verify_grounds_tool_citation` — `[T1]` with matching tool text ⇒ `document_found`, `is_grounded`, retained in `answer_clean`.
- `test_verify_dangling_tool_citation_flagged_and_stripped` — `[T9]` with no tool evidence ⇒ not found, not grounded, stripped.
- `test_verify_mixed_doc_and_tool_citations` — `[D1]`+`[T1]` both found and retained.
- `test_generate_answer_forwards_tool_evidence_to_verifier` — monkeypatch `verify` to capture its `tool_evidence` arg; assert `generate_answer` forwards the tool sources.

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_grounding.py -k "tool_citation or tool_evidence or mixed_doc_and_tool" -q`
Expected: FAIL — `TypeError: verify() got an unexpected keyword argument 'tool_evidence'`.

- [x] **Step 3: Widen the regex**

In `grounding.py`: `_CITATION_RE` → `re.compile(r"\[([DT]\d+)\]")`; `_split_sentences` lookahead → `(?!\[[DT]\d+\])`.

- [x] **Step 4: Add tool_evidence to `verify` and the unified id→text map**

Import `EvidenceSource`. Add `tool_evidence: list[EvidenceSource] | None = None`. Replace the doc-only map with `text_by_id = {doc.id: doc.content ...}` + `setdefault(source.id, source.text)` for each tool source; score against `evidence_text`.

- [x] **Step 5: Forward tool_evidence at the call site**

In `pipeline.py`, pass `tool_evidence=tool_evidence` into `GroundingVerifier().verify(...)` (the local is already computed above the block).

- [x] **Step 6: Update docstrings**

`GroundingVerifier` and `CitationVerdict` → `[Dx]/[Tx]`.

- [x] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_grounding.py -q`
Expected: PASS — 16 tests (12 existing regression + 4 new).

- [x] **Step 8: Run related suites for regressions**

Run: `python -m pytest tests/unit/test_grounding.py tests/unit/test_grounded_generation.py tests/unit/test_context_pipeline.py tests/unit/test_mcp_server.py -q`
Expected: PASS (104). Also `test_rag_tool_evidence.py`, `test_rag_safety.py`, `test_tool_categories.py` green individually.

- [x] **Step 9: Lint + commit**

Run: `ruff check src/context/grounding.py src/context/pipeline.py src/context/models.py tests/unit/test_grounding.py` — clean, then commit.

---

## Final verification

- [x] `python -m pytest tests/unit/test_grounding.py -q` — 16 green.
- [x] Existing `[Dx]`-only tests unchanged by the widened regex.
- [x] `ruff check` clean on all touched files.
- [x] `verify(answer, context)` (no `tool_evidence`) behaves exactly as before.
