# SPEC — Optimize the Agentic RAG Loop

Target file: [src/agents/agentic_rag.py](src/agents/agentic_rag.py)
Tests: [tests/unit/test_agentic_rag.py](tests/unit/test_agentic_rag.py)

## 1. Objective

Harden `AgenticRAGLoop` against a set of correctness and robustness gaps found in
review, **without changing its externally observable happy-path behavior**. The loop
stays a query-enhance → iterative-retrieve → sufficiency-gate → gap-follow-up →
synthesize pipeline. This is a surgical optimization, not a redesign.

Target user: developers running the search stack; the loop is invoked by the web
backend and the agent CLL. Changes must be transparent to both.

### In scope (6 changes)

| # | Change | Type |
|---|--------|------|
| 1 | Initialize `merged` before the loop | Correctness (crash guard) |
| 2 | Normalize queries for dedup bookkeeping | Robustness |
| 3 | Robust document dedup key (normalized URL / content hash) | Robustness |
| 4 | Cap follow-up queries per round (`max_followups_per_round=5`) | Cost control |
| 5 | De-duplicate the `_clean_line` call in `_parse_gap_queries` | Cleanup |
| 6 | Bounded timeout on the LLM sufficiency check | Robustness (no hang) |

### Explicitly NOT in scope (decided with user)

- **Do not** flip the sufficiency-check fail default. On judge failure/timeout the loop
  still returns `True` (fail-open, stop looping). We bound it with a timeout instead.
- **Do not** add an empty-evidence canned fallback. When all retrieval fails the loop
  still calls `generate_answer` with zero docs (current behavior preserved).
- **Do not** add a timeout to gap analysis (`_generate_followup`) — only the
  sufficiency check.
- No other refactors, renames, prompt edits, or adjacent "improvements".

## 2. Detailed requirements

### Change 1 — initialize `merged`
Before `for round_idx in range(self.config.max_rounds):`, add:
```python
merged = SearchContextBundle(query=question, documents=[])
```
Guarantees `merged` is defined at the synthesis call for `max_rounds == 0`, empty
`current_queries`, or a first round with no novel queries.

### Change 2 — normalized query dedup
Add module-level helper:
```python
def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())
```
- `seen_queries` stores **normalized** forms.
- Novelty filter compares `_norm_query(q)` against `seen_queries`, for both the initial
  `current_queries` and follow-ups.
- **Retrieval still uses the original `q` string** — normalization only governs the
  dedup set, never what is sent to `retrieve_context`.

### Change 3 — robust document dedup key
Add module-level helper (add `import hashlib`):
```python
def _doc_key(doc: ContextDocument) -> str:
    if doc.url:
        return doc.url.strip().lower()
    return hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
```
Replace `key = doc.url or doc.content[:120]` with `key = _doc_key(doc)`. Full-content
hash removes the 120-char collision risk; URL is normalized for whitespace/case.
(Tracking-param stripping is out of scope.)

### Change 4 — cap follow-ups per round
- Add `max_followups_per_round: int = 5` to `AgenticRAGConfig`.
- After computing `novel_follow_ups`, truncate:
  `current_queries = novel_follow_ups[: self.config.max_followups_per_round]`.

### Change 5 — single-pass `_clean_line`
Rewrite the return of `_parse_gap_queries` to call `_clean_line` once per line:
```python
queries = []
for line in queries_section.splitlines():
    cleaned = _clean_line(line)
    if cleaned:
        queries.append(cleaned)
return queries
```

### Change 6 — bounded sufficiency-check timeout
- Add `sufficiency_timeout_s: float = 5.0` to `AgenticRAGConfig`.
- Make `_is_sufficient` **async**; run the blocking `complete` off-thread with a timeout:
  ```python
  raw = await asyncio.wait_for(
      asyncio.to_thread(self.llm.complete, [ChatMessage(role="user", content=prompt)]),
      timeout=self.config.sufficiency_timeout_s,
  )
  ```
  then apply the existing `_llm_text(...).strip().lower()` / `startswith("yes")` logic.
- On `asyncio.TimeoutError` **or** any exception: log a warning and return `True`
  (unchanged fail-open default — now bounded).
- Update the call site to `await self._is_sufficient(...)` (add `import asyncio`).

## 3. Commands

```bash
pip install -e .
ruff check . --fix && ruff format .
pytest tests/unit/test_agentic_rag.py -v      # focused
pytest                                         # full unit + regression gate
```

## 4. Project structure

Single-file change plus its test file. No new modules, no config/env changes, no API
surface changes. `AgenticRAGConfig` gains two fields with defaults, so all existing
call sites keep working unchanged.

## 5. Code style

- Match the file's existing idiom: module-level `_helpers`, `logger.warning` on caught
  exceptions, `from __future__ import annotations`, `dataclass` config.
- New helpers (`_norm_query`, `_doc_key`) sit beside `_clean_line`.
- No new third-party deps (`hashlib`, `asyncio` are stdlib).

## 6. Testing strategy

All 10 existing tests must stay green (they encode the LLM-response call order — the
changes add no extra `complete()` calls, so order is preserved). Add focused tests:

1. **`merged` init** — `AgenticRAGConfig(max_rounds=0)` → `run` returns an
   `AgenticRAGResult` with empty context, no crash.
2. **Query normalization** — initial queries differing only by case/whitespace
   (`"GPT-4 cost"` vs `"gpt-4 cost "`) trigger retrieval once.
3. **Doc dedup by content** — two retrieved docs with no URL and identical content but
   different ids collapse to one; distinct content stays separate.
4. **Follow-up cap** — gap analysis returning 8 queries retrieves ≤5 in the next round.
5. **Sufficiency timeout** — a `complete` that sleeps past `sufficiency_timeout_s`
   returns `True` (fail-open) and the loop proceeds to synthesis without hanging (use a
   small `sufficiency_timeout_s` like `0.05`).

## 7. Boundaries

- **Always**: keep happy-path output identical; keep all existing tests passing; run
  `ruff` + full `pytest` before declaring done; use a feature branch (never commit to
  `main`); open a PR when complete with a spec + plan committed on the branch.
- **Ask first**: any change to prompts, the fail-open sufficiency default, empty-evidence
  behavior, or the loop's public signature.
- **Never**: flip the sufficiency default to fail-closed; add the canned empty-evidence
  message; strip URL tracking params; refactor unrelated code; touch other agent loops.
