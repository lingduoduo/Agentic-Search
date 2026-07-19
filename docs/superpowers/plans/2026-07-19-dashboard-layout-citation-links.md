# Dashboard Layout Reorg + Citation-Link Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make search results appear directly below the query, and make `[RxQyDz]` citation links in the answer navigate to their source cards.

**Architecture:** Three surgical changes: (1) a backend helper rewrite so search-mode source cards carry the agent's real `[RxQyDz]` labels instead of re-indexed `[Dn]`; (2) a frontend regex broadening so those labels linkify; (3) a JSX reorder moving the admin/analytics panels below the results.

**Tech Stack:** Python (FastAPI backend), React 19 + TypeScript + Vite (frontend), pytest, vitest.

## Global Constraints

- Never commit to `main`; work on branch `fix/dashboard-layout-citation-links` (already created).
- Chat/RAG-mode citations use `[D1]` and must keep working — every change must remain backward-compatible with `[D\d+]`.
- Match existing code style; touch only what each task requires.

---

### Task 1: Backend — preserve `[RxQyDz]` labels on search-mode source cards

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_search_agent_documents`, ~line 597; add import near line 54)
- Test: `tests/unit/servers/web/test_loop_runners.py` (`_search_output` helper ~line 22; `test_run_search_agent_returns_canonical_tuple` ~line 36)

**Interfaces:**
- Consumes: `output.context.rounds` — a `list[list[SearchContext]]` where each `SearchContext` has a `.results: list[SearchResult]`. `citation_key(round, query, doc)` from `src.context.search` returns the string `"R{round}Q{query}D{doc}"` (all 1-based).
- Produces: `_search_agent_documents(output) -> list[ContextDocument]` where each doc's `id` = `citation_key(r, q, d)` (e.g. `"R1Q1D1"`), so `document.citation` == `"[R1Q1D1]"`, matching the labels in `output.final_answer`.

- [ ] **Step 1: Update the test helper and rewrite the assertion (failing test)**

In `tests/unit/servers/web/test_loop_runners.py`, update `_search_output` to carry `rounds` on the fake context, and change the search-agent test to build a real round structure and assert the preserved label.

Replace the `_search_output` helper (~line 22):

```python
def _search_output(*, turns=None, rounds=None, num_turns=1, final_answer="grounded", trace=("e1",)):
    if rounds is None:
        rounds = [list(turns)] if turns else []
    if turns is None:
        turns = [ctx for round_ctxs in rounds for ctx in round_ctxs]
    context = types.SimpleNamespace(turns=turns, rounds=rounds)
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=num_turns,
        final_answer=final_answer,
        context=context,
        control_flow_trace=list(trace),
    )
```

Replace the body of `test_run_search_agent_returns_canonical_tuple` (~line 36) so it asserts the `[R1Q1D1]` label:

```python
@pytest.mark.asyncio
async def test_run_search_agent_returns_canonical_tuple(monkeypatch):
    ctx = types.SimpleNamespace(
        results=[SearchResult(contents="Title\nbody", url=None)]
    )
    monkeypatch.setattr(
        "src.agents.search.SearchAgentLoop.run",
        AsyncMock(return_value=_search_output(rounds=[[ctx]])),
    )
    answer, citations, documents, intent, extra = await web_app._run_search_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        on_turn=None,
        on_trace=None,
    )
    assert answer == "grounded"
    assert intent == "search"
    assert len(documents) == 1
    assert documents[0].citation == "[R1Q1D1]"
    assert citations == ["[R1Q1D1]"]
    assert extra["num_turns"] == 1
    assert extra["control_flow_trace"] == ["e1"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/servers/web/test_loop_runners.py::test_run_search_agent_returns_canonical_tuple -v`
Expected: FAIL — current `_search_agent_documents` iterates `.turns` and produces `[D1]`, so `documents[0].citation == "[D1]"` ≠ `"[R1Q1D1]"`.

- [ ] **Step 3: Add the `citation_key` import**

In `src/internal/servers/web/app.py`, change the existing import line (line 54):

```python
from src.context.search import SearchResult
```

to:

```python
from src.context.search import SearchResult, citation_key
```

- [ ] **Step 4: Rewrite `_search_agent_documents`**

Replace the whole function (~line 597). `ContextDocument` is a frozen dataclass, so
we build each doc directly with the label as its `id` (reusing `from_search_result`
only to split title/content):

```python
def _search_agent_documents(output) -> list[ContextDocument]:
    """Extract documents from a SearchAgentLoop output, preserving each result's
    real ``[RxQyDz]`` citation label so answer markers resolve to source cards.

    Rounds/queries/docs are enumerated 1-based to match
    ``SearchAgentLoop._format_round_information`` (the labels the model cited).
    Dedup is intentionally skipped here: every cited label needs its own card so
    no citation dangles, even when the same doc is retrieved under two queries.
    """
    documents: list[ContextDocument] = []
    rounds = getattr(output.context, "rounds", None) or []
    for round_idx, round_ctxs in enumerate(rounds, 1):
        for query_idx, ctx in enumerate(round_ctxs, 1):
            for doc_idx, result in enumerate(ctx.results, 1):
                key = citation_key(round_idx, query_idx, doc_idx)
                base = ContextDocument.from_search_result(result, index=doc_idx)
                documents.append(
                    ContextDocument(
                        id=key,
                        title=base.title,
                        content=base.content,
                        url=base.url,
                        score=base.score,
                        metadata=base.metadata,
                    )
                )
    return documents
```

Note: `_dedupe_documents` is no longer called here. Leave the function defined (other paths may use it) — do not delete it.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/unit/servers/web/test_loop_runners.py -v`
Expected: PASS (all tests in the file, including the agentic-rag ones which use `turns=`).

- [ ] **Step 6: Run the broader web unit suite for regressions**

Run: `pytest tests/unit/test_execution_fallbacks.py tests/unit/test_search_filters_plumbing.py tests/unit/servers/web/ -q`
Expected: PASS. If any test asserts a `[Dn]` citation from a search-agent path, update its expectation to the `[RxQyDz]` label (grep the failure for `[D` and switch to the round/query/doc label the fixture implies).

- [ ] **Step 7: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_loop_runners.py
git commit -m "fix: preserve [RxQyDz] citation labels on search-mode source cards"
```

---

### Task 2: Frontend — linkify `[RxQyDz]` citations

**Files:**
- Modify: `web/src/components/AnswerPanel.tsx` (`CITATION_RE` line 16; the two `/^\[D\d+\]$/` tests at lines 25 and 35)
- Test: `web/src/components/__tests__/AnswerPanel.test.tsx`

**Interfaces:**
- Consumes: answer text containing `[R1Q1D1]` (search) or `[D1]` (chat/RAG).
- Produces: `<a href="#source-[R1Q1D1]" class="citation-link">[R1Q1D1]</a>` for search labels; `[D1]` behavior unchanged.

- [ ] **Step 1: Add a failing test for the `[RxQyDz]` link**

In `web/src/components/__tests__/AnswerPanel.test.tsx`, add after the existing `[D1]` test (~line 136):

```tsx
it("renders citation [R1Q1D1] as an anchor link", () => {
  render(
    <AnswerPanel
      answer="See [R1Q1D1] for details."
      citations={[]}
    />
  );
  const link = screen.getByRole("link", { name: "[R1Q1D1]" });
  expect(link).toHaveAttribute("href", "#source-[R1Q1D1]");
  expect(link).toHaveClass("citation-link");
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm test -- AnswerPanel`
Expected: FAIL — `getByRole("link", { name: "[R1Q1D1]" })` finds no link because `CITATION_RE` only matches `[D\d+]`.

- [ ] **Step 3: Broaden the regex and the two guard tests**

In `web/src/components/AnswerPanel.tsx`, change line 16:

```tsx
const CITATION_RE = /(\[D\d+\])/;
```

to:

```tsx
const CITATION_RE = /(\[(?:R\d+Q\d+)?D\d+\])/;
```

Change the guard test on line 25 (inside the array branch of `linkifyCitations`):

```tsx
        /^\[D\d+\]$/.test(part)
```

to:

```tsx
        /^\[(?:R\d+Q\d+)?D\d+\]$/.test(part)
```

Change the identical guard test on line 35 (the string branch):

```tsx
    /^\[D\d+\]$/.test(part)
```

to:

```tsx
    /^\[(?:R\d+Q\d+)?D\d+\]$/.test(part)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd web && npm test -- AnswerPanel`
Expected: PASS — both the new `[R1Q1D1]` test and the existing `[D1]` test pass.

- [ ] **Step 5: Type-check**

Run: `cd web && npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/AnswerPanel.tsx web/src/components/__tests__/AnswerPanel.test.tsx
git commit -m "fix: linkify [RxQyDz] search citations in AnswerPanel"
```

---

### Task 3: Layout — move admin/analytics panels below the results

**Files:**
- Modify: `web/src/App.tsx` (move the `adminSummary` + analytics JSX block from ~lines 368-375 to after `results-layout`, ~line 425)
- Test: `web/src/components/__tests__/App.test.tsx`

**Interfaces:**
- Consumes: nothing new — pure JSX reorder of already-rendered components.
- Produces: DOM order where `.results-layout` precedes the admin/analytics panels.

- [ ] **Step 1: Add a failing DOM-order test**

The default suite mocks `getAdminSummary` as **rejected** (App.test.tsx lines
11-14), so `AdminOverview` is normally absent. This test overrides that mock to
resolve, waits for the `.admin-overview` panel (root class confirmed in
`AdminOverview.tsx:31`), then asserts it comes AFTER `.results-layout` in DOM
order. Add it as a bare top-level `it(...)` at the end of the file (the file uses
bare top-level `it` blocks, not a wrapping `describe`):

```tsx
it("renders results-layout above the admin overview panel", async () => {
  vi.mocked(api.getAdminSummary).mockResolvedValueOnce({
    health_label: "OK",
    health_score: 1,
    metrics: [],
    sections: [],
  });
  render(<App />);
  const admin = await screen.findByLabelText(/admin and observability/i);
  const layout = document.querySelector(".results-layout");
  expect(layout).toBeInTheDocument();
  expect(
    layout!.compareDocumentPosition(admin) &
      Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
});
```

`api` is already imported (`import * as api from "../../api"` at line 16), and
`AdminOverview`'s root `<section aria-label="Admin and observability">` is what
`findByLabelText` targets.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm test -- App`
Expected: FAIL — with the admin panel currently rendered ABOVE `.results-layout`,
`DOCUMENT_POSITION_FOLLOWING` is not set, so the assertion is falsy.

- [ ] **Step 3: Move the JSX block**

In `web/src/App.tsx`, cut this block (lines ~368-375):

```tsx
        {adminSummary && <AdminOverview summary={adminSummary} />}
        {(analyticsByLLM || analyticsByPersona || analyticsByFlow) && (
          <AnalyticsDashboard
            byLLM={analyticsByLLM}
            byPersona={analyticsByPersona}
            byFlow={analyticsByFlow}
          />
        )}
```

Delete it from its current position (between the dev console block and `{showConnectors && ...}`), and paste it immediately AFTER the closing `</div>` of `results-layout` (the `<div className={`results-layout...`}>...</div>`, closing ~line 425), before the closing `</section>` of `.workspace`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npm test -- App`
Expected: PASS.

- [ ] **Step 5: Type-check and full frontend test run**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add web/src/App.tsx web/src/components/__tests__/App.test.tsx
git commit -m "feat: move admin/analytics panels below results so results follow the query"
```

---

## Final verification

- [ ] Run `pytest -q` (backend suite) — expect green (or only pre-existing failures unrelated to this change; check `main` if unsure per the CI-torch-gap note).
- [ ] Run `cd web && npm run typecheck && npm test` — expect green.
- [ ] Manual smoke (optional, per web-stack startup memory): start the 3-process stack, run a search-mode query, confirm the answer's `[R…Q…D…]` citations are clickable and scroll to the matching source card, and confirm the answer/sources sit directly under the query box with admin/analytics below.
