# Retrieval Docs — Agentic Loop & Stacks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `docs/retrieval.md` to document the agentic search loop, the direct-first sufficiency gate, the two independent local retrieval stacks, the four routing layers, and two accuracy caveats — additively, without touching existing prose or any code.

**Architecture:** Documentation-only change to a single file. No code, no tests, no behavior change. Verification is source-accuracy (grep/read against the modules described) plus markdown well-formedness, not pytest.

**Tech Stack:** Markdown.

## Global Constraints

- **Docs only** — no `src/` or `tests/` changes; no behavior change.
- **Additive** — do not rewrite or delete existing sections (target: 0 deletions).
- Every documented claim must be verified against current source before writing.
- Not a code cleanup: the "dead code"/"duplication" leads were investigated and
  rejected as wired/tested/intentional (see
  `project_retrieval_orchestrators_separate` memory) — do not act on them here.
- Never commit to `main`; work on branch `docs/retrieval-agentic-loop-and-stacks` (already created).

---

### Task 1: Verify claims, then make the five additive edits

**Files:**
- Modify: `docs/retrieval.md`
- Read-only (verification): `src/internal/retrieval/backends/local.py`,
  `src/internal/servers/retrieval/{server,demo,hybrid}.py`,
  `src/context/grounding.py`, `src/agents/search/search.py`,
  `src/agents/components/{loop_controller,evidence_judge}.py`,
  `src/training/evaluation.py`,
  `src/internal/document_index/FILTER_SEMANTICS.md`.

**Interfaces:**
- Consumes: the current behavior of the modules above (facts only).
- Produces: new prose in `docs/retrieval.md`. No public interface change.

- [x] **Step 1: Verify every claim against source**

Confirm each fact before it goes in the doc:
- `LocalBackend._KNOWN_DOC_KEYS == {"id","title","text","contents","url"}` and
  `_apply_filters` matches only `metadata` keys (post-hoc, standard keys skipped).
- `server.py` exposes `POST /search`; `demo.py`/`hybrid.py` expose `POST /retrieve`.
- `grounding.py` `_CITATION_RE == r"\[(D\d+)\]"` (D-only).
- `search.py` action tags, `DEFAULT_SYSTEM_PROMPT` budget (`max_search_limit=5`,
  `max_url_fetch=3`), `[RxQyDz]` citation labelling.
- `loop_controller.py` `effective_search_limit` (clamped to `max_search_limit_cap`)
  and `ACCEPT`/`FORCE`/`REJECT`; `evidence_judge.py` `_to_score` blend;
  `evaluation.py` threshold gate defaults.
- `FILTER_SEMANTICS.md` path exists.

Expected: all confirmed (any mismatch → fix the prose to match code, not vice versa).

- [x] **Step 2: Edit 1 — two-stacks callout**

After the retrieval-servers table (before "Web search servers"), add a blockquote
distinguishing `demo.py`/`hybrid.py` (`/retrieve`) from
`server.py`→`RetrievalService`→`LocalBackend` (`/search`), and noting the
optimization layers apply to the latter only.

- [x] **Step 3: Edit 2 — direct-first sufficiency gate + agentic loop section**

Before "## Neural reranking", add "### The direct-first sufficiency gate"
(exact/fuzzy/semantic tier table + no-e5 caveat) and "## The agentic search loop"
(XML action-tag table, turn cycle, `[RxQyDz]` citations, and the
"### Adaptive budget and the sufficiency control layer" subsection).

- [x] **Step 4: Edit 3 — four-routing-layers table**

In "## Routing and query construction", after the first paragraph, add the
"Four routing layers, four jobs" table (intent / provider cascade /
retriever-target / transform).

- [x] **Step 5: Edit 4 & 5 — the two caveats**

Add the `LocalBackend` post-hoc-filter caveat in the auto-routed request section,
and the grounding-verifier `[Dx]`-only caveat in "### Approved tool evidence".

- [x] **Step 6: Verify the rendered doc**

Run: `git diff --stat docs/retrieval.md` (expect additive only), confirm
`FILTER_SEMANTICS.md` path resolves, and check headings nest / tables balance.
Expected: `1 file changed, N insertions(+)`, 0 deletions.

- [x] **Step 7: Commit**

```bash
git add docs/retrieval.md
git commit -m "docs(retrieval): document the agentic loop, two local stacks, and routing layers"
```

---

## Final verification

- [x] `git diff --stat` shows `docs/retrieval.md` additive only (0 deletions).
- [x] Referenced `src/internal/document_index/FILTER_SEMANTICS.md` exists.
- [x] Pre-commit (trailing-whitespace / end-of-file) passes on push.
- [x] All documented claims traced to a verified source location.
