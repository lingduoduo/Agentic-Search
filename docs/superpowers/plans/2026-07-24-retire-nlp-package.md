# Retire the natural_language_processing Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the ~2,300 LOC dead `natural_language_processing/` package after extracting its one live function (`cohere_rerank_api`) into `retrieval/`, with zero live-behavior change.

**Architecture:** Extract-then-delete. Task 1 moves `cohere_rerank_api` + `CohereBillingLimitError` into a self-contained `retrieval/cohere_rerank.py` and repoints `reranker.py` (behavior-preserving; the package is now fully orphaned). Task 2 deletes the whole package + its one test. Task 3 gates + PR. Spec: `docs/superpowers/specs/2026-07-24-retire-nlp-package-design.md`.

**Tech Stack:** Python, pytest, ruff.

## Global Constraints

- Branch: `chore/retire-nlp-package`, off `main`. Never commit to `main`.
- Zero live-behavior change: the cohere rerank path must work identically.
- `ruff check .` + `pytest` green at the end of every task.
- Do NOT touch `document_index/` or relocate `setup_logger` (out of scope — the new module keeps importing it from `document_index.utils`).

---

### Task 1: Extract `cohere_rerank_api` into `retrieval/cohere_rerank.py` and repoint the reranker

**Files:**
- Create: `src/internal/retrieval/cohere_rerank.py`
- Modify: `src/internal/retrieval/reranker.py` (the guarded import block, lines ~18-22)
- Test: `tests/unit/retrieval/test_reranker.py` (existing — should pass unchanged)

**Interfaces:**
- Produces: `retrieval.cohere_rerank.cohere_rerank_api`, `retrieval.cohere_rerank.CohereBillingLimitError`.
- `reranker.py` consumes `cohere_rerank_api` (name unchanged in the `reranker` namespace, so the existing patch target `src.internal.retrieval.reranker.cohere_rerank_api` still resolves).

- [ ] **Step 1: Create the new module (verbatim extraction)**

Create `src/internal/retrieval/cohere_rerank.py`:

```python
"""Cohere rerank API helper.

Extracted from the retired ``natural_language_processing`` package;
``retrieval/reranker.py`` is its only caller. Kept self-contained: importing this
module requires the optional ``cohere`` dependency, so callers import it inside a
try/except ImportError guard.
"""

from cohere import AsyncClient as CohereAsyncClient
from cohere.core.api_error import ApiError

from src.internal.document_index.utils import setup_logger

logger = setup_logger(__name__)


class CohereBillingLimitError(Exception):
    """Raised when Cohere rejects requests because the billing cap is reached."""


async def cohere_rerank_api(
    query: str, docs: list[str], model_name: str, api_key: str
) -> list[float]:
    cohere_client = CohereAsyncClient(api_key=api_key)
    try:
        response = await cohere_client.rerank(
            query=query, documents=docs, model=model_name
        )
    except ApiError as err:
        if err.status_code == 402:
            logger.warning(
                "Cohere rerank request rejected due to billing cap. Falling back to retrieval ordering until billing resets."
            )
            raise CohereBillingLimitError(
                "Cohere billing limit reached for reranking"
            ) from err
        raise
    results = response.results
    sorted_results = sorted(results, key=lambda item: item.index)
    return [result.relevance_score for result in sorted_results]
```

- [ ] **Step 2: Repoint the reranker import**

In `src/internal/retrieval/reranker.py`, replace the guarded import block:

```python
try:
    from src.internal.natural_language_processing.search_nlp_models import (
        cohere_rerank_api,
    )
except ImportError:
    cohere_rerank_api = None  # type: ignore[assignment]
```

with:

```python
try:
    from src.internal.retrieval.cohere_rerank import cohere_rerank_api
except ImportError:
    cohere_rerank_api = None  # type: ignore[assignment]
```

Leave everything else in `reranker.py` unchanged (the `cohere_rerank_api(...)` call site stays as-is).

- [ ] **Step 3: Verify the reranker + its tests still pass**

Run: `python -c "import src.internal.retrieval.reranker" && pytest tests/unit/retrieval/test_reranker.py -q && ruff check src/internal/retrieval/`
Expected: import OK; the reranker test suite passes (its patch of `src.internal.retrieval.reranker.cohere_rerank_api` still resolves because the name is imported into that module); ruff clean.

- [ ] **Step 4: Confirm the package is now fully orphaned**

Run: `grep -rn "natural_language_processing" src/ examples/ --include="*.py"`
Expected: only matches inside `src/internal/natural_language_processing/` itself (the package's own internal imports) — no external src importer remains. If any external importer survives, STOP and report BLOCKED.

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/cohere_rerank.py src/internal/retrieval/reranker.py
git commit -m "refactor(retrieval): extract cohere_rerank_api out of natural_language_processing

Self-contained retrieval/cohere_rerank.py (cohere_rerank_api + CohereBillingLimitError),
repointed reranker.py's guarded import. Behavior-identical; leaves the NLP package
fully orphaned for deletion.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Delete the natural_language_processing package + its test

**Files:**
- Delete: `src/internal/natural_language_processing/` (whole dir)
- Delete: `tests/unit/test_query_embedding_cache.py`

- [ ] **Step 1: Re-confirm no live importer (post-Task-1)**

Run: `grep -rn "natural_language_processing" src/ examples/ tests/ --include="*.py" | grep -v "src/internal/natural_language_processing/"`
Expected: only `tests/unit/test_query_embedding_cache.py` (about to be deleted). If any src/ importer remains, STOP and report BLOCKED.

- [ ] **Step 2: Delete the package and its test**

```bash
git rm -r src/internal/natural_language_processing
git rm tests/unit/test_query_embedding_cache.py
```

- [ ] **Step 3: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass. (The reranker's cohere path resolves from the new module; no other code referenced the package.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete the dead natural_language_processing package (~2,300 LOC)

Onyx heritage superseded by document_index/embedding.py + embedding_cache.py +
retrieval/ + the rerank server. Its one live function was extracted to
retrieval/cohere_rerank.py in the prior commit. Also removes the test-only,
superseded query_embedding_cache (live cache is document_index/embedding_cache.py).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Full gate + push + open PR

**Files:** none (verification + integration).

- [ ] **Step 1: Full gate**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 2: Confirm removal + cohere-absent degradation + diff shape**

Run: `grep -rn "natural_language_processing" src/ tests/ examples/ --include="*.py"`
Expected: no output.
Run: `git diff --stat main...HEAD`
Expected: `retrieval/cohere_rerank.py` added; `retrieval/reranker.py` edited (import line); the whole `natural_language_processing/` dir + `test_query_embedding_cache.py` deleted; spec + plan added. No `document_index/` change.

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin chore/retire-nlp-package
gh pr create --base main --title "chore: retire the dead natural_language_processing package" --body "$(cat <<'EOF'
Removes the ~2,300 LOC `natural_language_processing/` package — Onyx heritage superseded by `document_index/embedding.py`, `document_index/embedding_cache.py`, `retrieval/`, and the rerank server.

A simplification audit found the package's *only* live thread into the running system is one ~20-line async function, `cohere_rerank_api` (reached from `retrieval/reranker.py` when `RERANKER_PROVIDER=cohere`). Everything else — the `search_nlp_models.py` embedding/rerank stack, the BPE tokenizer utils, `constants.py`, `english_stopwords.py`, `_stubs.py`, and the test-only `query_embedding_cache.py` — has no live caller.

**Change:**
- Extracted `cohere_rerank_api` + `CohereBillingLimitError` into a self-contained `src/internal/retrieval/cohere_rerank.py` (next to its only caller)
- Repointed `retrieval/reranker.py`'s guarded import
- Deleted the entire `natural_language_processing/` package + `tests/unit/test_query_embedding_cache.py`

**Verified:** single guarded live caller; no `src/__init__.py` re-export trap; the `EmbeddingModel`/`RerankingModel` names elsewhere are unrelated config classes. The reranker test patches a name on the `reranker` module, unaffected by the moved import source. Behavior-identical; full suite green; ruff+format clean; `import src` works with `cohere` absent (guard preserved).

Note (separate, deferred): the "tokenization duplication" between the packages is a mirage — the live tokenizers serve different purposes and are justified-separate. The one remaining `document_index/` smell (generic `setup_logger` imported cross-package by `metrics/`) is left for a later relocation PR.

Spec: `docs/superpowers/specs/2026-07-24-retire-nlp-package-design.md`
Plan: `docs/superpowers/plans/2026-07-24-retire-nlp-package.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(If `gh pr create` fails with a GitHub GraphQL/5xx error, the branch is still pushed — retry until it succeeds.)

---

## Self-Review

**Spec coverage:** extraction + reranker repoint + orphan-confirmation (T1); package + test deletion (T2); gate + no-importer check + PR (T3). Every spec success-criterion maps to a step: behavior-identical cohere path (T1 S3), import-src-with-cohere-absent (T3 gate via the reranker guard), no `natural_language_processing` reference (T2 S1 + T3 S2), ruff+pytest green (each task), factory cohere path resolves (T1 S3 exercises `reranker`).

**Placeholder scan:** no vague steps — the new module and the reranker edit are given in full; deletions and greps are exact.

**Type consistency:** `cohere_rerank_api(query, docs, model_name, api_key)` signature is identical to the original and unchanged at the `reranker.py:118` call site; `CohereBillingLimitError` name preserved. The import name `cohere_rerank_api` is identical across the new module, the reranker import, and the test's patch target.
