# Prune the Orphaned Onyx Ingestion Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete ~5,800 LOC of dead Onyx-heritage async ingestion code (background worker fleet, `servers/indexing`, connector classes, `chunk_content_enrichment.py`, `document_index/indexing.py`) with zero live-behavior change.

**Architecture:** Sever the two live edges + the lazy re-export blocks first, then delete bottom-up. Ordering matters: cluster TESTS are deleted FIRST (removing a test never breaks collection), THEN the connector re-exports are severed (so no surviving test imports a removed name), THEN code is deleted from consumers down to leaves. Spec: `docs/superpowers/specs/2026-07-24-prune-onyx-ingestion-cluster-design.md`.

**Tech Stack:** Python, pytest, ruff.

## Global Constraints

- Branch: `chore/prune-onyx-ingestion-cluster`, created off `origin/main`. Never commit to `main`.
- Zero change to live runtime behavior — only dead cluster code is removed.
- `python -c "import src"` MUST succeed after every task (the `src/__init__.py` eager re-export is the edge that breaks package import if mishandled).
- `ruff check .` and `pytest` must pass at the end of every task.
- **KEEP** (do NOT delete): `src/internal/connectors/models.py`; the `document_index` build tool (`index_builder`/`cli`/`chunking`/`embedding`/`pipeline`/`faiss_io`/`_common`/`models`/`retrieval`); `document_index/__init__.py`'s `_BUILDER_EXPORTS`; the `weaviate` docker-compose service; the `servers/connectors/api.py` router (DB-only).
- All Weaviate code is PR3 — do not touch it here.

---

### Task 1: Delete the cluster-only test files

Deleting tests never breaks collection, and doing it first lets later edge-severing edits stay green (no surviving test imports a removed connector name or lazy export).

**Files:**
- Delete: `tests/unit/servers/backgroundworker/test_docprocessing.py`
- Delete: `tests/unit/servers/backgroundworker/test_heavy_worker.py`
- Delete: `tests/unit/servers/backgroundworker/test_indexing_pipeline_facade.py`
- Delete: `tests/unit/test_connectors.py`
- Delete: `tests/unit/test_connectors_poll_slim.py`
- Delete: `tests/unit/test_indexing_server_facade.py`
- Delete: `tests/unit/document_index/test_chunk_content_enrichment.py`

- [ ] **Step 1: Confirm each targets only cluster code**

Run: `for f in tests/unit/servers/backgroundworker/test_docprocessing.py tests/unit/servers/backgroundworker/test_heavy_worker.py tests/unit/servers/backgroundworker/test_indexing_pipeline_facade.py tests/unit/test_connectors.py tests/unit/test_connectors_poll_slim.py tests/unit/test_indexing_server_facade.py tests/unit/document_index/test_chunk_content_enrichment.py; do echo "== $f =="; grep -E "^(from|import) " "$f" | grep -E "backgroundworker|servers.indexing|connectors\.(basic|interface|web)|connectors import|chunk_content_enrichment|document_index import"; done`
Expected: every import shown is from a cluster module (backgroundworker, servers.indexing, connectors classes, chunk_content_enrichment, or the lazy document_index indexing exports). If a test imports a SURVIVING module for a non-cluster reason, STOP and report BLOCKED.

- [ ] **Step 2: Delete them**

```bash
git rm tests/unit/servers/backgroundworker/test_docprocessing.py \
  tests/unit/servers/backgroundworker/test_heavy_worker.py \
  tests/unit/servers/backgroundworker/test_indexing_pipeline_facade.py \
  tests/unit/test_connectors.py \
  tests/unit/test_connectors_poll_slim.py \
  tests/unit/test_indexing_server_facade.py \
  tests/unit/document_index/test_chunk_content_enrichment.py
```

- [ ] **Step 3: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass (fewer tests collected).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(ingestion-cluster): remove cluster-only test files

Deleted ahead of the code so the edge-severing edits stay green.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Sever the live/lazy edges into the cluster

After this task, no live or lazy import path reaches the cluster; the cluster code still exists but is fully orphaned.

**Files:**
- Modify: `src/__init__.py:50-81` (connector re-exports)
- Modify: `src/internal/connectors/__init__.py` (drop `.basic`/`.interface`)
- Modify: `src/internal/document_index/__init__.py` (drop `_INDEXING_EXPORTS`)
- Modify: `src/internal/servers/web/debug_router.py:62-83` (`/workers` handler)
- Modify: `docker/docker-compose.yml:173-210` (remove worker services)
- Modify: `tests/unit/document_index/test_imports.py` (remove one test)

- [ ] **Step 1: Prove no live (non-test) consumer of the removed connector class names**

Run: `grep -rnE "from src import (BaseConnector|CheckpointedConnector|CheckpointedConnectorWithPermSync|CredentialsConnector|CredentialsProviderInterface|EventConnector|HierarchyConnector|InMemoryConnector|LoadConnector|LocalFileConnector|LocalFilePollConnector|LocalFileSlimConnector|LocalFileSlimConnectorWithPermSync|OAuthConnector|PollConnector|Resolver|SearchConnector|SlimConnector|SlimConnectorWithPermSync|StaticCredentialsProvider|batched)" src/ examples/ --include="*.py"`
Expected: no output. (Tests that used these were deleted in Task 1.) If any live module consumes one, STOP and report BLOCKED with the location.

- [ ] **Step 2: Edit `src/__init__.py` — replace the connector-class block (lines 50-81) with only the 5 model-backed keepers**

Replace the entire block from `from .internal.connectors import BaseConnector as BaseConnector` (line 50) through `from .internal.connectors import batched as batched` (line 81) with exactly:

```python
from .internal.connectors import ConnectorCheckpoint as ConnectorCheckpoint
from .internal.connectors import ConnectorFailure as ConnectorFailure
from .internal.connectors import Document as Document
from .internal.connectors import HierarchyNode as HierarchyNode
from .internal.connectors import SlimDocument as SlimDocument
```

(These five come from `connectors/models.py`, which stays. All other names on lines 50-81 are connector classes/interfaces being deleted.)

- [ ] **Step 3: Edit `src/internal/connectors/__init__.py` — keep only the `.models` re-exports**

Replace the whole file body (lines 3-37, the `.basic` and `.interface` imports) so only the models block remains:

```python
"""Connector data models (connector classes were removed — see PR2 cleanup)."""

from .models import ConnectorCheckpoint as ConnectorCheckpoint
from .models import ConnectorFailure as ConnectorFailure
from .models import Document as Document
from .models import HierarchyNode as HierarchyNode
from .models import SlimDocument as SlimDocument
```

- [ ] **Step 4: Edit `src/internal/document_index/__init__.py` — drop `_INDEXING_EXPORTS`, keep `_BUILDER_EXPORTS`**

Replace the file contents with:

```python
"""Document-index backends, text handling, and indexing entry points."""

_BUILDER_EXPORTS = {
    "IndexBuilder",
    "IndexBuilderConfig",
    "IndexingHeartbeatInterface",
}


def __getattr__(name: str):
    if name in _BUILDER_EXPORTS:
        from . import index_builder

        return getattr(index_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 5: Edit `src/internal/servers/web/debug_router.py` — gut the `/workers` handler**

Replace the `workers()` handler (lines 62-83) with the null-safe stub (keep the endpoint, drop the `MonitoringWorker` import):

```python
    @router.get("/workers")
    def workers() -> dict:
        """Indexing-pipeline snapshot placeholder.

        The async ingestion worker fleet was removed; this endpoint now always
        returns ``metrics: null``. Kept for Dev Console compatibility.
        """
        return {"metrics": None}
```

- [ ] **Step 6: Edit `docker/docker-compose.yml` — remove the two worker services**

Delete lines 173-210 inclusive (the entire `worker-light:` service, the `worker-heavy:` service, and the blank line between/after them), so the `weaviate` service (unchanged) and the `volumes:` block that follows remain intact. Do NOT touch the `weaviate` service or `weaviate_data` volume.

- [ ] **Step 7: Edit `tests/unit/document_index/test_imports.py` — remove the enrichment smoke test**

Delete exactly:

```python
def test_chunk_content_enrichment_importable():
    import src.internal.document_index.chunk_content_enrichment  # noqa: F401
```

Leave every other test in the file.

- [ ] **Step 8: Verify green (import + lint + tests)**

Run: `python -c "import src" && python -c "import src.internal.connectors as c; print(c.Document, c.ConnectorFailure)" && ruff check . && pytest -q`
Expected: `import src` succeeds; the connectors models still resolve; ruff clean; pytest green.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore(ingestion-cluster): sever live/lazy edges into the dead cluster

- src/__init__.py: drop connector-class re-exports, keep the 5 model-backed names
- connectors/__init__.py: keep only .models re-exports
- document_index/__init__.py: drop _INDEXING_EXPORTS (keep _BUILDER_EXPORTS)
- debug_router.py: /workers now returns null-safe stub (drops MonitoringWorker)
- docker-compose.yml: remove worker-light/worker-heavy (weaviate untouched)
- test_imports.py: drop chunk_content_enrichment smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Delete the worker fleet and the `servers/indexing` pipeline

**Files:**
- Delete: `src/internal/servers/backgroundworker/` (whole dir)
- Delete: `src/internal/servers/indexing/` (whole dir, recursive)

- [ ] **Step 1: Confirm no surviving importer**

Run: `grep -rnE "servers\.backgroundworker|servers import backgroundworker|servers\.indexing|servers import indexing" src/ examples/ tests/ --include="*.py"`
Expected: no output (debug_router edge severed in Task 2; cluster tests deleted in Task 1). If anything shows, STOP and report BLOCKED.

- [ ] **Step 2: Delete both dirs**

```bash
git rm -r src/internal/servers/backgroundworker src/internal/servers/indexing
```

- [ ] **Step 3: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(ingestion-cluster): delete backgroundworker fleet + servers/indexing

No __main__ guard; nothing launched them. Zero surviving importer after the
edges were severed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Delete the connector classes

`servers/indexing` and the worker fleet (the only consumers of the classes) are already gone.

**Files:**
- Delete: `src/internal/connectors/basic.py`
- Delete: `src/internal/connectors/interface.py`
- Delete: `src/internal/connectors/web.py`

- [ ] **Step 1: Confirm no surviving importer**

Run: `grep -rnE "connectors\.(basic|interface|web)|from \.basic|from \.interface|from \.web" src/ examples/ tests/ --include="*.py"`
Expected: no output. (`connectors/__init__.py` and `src/__init__.py` were repointed to `.models` in Task 2.) If anything shows, STOP and report BLOCKED.

- [ ] **Step 2: Delete the three modules**

```bash
git rm src/internal/connectors/basic.py src/internal/connectors/interface.py src/internal/connectors/web.py
```

- [ ] **Step 3: Verify green + connectors.models still resolves**

Run: `python -c "import src.internal.connectors as c; print(c.Document, c.ConnectorFailure, c.SlimDocument)" && ruff check . && pytest -q`
Expected: models resolve; ruff clean; pytest green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(ingestion-cluster): delete connector classes (keep models.py)

basic.py/interface.py/web.py had no live consumers after the worker fleet and
servers/indexing were removed. connectors/models.py stays (live: Document,
ConnectorFailure, ConnectorStopSignal, ConnectorCheckpoint, HierarchyNode,
SlimDocument).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Delete the two orphaned `document_index` modules

Their only consumers (`servers/indexing/embedder.py`, `backgroundworker/docprocessing.py`) are gone; the `_INDEXING_EXPORTS` lazy path was removed in Task 2.

**Files:**
- Delete: `src/internal/document_index/chunk_content_enrichment.py`
- Delete: `src/internal/document_index/indexing.py`

- [ ] **Step 1: Confirm no surviving importer**

Run: `grep -rnE "chunk_content_enrichment|generate_enriched_content_for_chunk|cleanup_content_for_chunks|document_index\.indexing|document_index import indexing" src/ examples/ tests/ --include="*.py"`
Expected: no output. (The `models.py:660` mention of enrichment is a comment, not an import — if it appears, confirm it is a comment and continue.)

- [ ] **Step 2: Delete both modules**

```bash
git rm src/internal/document_index/chunk_content_enrichment.py src/internal/document_index/indexing.py
```

- [ ] **Step 3: Verify green**

Run: `python -c "import src.internal.document_index as di; print(di.IndexBuilder)" && ruff check . && pytest -q`
Expected: the surviving `_BUILDER_EXPORTS` lazy path resolves; ruff clean; pytest green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(ingestion-cluster): delete chunk_content_enrichment.py + indexing.py

Onyx contextual-RAG enrichment + batch-indexing orchestration; only consumers
were servers/indexing and docprocessing, now deleted.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update the docs that reference the removed dirs

**Files:**
- Modify: `docs/ingestion.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Find the references**

Run: `grep -n "backgroundworker\|servers/indexing\|connectors/" docs/ingestion.md docs/architecture.md`

- [ ] **Step 2: Update the prose/tree**

In `docs/architecture.md`, remove the `backgroundworker/` and connector-class lines from the directory tree / component list (keep any reference to `connectors/models.py` data models and the DB-backed `connectors` router, which remain). In `docs/ingestion.md`, remove or rewrite the section that describes the async `backgroundworker` ingestion pipeline as a live component. Keep descriptions accurate to what remains (the offline `index_builder` build tool + retrieval servers). Do not invent new content — delete the now-false claims.

- [ ] **Step 3: Confirm no dangling live references remain**

Run: `grep -n "backgroundworker\|servers/indexing" docs/ingestion.md docs/architecture.md`
Expected: no output (archived docs under `docs/superpowers/archive/` may keep theirs — leave those).

- [ ] **Step 4: Commit**

```bash
git add docs/ingestion.md docs/architecture.md
git commit -m "docs: drop references to the removed async ingestion cluster

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final full-suite gate + push + open PR

**Files:** none (verification + integration).

- [ ] **Step 1: Full gate**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 2: Confirm the diff shape**

Run: `git diff --stat origin/main...HEAD`
Expected: `backgroundworker/` + `servers/indexing/` dirs deleted; `connectors/{basic,interface,web}.py` deleted; `chunk_content_enrichment.py` + `indexing.py` deleted; 7 test files deleted; `src/__init__.py`, `connectors/__init__.py`, `document_index/__init__.py`, `debug_router.py`, `docker-compose.yml`, `test_imports.py`, `docs/ingestion.md`, `docs/architecture.md` edited; spec + this plan added. No other source touched. `connectors/models.py` NOT deleted; `weaviate` docker service intact.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin chore/prune-onyx-ingestion-cluster
gh pr create --base main --title "chore: prune the orphaned Onyx ingestion cluster (PR2 of 3)" --body "$(cat <<'EOF'
PR2 of a 3-PR dead-code-removal campaign on the indexing/document-store side (PR1: #461).

Removes ~5,800 LOC of dead Onyx-heritage async ingestion code, none of it reached
by the retrieval servers, web backend, example CLI, or the documented index_builder
build command:

- `src/internal/servers/backgroundworker/` — the worker fleet (no `__main__` guard; nothing launched it)
- `src/internal/servers/indexing/` — the Onyx ingestion pipeline
- `src/internal/connectors/{basic,interface,web}.py` — connector classes (models.py stays — it's live)
- `src/internal/document_index/chunk_content_enrichment.py` — Onyx contextual-RAG enrichment
- `src/internal/document_index/indexing.py` — batch-indexing orchestration (deferred here from PR1)

Two live edges into the cluster were severed first: the eager connector-class
re-export in `src/__init__.py` and the request-time `MonitoringWorker` import in
`debug_router.py` (`/workers` now returns a null-safe stub). The lazy
`_INDEXING_EXPORTS` block in `document_index/__init__.py` was removed
(`_BUILDER_EXPORTS` kept). `docker-compose.yml` loses the non-functional
`worker-light`/`worker-heavy` services; the `weaviate` service stays (it backs PR3).

Zero live-behavior change. `python -c "import src"` verified after every step;
full suite green; ruff + format clean.

Spec: `docs/superpowers/specs/2026-07-24-prune-onyx-ingestion-cluster-design.md`
Plan: `docs/superpowers/plans/2026-07-24-prune-onyx-ingestion-cluster.md`

Follow-up: PR3 (Weaviate, entirely). PR4 (ingestion DB tables) deferred.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opened against `main`.

---

## Self-Review

**Spec coverage:** Every spec DELETE target maps to a task — cluster tests (T1), backgroundworker + servers/indexing (T3), connector classes (T4), chunk_content_enrichment + indexing.py (T5). Every spec EDIT maps — src/__init__.py + connectors/__init__.py + document_index/__init__.py + debug_router.py + docker-compose.yml + test_imports.py trim (T2), docs (T6). Deletion-order constraint (sever edges before deleting; tests first) is realized by T1→T2→T3→T4→T5. Final gate + PR is T7. The KEEP list (models.py, build tool, _BUILDER_EXPORTS, weaviate service, connectors router) is carried in Global Constraints and asserted in T7 Step 2.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Each edit shows exact replacement code (for files whose content was read) or an exact target + exact grep verification. The docs task (T6) is inherently prose-editing; it names the exact false claims to remove and a grep gate, not a vague "update docs".

**Type consistency:** Removal + re-export trims only; the 5 kept connector names (`ConnectorCheckpoint`, `ConnectorFailure`, `Document`, `HierarchyNode`, `SlimDocument`) are identical across `src/__init__.py`, `connectors/__init__.py`, and the T4/T5 verification greps. `_BUILDER_EXPORTS` names are unchanged.
