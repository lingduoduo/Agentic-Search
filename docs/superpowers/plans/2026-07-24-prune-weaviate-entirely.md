# Remove Weaviate Entirely Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete ~1,430 LOC of dead Weaviate + now-orphaned `DocumentIndex` ABC code, drop the `weaviate-client` dependency, and remove Weaviate from docker/config — with zero live-behavior change.

**Architecture:** Sever the one env-gated `service.py` edge first (verify the `local` default still builds), then delete backend → write-path/factory/disabled → the orphaned `interfaces.py`, then clean docker/requirements/config. The default `RETRIEVAL_BACKEND=local` path is weaviate-independent. Spec: `docs/superpowers/specs/2026-07-24-prune-weaviate-entirely-design.md`.

**Tech Stack:** Python, pytest, ruff, docker-compose.

## Global Constraints

- Branch: `chore/prune-weaviate-entirely`, off `main` (post-PR1/PR2 merge). Never commit to `main`.
- Zero change to live runtime behavior — only dead Weaviate/interface code removed.
- `python -c "import src"` MUST succeed after every task.
- `ruff check .` + `pytest` green at the end of every task.
- **KEEP:** `document_index/models.py` + its types; `retrieval/backends/base.py`; `retrieval/service.py` (edit only); `VectorDbSettings`/`DISABLE_VECTOR_DB` wiring in `app_configs.py` (docstring-only touch); the `postgres`/`redis`/`retrieval`/`web` docker services.
- `interfaces.py` may be deleted ONLY after weaviate/factory/disabled are gone (they import it).

---

### Task 1: Drop the Weaviate backend branch in `service.py`

**Files:**
- Modify: `src/internal/retrieval/service.py` (`_build_weaviate_backend` + `_build_backend`)

- [ ] **Step 1: Remove `_build_weaviate_backend()`**

Delete this function entirely (currently lines 51-56) plus one surrounding blank line:

```python
def _build_weaviate_backend() -> RetrievalBackend:
    from .backends.weaviate import WeaviateBackend

    return WeaviateBackend(
        collection_name=os.environ["WEAVIATE_COLLECTION"],
    )
```

- [ ] **Step 2: Remove the weaviate branch in `_build_backend()` and fix the error message**

Replace the current `_build_backend`:

```python
def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    if name == "weaviate":
        return _build_weaviate_backend()
    raise ValueError(
        f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local, weaviate"
    )
```

with:

```python
def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    raise ValueError(
        f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local"
    )
```

- [ ] **Step 3: Verify the default path still builds + suite green**

Run: `python -c "import src" && python -c "import src.internal.retrieval.service as s; print(s._build_backend.__name__)" && ruff check . && pytest -q`
Expected: import OK; `_build_backend` resolves (no weaviate import); ruff clean; pytest green.

- [ ] **Step 4: Commit**

```bash
git add src/internal/retrieval/service.py
git commit -m "chore(weaviate): drop the RETRIEVAL_BACKEND=weaviate branch in service.py

Default local backend path unchanged; error message now lists 'local' only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Delete the Weaviate query backend

**Files:**
- Delete: `src/internal/retrieval/backends/weaviate.py`
- Delete: `tests/unit/retrieval/test_weaviate_backend.py`

- [ ] **Step 1: Confirm no surviving importer**

Run: `grep -rnE "backends\.weaviate|WeaviateBackend|from \.weaviate" src/ examples/ tests/ --include="*.py"`
Expected: only `tests/unit/retrieval/test_weaviate_backend.py` (about to be deleted). The `service.py` edge was removed in Task 1. If anything else shows, STOP and report BLOCKED.

- [ ] **Step 2: Delete both files**

```bash
git rm src/internal/retrieval/backends/weaviate.py tests/unit/retrieval/test_weaviate_backend.py
```

- [ ] **Step 3: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(weaviate): delete the opt-in WeaviateBackend query backend

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Delete the Weaviate write path, factory, and disabled index

**Files:**
- Delete: `src/internal/document_index/weaviate/` (whole dir)
- Delete: `src/internal/document_index/factory.py`
- Delete: `src/internal/document_index/disabled.py`
- Delete: `tests/unit/document_index/test_weaviate.py`, `test_factory.py`, `test_disabled.py`
- Modify: `tests/unit/document_index/test_imports.py` (remove `test_disabled_importable`)

- [ ] **Step 1: Confirm no surviving non-interfaces importer**

Run: `grep -rnE "document_index\.weaviate|WeaviateDocumentIndex|document_index\.factory|get_default_document_index|get_all_document_indices|document_index\.disabled|DisabledDocumentIndex" src/ examples/ tests/ --include="*.py"`
Expected: hits only in the files being deleted here (their own definitions) and in `test_weaviate.py`/`test_factory.py`/`test_disabled.py` (also being deleted). No other `src/` importer. If anything else shows, STOP and report BLOCKED.

- [ ] **Step 2: Delete the modules and tests**

```bash
git rm -r src/internal/document_index/weaviate
git rm src/internal/document_index/factory.py src/internal/document_index/disabled.py
git rm tests/unit/document_index/test_weaviate.py tests/unit/document_index/test_factory.py tests/unit/document_index/test_disabled.py
```

- [ ] **Step 3: Trim `test_imports.py` — remove the disabled smoke test**

Delete exactly:

```python
def test_disabled_importable():
    import src.internal.document_index.disabled  # noqa: F401
```

Leave `test_interfaces_importable` for now (removed in Task 4) and `test_document_index_utils_importable`.

- [ ] **Step 4: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(weaviate): delete write path + factory + disabled index

WeaviateDocumentIndex/schema, get_default_document_index factory (zero src
callers), and the DisabledDocumentIndex no-op. document_index/models.py stays.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Delete the now-orphaned `DocumentIndex` ABC

`interfaces.py`'s only importers (weaviate/factory/disabled) are gone after Task 3.

**Files:**
- Delete: `src/internal/document_index/interfaces.py`
- Modify: `tests/unit/document_index/test_imports.py` (remove `test_interfaces_importable`)

- [ ] **Step 1: Confirm fully dead**

Run: `grep -rnE "document_index\.interfaces|from \.interfaces|from src.internal.document_index.interfaces|DocumentIndex\b" src/ examples/ tests/ --include="*.py"`
Expected: no `src/` importer of `document_index.interfaces` remains (the three importers were deleted in Task 3). Note: `DocumentIndex` as a bare word may appear in unrelated contexts — confirm any hit is NOT importing `document_index/interfaces.py`. The only remaining reference should be `test_imports.py`'s `test_interfaces_importable` (about to be removed). If a live `src/` module imports `document_index.interfaces`, STOP and report BLOCKED.

- [ ] **Step 2: Delete the module**

```bash
git rm src/internal/document_index/interfaces.py
```

- [ ] **Step 3: Trim `test_imports.py` — remove the interfaces smoke test**

Delete exactly:

```python
def test_interfaces_importable():
    import src.internal.document_index.interfaces  # noqa: F401
```

The file should now contain only `test_document_index_utils_importable`.

- [ ] **Step 4: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(weaviate): delete orphaned DocumentIndex ABC (interfaces.py)

Its only importers were the weaviate write path, factory, and disabled index,
all removed in the prior task.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Remove Weaviate from docker-compose + requirements

**Files:**
- Modify: `docker/docker-compose.yml`
- Modify: `requirements.txt`
- Modify: `requirements-unit-test.txt`

- [ ] **Step 1: Edit `docker/docker-compose.yml` — remove all Weaviate**

Make these removals (do NOT touch other services):
- The `weaviate:` service block (currently lines 97-117, from `  weaviate:` through its `networks:` `- agentic_search_default`), plus one surrounding blank line.
- The `weaviate_data:` named-volume declaration (currently line 176).
- The five `WEAVIATE_*` env entries in the shared `&app-env` anchor (currently lines 39-43: `WEAVIATE_HTTP_HOST`, `WEAVIATE_HTTP_PORT`, `WEAVIATE_GRPC_HOST`, `WEAVIATE_GRPC_PORT`, `WEAVIATE_API_KEY`).
- The header comment (currently line 6) `#   docker compose -f docker/docker-compose.yml up postgres redis weaviate` → drop the trailing `weaviate` so it reads `... up postgres redis`.

- [ ] **Step 2: Edit `requirements.txt` + `requirements-unit-test.txt`**

Remove the `weaviate-client>=4.9.0` line from each (currently `requirements.txt:11` and `requirements-unit-test.txt:9`).

- [ ] **Step 3: Verify valid YAML + no lingering weaviate refs + suite green**

Run: `python -c "import yaml; yaml.safe_load(open('docker/docker-compose.yml'))" && echo YAML_OK`
Expected: `YAML_OK` (compose file still parses).
Run: `grep -rniE "weaviate" docker/docker-compose.yml requirements.txt requirements-unit-test.txt`
Expected: no output.
Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add docker/docker-compose.yml requirements.txt requirements-unit-test.txt
git commit -m "chore(weaviate): remove weaviate docker service + weaviate-client dep

Drops the weaviate service/volume/env from docker-compose (no service depends_on
it) and the weaviate-client requirement (no surviving importer).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Fix the stale Weaviate config docstring

**Files:**
- Modify: `src/internal/configs/app_configs.py` (`VectorDbSettings` docstring only)

- [ ] **Step 1: Update the docstring**

At `class VectorDbSettings:` (currently line 111-112), replace the docstring:

```python
    """Settings for the Weaviate vector database backend."""
```

with:

```python
    """Vector-DB feature flags (vestigial: the Weaviate backend was removed;
    kept for AppSettings compatibility)."""
```

Do NOT change `disable_vector_db` or any field, and do NOT touch `load_app_settings`.

- [ ] **Step 2: Verify green**

Run: `python -c "import src" && ruff check . && pytest -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/internal/configs/app_configs.py
git commit -m "docs: mark VectorDbSettings vestigial after Weaviate removal (docstring only)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final full-suite gate + push + open PR

**Files:** none (verification + integration).

- [ ] **Step 1: Full gate**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 2: Confirm no weaviate remnants + diff shape**

Run: `grep -rniE "weaviate" src/ tests/ requirements.txt requirements-unit-test.txt docker/docker-compose.yml --include="*.py" --include="*.txt" --include="*.yml"`
Expected: no output (archived docs under `docs/superpowers/archive/` are out of scope and not grepped here).
Run: `git diff --stat main...HEAD`
Expected: `weaviate/` dir + `factory.py` + `disabled.py` + `interfaces.py` + `backends/weaviate.py` deleted; 4 test files deleted; `service.py`, `docker-compose.yml`, `requirements.txt`, `requirements-unit-test.txt`, `test_imports.py`, `app_configs.py` edited; spec + this plan added. `document_index/models.py` NOT deleted.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin chore/prune-weaviate-entirely
gh pr create --base main --title "chore: remove Weaviate entirely (PR3 of 3)" --body "$(cat <<'EOF'
PR3 (final) of the indexing/document-store simplification campaign (PR1 #461, PR2 #462).

Removes Weaviate entirely — the dead write path, the opt-in query backend, and all
wiring — plus the now-orphaned DocumentIndex ABC. ~1,430 LOC of source removed; the
weaviate-client dependency dropped.

**Deleted**
- `src/internal/document_index/weaviate/` — WeaviateDocumentIndex + schema
- `src/internal/document_index/factory.py` — get_default_document_index (zero src callers; orphan even before this PR)
- `src/internal/document_index/disabled.py` — DisabledDocumentIndex no-op
- `src/internal/retrieval/backends/weaviate.py` — opt-in WeaviateBackend
- `src/internal/document_index/interfaces.py` — the DocumentIndex ABC + capability protocols (only the three files above implemented/used it)
- 4 Weaviate-cluster test files

**Edited**
- `service.py` — dropped the `RETRIEVAL_BACKEND=weaviate` branch; the default `local` path is unchanged and still constructs via `from_env()`
- `docker-compose.yml` — removed the weaviate service, volume, and WEAVIATE_* env (no service depends_on weaviate)
- `requirements.txt` + `requirements-unit-test.txt` — dropped weaviate-client
- `test_imports.py` — dropped the disabled/interfaces smoke tests
- `app_configs.py` — marked VectorDbSettings vestigial (docstring only; kept wired)

Zero live-behavior change: the edge audit found no re-export traps, and the default
`local` retrieval backend is weaviate-independent. `python -c "import src"` verified
after every step; full suite green; ruff + format clean; compose file still parses.

Spec: `docs/superpowers/specs/2026-07-24-prune-weaviate-entirely-design.md`
Plan: `docs/superpowers/plans/2026-07-24-prune-weaviate-entirely.md`

Completes the 3-PR campaign. Deferred PR4 (ingestion DB tables + connectors/documents
routers, incl. the /run endpoint that records an unused IndexAttemptRecord) remains a
separate follow-up. VectorDbSettings/DISABLE_VECTOR_DB left wired-but-vestigial.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opened against `main`.

---

## Self-Review

**Spec coverage:** Every spec DELETE maps to a task — backends/weaviate.py (T2), weaviate/ + factory + disabled (T3), interfaces.py (T4). Every spec EDIT maps — service.py (T1), docker-compose + requirements (T5), test_imports trims (T3+T4), app_configs docstring (T6). Order matches the spec (service edge first, interfaces last-of-code). Final gate + PR is T7.

**Placeholder scan:** No TBD/vague steps. Exact replacement code shown for service.py, test_imports trims, and the app_configs docstring; exact removal targets + grep/YAML verification for docker-compose and requirements.

**Type consistency:** Removal + one-branch edit. `_build_backend`/`_build_local_backend` signatures unchanged; the kept `RetrievalBackend`/`RetrievalResult` (from `backends/base.py`) are untouched. `RETRIEVAL_BACKEND` default `"local"` preserved.
