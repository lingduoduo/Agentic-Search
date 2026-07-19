# Rerank URL Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `rerank_url` from an `AGENTIC_SEARCH_RERANK_URL` env var through `ServiceSettings` into `SearchExperienceSettings`, so the web backend's already-built rerank stage activates when a reranker server is provided.

**Architecture:** Mirror the existing optional-URL config `fetch_url` exactly — a nullable `ServiceSettings` field loaded via `get_env_str(..., None)`, then copied into `SearchExperienceSettings.from_app_settings`. All downstream rerank plumbing already exists; this only supplies the value.

**Tech Stack:** Python, dataclass config, pytest.

## Global Constraints

- Default behavior unchanged: no env var ⇒ `rerank_url=None` ⇒ rerank stage skipped, exactly as today.
- Opt-in only; no frontend change, no dev toggle.
- Do not modify the reranker server, `RerankHTTPRankingStage`, or the existing `rerank_url` threading.
- Never commit to `main`; work on branch `feat/rerank-url-wiring` (already created).

---

### Task 1: Wire rerank_url through config → web settings, with docs

**Files:**
- Modify: `src/internal/configs/app_configs.py` (`ServiceSettings` dataclass line 31 area; loader line 166 area)
- Modify: `src/internal/servers/web/app.py` (`SearchExperienceSettings.from_app_settings` return block, lines 159-165)
- Modify: `README.md`, `.claude/CLAUDE.md` (retrieval server run docs)
- Test: `tests/unit/test_configs.py`

**Interfaces:**
- Consumes: `get_env_str(source, name, default)` (existing), `load_app_settings(source)` (existing), `SearchExperienceSettings.from_app_settings(app_settings)` (existing).
- Produces: `ServiceSettings.rerank_url: str | None` populated from `AGENTIC_SEARCH_RERANK_URL`; `SearchExperienceSettings.rerank_url` copied from it.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_configs.py`, add a config-loader test and extend the web-settings test.

Add a new test (near `test_load_app_settings_reads_typed_environment`):

```python
def test_load_app_settings_reads_rerank_url():
    assert (
        load_app_settings(
            {"AGENTIC_SEARCH_RERANK_URL": "http://localhost:8002/rerank"}
        ).services.rerank_url
        == "http://localhost:8002/rerank"
    )
    # Unset ⇒ None (rerank stays off by default).
    assert load_app_settings({}).services.rerank_url is None
```

Extend `test_web_settings_can_be_built_from_app_settings` (add the env key and assertion):

```python
def test_web_settings_can_be_built_from_app_settings():
    app_settings = load_app_settings(
        {
            "AGENTIC_SEARCH_RETRIEVAL_URL": "http://search.test/retrieve",
            "AGENTIC_SEARCH_WEB_TOP_K": "9",
            "AGENTIC_SEARCH_WEB_DB_PATH": "/tmp/search.sqlite3",
            "AGENTIC_SEARCH_RERANK_URL": "http://rr.test/rerank",
        }
    )

    web_settings = SearchExperienceSettings.from_app_settings(app_settings)

    assert web_settings.search_url == "http://search.test/retrieve"
    assert web_settings.top_k == 9
    assert web_settings.db_path == "/tmp/search.sqlite3"
    assert web_settings.rerank_url == "http://rr.test/rerank"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_configs.py::test_load_app_settings_reads_rerank_url tests/unit/test_configs.py::test_web_settings_can_be_built_from_app_settings -v`
Expected: FAIL — `test_load_app_settings_reads_rerank_url` errors with `AttributeError: 'ServiceSettings' object has no attribute 'rerank_url'`; the web-settings test fails on the new `rerank_url` assertion.

- [ ] **Step 3: Add the `ServiceSettings` field**

In `src/internal/configs/app_configs.py`, in the `ServiceSettings` dataclass, add after `fetch_url: str | None = None` (line 31):

```python
    rerank_url: str | None = None
```

- [ ] **Step 4: Load it from the env in the settings loader**

In the same file, in the `ServiceSettings(...)` construction inside the loader, add after the `fetch_url=get_env_str(...)` line (line 166):

```python
            rerank_url=get_env_str(source, "AGENTIC_SEARCH_RERANK_URL", None),
```

- [ ] **Step 5: Copy it into the web settings**

In `src/internal/servers/web/app.py`, in `SearchExperienceSettings.from_app_settings`'s returned `cls(...)` (after `search_url=app_settings.services.retrieval_url,`, line 160):

```python
            rerank_url=app_settings.services.rerank_url,
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_configs.py -v`
Expected: PASS (all config tests, including the two new/extended assertions).

- [ ] **Step 7: Update the docs**

In `README.md`, in the "Running the 3-process local stack" retrieval section (after the hybrid server command block), add:

```markdown
# Optional — cross-encoder reranker (Terminal 1b). Then set the env on the web
# backend and restart it so retrieved docs are reranked before display:
python3 -m src.internal.servers.retrieval.rerank --port 8002
# web backend env: AGENTIC_SEARCH_RERANK_URL=http://localhost:8002/rerank
```

In `.claude/CLAUDE.md`, in the retrieval-servers run section (the same 3-process block), add the same optional reranker note so the two stay consistent.

- [ ] **Step 8: Run the broader config + web settings suites for regressions**

Run: `python -m pytest tests/unit/test_configs.py -q`
Expected: PASS. (This change only adds a nullable field with a `None` default, so no existing construction of `ServiceSettings` breaks.)

- [ ] **Step 9: Commit**

```bash
git add src/internal/configs/app_configs.py src/internal/servers/web/app.py tests/unit/test_configs.py README.md .claude/CLAUDE.md
git commit -m "feat: wire AGENTIC_SEARCH_RERANK_URL so web reranking is reachable"
```

---

## Final verification

- [ ] `python -m pytest tests/unit/test_configs.py -q` — green.
- [ ] `python -c "from src.internal.configs import load_app_settings; print(load_app_settings({'AGENTIC_SEARCH_RERANK_URL':'http://x/rerank'}).services.rerank_url); print(load_app_settings({}).services.rerank_url)"` — prints the URL, then `None`.
- [ ] `ruff check src/internal/configs/app_configs.py src/internal/servers/web/app.py` — clean.
- [ ] Manual (optional): run `rerank.py --port 8002`, start the web backend with `AGENTIC_SEARCH_RERANK_URL=http://localhost:8002/rerank`, query at `:5173`, confirm source cards are reordered with cross-encoder scores; unset the env ⇒ identical to today.
