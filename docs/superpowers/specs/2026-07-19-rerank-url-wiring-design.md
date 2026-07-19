# Wire rerank_url so reranking is reachable from the web app

**Date:** 2026-07-19
**Status:** Approved

## Problem

Reranking is fully implemented but unreachable from the web UI (`:5173`). The
reranker server (`src/internal/servers/retrieval/rerank.py`), the ranking stage
(`RerankHTTPRankingStage`), and the `rerank_url` parameter threaded through every
web retrieval route all exist. The web backend reranks documents whenever
`settings.rerank_url` is non-`None` (`_rank_documents`, `app.py`).

But nothing ever sets it:
- `SearchExperienceSettings.from_app_settings()` populates `search_url`, `top_k`,
  `db_path`, etc. — but **not** `rerank_url`, so it defaults to `None`.
- There is no `rerank_url` field in `ServiceSettings` and no
  `AGENTIC_SEARCH_RERANK_URL` env var.

So on the standard demo stack the rerank stage is always skipped, and source
cards show only base retrieval scores/order.

## Goal

Add the missing config wiring so reranking activates when (and only when) a
reranker server is running and its URL is provided — opt-in, with default
behavior unchanged.

## Design

Mirror the existing optional-URL config `fetch_url` exactly.

**1. `ServiceSettings`** (`src/internal/configs/app_configs.py`)
Add a nullable field after `fetch_url`:
```python
rerank_url: str | None = None
```
In the `AppSettings` loader, next to the `fetch_url` line:
```python
rerank_url=get_env_str(source, "AGENTIC_SEARCH_RERANK_URL", None),
```

**2. `SearchExperienceSettings.from_app_settings`** (`src/internal/servers/web/app.py`)
Add one line to the returned `cls(...)`:
```python
rerank_url=app_settings.services.rerank_url,
```
`SearchExperienceSettings` already declares `rerank_url: str | None = None` and
threads it through every route, so no other web code changes.

**3. Docs** (`README.md`, `.claude/CLAUDE.md`)
Document the optional reranker server as a 4th process and the env var:
```bash
# Optional — cross-encoder reranker; set the env on the web backend + restart:
python3 -m src.internal.servers.retrieval.rerank --port 8002
AGENTIC_SEARCH_RERANK_URL=http://localhost:8002/rerank
```

## Scope / non-goals

- No frontend change: reranked results surface as reordered source cards with
  cross-encoder scores in the existing SourceGrid; no new "reranked" badge.
- No dev toggle / query-time rerank control.
- No change to the reranker server, `RerankHTTPRankingStage`, or the existing
  `rerank_url` threading — those already work.
- Default behavior unchanged: no env var ⇒ `rerank_url=None` ⇒ rerank stage
  skipped, exactly as today.

## Verification

- Config: with `AGENTIC_SEARCH_RERANK_URL` set, `load_app_settings().services.rerank_url`
  equals it; unset ⇒ `None`.
- Web settings: `SearchExperienceSettings.from_app_settings(app_settings)` copies
  `rerank_url` from the app settings (non-`None` when provided, `None` otherwise).
- Existing web/config tests stay green.
- Manual (optional): run `rerank.py`, set the env, restart the web backend, and
  confirm source cards at `:5173` come back reordered with cross-encoder scores;
  with the env unset, behavior is identical to today.
