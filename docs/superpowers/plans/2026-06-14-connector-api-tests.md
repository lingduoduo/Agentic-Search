# Connector API HTTP Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HTTP-level unit tests for all 7 `/admin/connectors/*` endpoints so the connector CRUD pipeline has the same coverage level as `/admin/tools/*` (added in PR #250).

**Architecture:** `create_connectors_router` in `src/internal/servers/connectors/api.py` wires 7 routes behind `make_require_admin`. The `ingest` endpoint is the exception — it requires no auth. Tests use `TestClient` + `SearchExperienceSettings(db_path=tmp_path/…)` exactly like `test_tool_admin_api.py`. No mocking of the DB — SQLite is fast and in-process.

**Tech Stack:** Python 3.12, FastAPI `TestClient`, pytest, `AgenticSearchStore`, `generate_user_jwt_token`.

---

## File Map

| File | Change |
|------|--------|
| `tests/unit/servers/web/test_connector_api.py` | Create — all HTTP tests |

---

## Task 1: Auth guard and list

**Files:**
- Create: `tests/unit/servers/web/test_connector_api.py`

- [ ] **Step 1: Write the module skeleton and auth tests**

```python
"""HTTP-level tests for /admin/connectors/* endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

_ADMIN = "admin"


def _admin_headers() -> dict[str, str]:
    token = generate_user_jwt_token(user_id=_ADMIN)
    return {"Authorization": f"Bearer {token}"}


def _settings() -> AppSettings:
    return AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))


def _make_app(tmp_path):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=_settings(),
    )


def test_list_connectors_requires_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert client.get("/admin/connectors").status_code == 401


def test_list_connectors_empty(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/admin/connectors", headers=_admin_headers())
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run to verify they pass**

```bash
pytest tests/unit/servers/web/test_connector_api.py -v 2>&1 | tail -15
```

Expected: 2 tests PASS.

---

## Task 2: Create and 409 duplicate

- [ ] **Step 1: Add create and duplicate tests**

```python
def test_create_connector_returns_201(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/connectors",
        json={"name": "My Web", "source": "web", "config": {"urls": ["https://example.test"]}},
        headers=_admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"].startswith("connector_")
    assert data["name"] == "My Web"
    assert data["source"] == "web"
    assert data["enabled"] is True


def test_create_connector_duplicate_name_returns_409(tmp_path):
    client = TestClient(_make_app(tmp_path))
    payload = {"name": "Dup", "source": "file"}
    client.post("/admin/connectors", json=payload, headers=_admin_headers())
    resp = client.post("/admin/connectors", json=payload, headers=_admin_headers())
    assert resp.status_code == 409


def test_list_connectors_returns_created(tmp_path):
    client = TestClient(_make_app(tmp_path))
    client.post(
        "/admin/connectors",
        json={"name": "Slack", "source": "slack"},
        headers=_admin_headers(),
    )
    resp = client.get("/admin/connectors", headers=_admin_headers())
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()]
    assert "Slack" in names
```

- [ ] **Step 2: Run**

```bash
pytest tests/unit/servers/web/test_connector_api.py -v 2>&1 | tail -15
```

Expected: 5 tests PASS.

---

## Task 3: Get, update, delete

- [ ] **Step 1: Add get/update/delete tests**

```python
def test_get_connector_returns_detail(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "RSS", "source": "rss", "config": {"feed_url": "https://feed.test"}},
        headers=_admin_headers(),
    ).json()
    resp = client.get(f"/admin/connectors/{created['id']}", headers=_admin_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == created["id"]
    assert data["document_count"] == 0
    assert isinstance(data["attempts"], list)


def test_get_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.get("/admin/connectors/nonexistent", headers=_admin_headers()).status_code
        == 404
    )


def test_update_connector_name_and_enabled(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "Old Name", "source": "file"},
        headers=_admin_headers(),
    ).json()
    resp = client.patch(
        f"/admin/connectors/{created['id']}",
        json={"name": "New Name", "enabled": False},
        headers=_admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "New Name"
    assert data["enabled"] is False


def test_delete_connector_returns_204(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "ToDelete", "source": "web"},
        headers=_admin_headers(),
    ).json()
    assert (
        client.delete(
            f"/admin/connectors/{created['id']}", headers=_admin_headers()
        ).status_code
        == 204
    )
    # Gone
    assert (
        client.get(
            f"/admin/connectors/{created['id']}", headers=_admin_headers()
        ).status_code
        == 404
    )


def test_delete_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.delete("/admin/connectors/ghost", headers=_admin_headers()).status_code
        == 404
    )
```

- [ ] **Step 2: Run**

```bash
pytest tests/unit/servers/web/test_connector_api.py -v 2>&1 | tail -15
```

Expected: 10 tests PASS.

---

## Task 4: Run (enqueue attempt) and ingest (no auth)

- [ ] **Step 1: Add run and ingest tests**

```python
def test_run_connector_enqueues_attempt(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "SyncMe", "source": "web"},
        headers=_admin_headers(),
    ).json()
    resp = client.post(
        f"/admin/connectors/{created['id']}/run", headers=_admin_headers()
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["connector_id"] == created["id"]
    assert data["attempt_id"]


def test_run_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.post(
            "/admin/connectors/ghost/run", headers=_admin_headers()
        ).status_code
        == 404
    )


def test_ingest_no_connector_id_creates_shared_connector(tmp_path):
    """ingest endpoint requires no auth; creates _ingestion_api connector on first call."""
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/connectors/ingest",
        json={
            "documents": [
                {"title": "Doc A", "contents": "Content of doc A.", "url": "https://a.test"}
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 1
    assert data["connector_id"]


def test_ingest_with_explicit_connector_id(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "PushTarget", "source": "ingestion_api"},
        headers=_admin_headers(),
    ).json()
    resp = client.post(
        "/admin/connectors/ingest",
        json={
            "connector_id": created["id"],
            "documents": [
                {"title": "Doc B", "contents": "Content B."},
                {"title": "Doc C", "contents": "Content C."},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 2
    assert resp.json()["connector_id"] == created["id"]


def test_ingest_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/connectors/ingest",
        json={"connector_id": "ghost", "documents": [{"title": "X", "contents": "x"}]},
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/unit/servers/web/test_connector_api.py -v 2>&1 | tail -20
```

Expected: 15 tests PASS.

---

## Task 5: Full suite and commit

- [ ] **Step 1: Run full suite**

```bash
pytest --tb=short -q 2>&1 | tail -5
```

Expected: all pass (1558+).

- [ ] **Step 2: Commit on a new feature branch**

```bash
git checkout main && git pull
git checkout -b feat/connector-api-tests
git add tests/unit/servers/web/test_connector_api.py
git commit -m "test(connectors): add HTTP tests for /admin/connectors/* endpoints"
```

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/connector-api-tests
gh pr create \
  --title "test(connectors): HTTP tests for all /admin/connectors/* endpoints" \
  --body "$(cat <<'EOF'
## Summary

15 TestClient tests covering every route in `src/internal/servers/connectors/api.py`:

- Auth guard (401 without token)
- `GET /admin/connectors` — empty list, populated list
- `POST /admin/connectors` — 201 created, 409 duplicate name
- `GET /admin/connectors/{id}` — detail view with attempts, 404 unknown
- `PATCH /admin/connectors/{id}` — update name and enabled flag
- `DELETE /admin/connectors/{id}` — 204, then 404 on re-fetch, 404 unknown
- `POST /admin/connectors/{id}/run` — 202 with attempt_id, 404 unknown
- `POST /admin/connectors/ingest` — no auth required, auto-creates shared connector, explicit connector_id, 404 unknown connector_id

## Test plan

```bash
pytest tests/unit/servers/web/test_connector_api.py -v
pytest --tb=short -q
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
