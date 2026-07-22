# Memory CLI + Backend Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the conversation-memory service over HTTP (`/api/memory/*`) and add a `cli/cmd/memory` Go binary that manages memory (add/list/search/consolidate/profile/curate) against those endpoints — one source of truth (the web app's `AgenticSearchStore`).

**Architecture:** A backend `create_memory_router(db, llm)` (mirrors `debug_router.py`) wraps `src/internal/memory/service.py`; a new Go binary with `flag.FlagSet` subcommands calls it via `cli/api.Client`. Two DRY refactors: extract the CLI token helpers into `cli/clientauth`, and move `_maybe_encoder` into the service.

**Tech Stack:** Python 3.12 (FastAPI, pytest, ruff), Go 1.26 (`flag`, `net/http`, `golang-jwt/jwt/v5`, `go test`).

## Global Constraints

- Work on branch `feat/memory-cli` (already checked out). Never commit to `main`.
- Spec: `docs/superpowers/specs/2026-07-22-memory-cli-design.md`.
- Backend endpoints resolve `user_id` from `user_from_headers(request.headers)`, falling back to `DEFAULT_MEMORY_USER_ID` ("default_user"). LLM-only endpoints (`profile/generate`, `curate`) return HTTP 503 `{"detail": "LLM not configured"}` when `llm is None`.
- The router closes over `db`/`llm` (constructor args, no `Depends` for `db`); do NOT import from `app.py` (circular) — import `user_from_headers` from `src.internal.auth`.
- CLI reuses `cli/config` + `cli/api`; every new `Client` method is added to the `ClientAPI` interface (there is a `var _ ClientAPI = (*Client)(nil)` compile assertion). `baseURL` already includes `/api`, so method paths are `/memory/...`.
- Endpoint ⇄ CLI JSON contract (must match exactly): `POST /memory/save {text}→{memory_id}`; `GET /memory/list →{memories:[{id,text,updated_at}]}`; `POST /memory/search {query,max_results}→{results:[{id,text,score}]}`; `POST /memory/consolidate {resolve_conflicts}→{report}`; `GET /memory/profile →{profile:[...]}`; `POST /memory/profile/generate →{profile:[...]}`; `POST /memory/curate {session_id?}→{status,trajectory_id,counts,memory_count}`.
- Python: `pytest`, `ruff check . --fix && ruff format .`. Go: run from `cli/` — `go build ./...`, `go vet ./...`, `go test ./...`.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Auth-secret gotcha: the Go token minter must read `AGENTIC_SEARCH_AUTH_SECRET` (the secret the backend verifies with), falling back to `AUTH_SECRET` for back-compat.

---

### Task 1: Move `_maybe_encoder` into the service (DRY #2)

**Files:**
- Modify: `src/internal/memory/service.py` (add `maybe_build_encoder`, `os`+`logging` imports, module logger)
- Modify: `src/internal/mcp_server/tools/memory.py` (delete local `_maybe_encoder`, use the service's)
- Test: `tests/unit/memory/test_encoder.py` (create)

**Interfaces:**
- Produces: `service.maybe_build_encoder() -> Encoder | None` — returns `None` unless `AGENTIC_SEARCH_MEMORY_SEMANTIC` is truthy; builds `build_e5_encoder(...)` guarded, `None` on any failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/memory/test_encoder.py
from src.internal.memory import service


def test_maybe_build_encoder_off_by_default(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", raising=False)
    assert service.maybe_build_encoder() is None


def test_maybe_build_encoder_none_when_disabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", "no")
    assert service.maybe_build_encoder() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/memory/test_encoder.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'maybe_build_encoder'`.

- [ ] **Step 3a: Add to service.py**

In `src/internal/memory/service.py`, add `import logging` and `import os` to the top imports, add a module logger after the imports (`logger = logging.getLogger(__name__)`), and add this function right after the `Encoder = ...` alias:

```python
def maybe_build_encoder() -> Encoder | None:
    """Build the e5 memory encoder when AGENTIC_SEARCH_MEMORY_SEMANTIC is set,
    else return None so callers use the lexical search fallback."""
    if os.getenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return None
    try:
        from src.internal.servers.retrieval.hybrid import build_e5_encoder

        return build_e5_encoder(
            device=os.getenv("AGENTIC_SEARCH_MEMORY_EMBED_DEVICE", "cpu")
        )
    except Exception as exc:  # noqa: BLE001 — fall back to lexical
        logger.warning("Memory e5 encoder unavailable, using lexical search: %s", exc)
        return None
```

- [ ] **Step 3b: Repoint the MCP tool**

In `src/internal/mcp_server/tools/memory.py`: delete the local `_maybe_encoder()` function, add `from src.internal.memory.service import maybe_build_encoder` to the imports (next to the existing `from src.internal.memory.service import DEFAULT_MEMORY_USER_ID`), and change the call site in `search_memories` from `encoder=_maybe_encoder()` to `encoder=maybe_build_encoder()`. Leave the module `logger`/`os` imports (still used by `_build_llm`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/memory/test_encoder.py tests/unit/test_mcp_memory_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/memory/service.py src/internal/mcp_server/tools/memory.py tests/unit/memory/test_encoder.py
git commit -m "refactor: move memory encoder builder into the service (shared by MCP + router)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Backend `create_memory_router(db, llm)`

**Files:**
- Create: `src/internal/memory/router.py`
- Modify: `src/internal/servers/web/app.py` (register the router)
- Test: `tests/unit/servers/web/test_memory_router.py` (create)

**Interfaces:**
- Consumes: `service.{save_memory,search_memories,consolidate_memories,generate_user_profile,get_user_profile,curate_from_conversation,maybe_build_encoder,DEFAULT_MEMORY_USER_ID}`; `db.get_user_memory_records`; `user_from_headers`.
- Produces: `create_memory_router(db, llm=None, *, default_user_id: str = DEFAULT_MEMORY_USER_ID) -> APIRouter`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/servers/web/test_memory_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db.store import AgenticSearchStore
from src.internal.memory.router import create_memory_router


class _FakeLLM:
    def __init__(self, text="[]"):
        self._text = text

    def complete(self, prompt, **kwargs):
        return self._text


def _client(db, llm=None):
    app = FastAPI()
    app.include_router(create_memory_router(db, llm))
    return TestClient(app)


def test_save_list_search_consolidate_profile_without_llm():
    db = AgenticSearchStore(":memory:")
    c = _client(db)

    r = c.post("/api/memory/save", json={"text": "User enjoys hiking in the mountains"})
    assert r.status_code == 200 and r.json()["memory_id"]

    listed = c.get("/api/memory/list").json()["memories"]
    assert listed[0]["text"] == "User enjoys hiking in the mountains"

    hits = c.post("/api/memory/search", json={"query": "mountain hiking", "max_results": 5}).json()
    assert hits["results"] and "hiking" in hits["results"][0]["text"]

    report = c.post("/api/memory/consolidate", json={"resolve_conflicts": True}).json()["report"]
    assert report["initial"] == 1 and report["final"] == 1

    assert c.get("/api/memory/profile").json()["profile"] == []
    db.close()


def test_llm_endpoints_503_without_llm():
    db = AgenticSearchStore(":memory:")
    c = _client(db, llm=None)
    assert c.post("/api/memory/profile/generate").status_code == 503
    assert c.post("/api/memory/curate", json={}).status_code == 503
    db.close()


def test_generate_profile_with_fake_llm():
    db = AgenticSearchStore(":memory:")
    db.add_user_memory("default_user", "User is a software engineer at TechCorp")
    llm = _FakeLLM('[{"topic":"work","subtopic":"role","content":"Software engineer at TechCorp"}]')
    c = _client(db, llm)
    profile = c.post("/api/memory/profile/generate").json()["profile"]
    assert profile[0]["topic"] == "work"
    assert c.get("/api/memory/profile").json()["profile"][0]["content"] == "Software engineer at TechCorp"
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/servers/web/test_memory_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.memory.router'`.

- [ ] **Step 3a: Create the router**

```python
# src/internal/memory/router.py
"""HTTP router exposing the conversation-memory service under /api/memory."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.internal.auth import user_from_headers
from src.internal.memory import service
from src.internal.memory.service import DEFAULT_MEMORY_USER_ID, maybe_build_encoder


class _SaveRequest(BaseModel):
    text: str = Field(..., min_length=1)


class _SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=100)


class _ConsolidateRequest(BaseModel):
    resolve_conflicts: bool = True


class _CurateRequest(BaseModel):
    session_id: str | None = None


def create_memory_router(
    db, llm=None, *, default_user_id: str = DEFAULT_MEMORY_USER_ID
) -> APIRouter:
    router = APIRouter(prefix="/api/memory", tags=["memory"])

    def _uid(request: Request) -> str:
        user = user_from_headers(request.headers)
        return user.id if user is not None else default_user_id

    @router.post("/save")
    def save(body: _SaveRequest, request: Request) -> dict:
        return {"memory_id": service.save_memory(db, _uid(request), body.text)}

    @router.get("/list")
    def list_memories(request: Request) -> dict:
        records = db.get_user_memory_records(_uid(request))
        return {
            "memories": [
                {"id": r.id, "text": r.memory_text, "updated_at": r.updated_at}
                for r in records
            ]
        }

    @router.post("/search")
    def search(body: _SearchRequest, request: Request) -> dict:
        hits = service.search_memories(
            db,
            _uid(request),
            body.query,
            max_results=body.max_results,
            encoder=maybe_build_encoder(),
        )
        return {
            "results": [{"id": r.id, "text": r.memory_text, "score": s} for r, s in hits]
        }

    @router.post("/consolidate")
    def consolidate(body: _ConsolidateRequest, request: Request) -> dict:
        return {
            "report": service.consolidate_memories(
                db, _uid(request), resolve_conflicts=body.resolve_conflicts
            )
        }

    @router.get("/profile")
    def get_profile(request: Request) -> dict:
        return {"profile": [asdict(e) for e in service.get_user_profile(db, _uid(request))]}

    @router.post("/profile/generate")
    def generate_profile(request: Request) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not configured")
        entries = service.generate_user_profile(db, _uid(request), llm)
        return {"profile": [asdict(e) for e in entries]}

    @router.post("/curate")
    async def curate(body: _CurateRequest, request: Request) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not configured")
        return await service.curate_from_conversation(
            db, _uid(request), llm, session_id=body.session_id
        )

    return router
```

- [ ] **Step 3b: Register it in the app**

In `src/internal/servers/web/app.py`, in the router-registration block (right after the `create_feedback_router(db)` block, ~line 404), add:

```python
    # --- Memory ---
    from src.internal.memory.router import create_memory_router

    app.include_router(create_memory_router(db, llm))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_memory_router.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/internal/memory/router.py src/internal/servers/web/app.py tests/unit/servers/web/test_memory_router.py
git commit -m "feat: /api/memory HTTP endpoints over the memory service

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Extract `cli/clientauth` (DRY #1)

**Files:**
- Create: `cli/clientauth/clientauth.go`
- Create: `cli/clientauth/clientauth_test.go`
- Modify: `cli/cmd/query/main.go` (delete local `resolveToken`/`resolveSecret`/`mintJWT`, call `clientauth.*`)

**Interfaces:**
- Produces (package `clientauth`, module path `github.com/lingduoduo/Agentic-Search/cli/clientauth`):
  - `func ResolveToken(flagToken, configToken, userID, email, secret string) (string, error)`
  - `func ResolveSecret(s string) string` (precedence: arg → `AGENTIC_SEARCH_AUTH_SECRET` → `AUTH_SECRET`)
  - `func MintJWT(userID, email, secret string) (string, error)` (HS256, `sub`+`iat`[+`email`])

- [ ] **Step 1: Write the failing test**

```go
// cli/clientauth/clientauth_test.go
package clientauth_test

import (
	"testing"

	jwtlib "github.com/golang-jwt/jwt/v5"
	"github.com/lingduoduo/Agentic-Search/cli/clientauth"
)

func TestMintJWTHasSubClaim(t *testing.T) {
	tok, err := clientauth.MintJWT("alice", "a@example.com", "s3cret")
	if err != nil {
		t.Fatalf("MintJWT: %v", err)
	}
	parsed, err := jwtlib.Parse(tok, func(*jwtlib.Token) (any, error) { return []byte("s3cret"), nil })
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	claims := parsed.Claims.(jwtlib.MapClaims)
	if claims["sub"] != "alice" {
		t.Errorf("sub = %v, want alice", claims["sub"])
	}
	if claims["email"] != "a@example.com" {
		t.Errorf("email = %v, want a@example.com", claims["email"])
	}
}

func TestResolveSecretPrefersAgenticEnv(t *testing.T) {
	t.Setenv("AGENTIC_SEARCH_AUTH_SECRET", "agentic")
	t.Setenv("AUTH_SECRET", "legacy")
	if got := clientauth.ResolveSecret(""); got != "agentic" {
		t.Errorf("ResolveSecret = %q, want agentic", got)
	}
	if got := clientauth.ResolveSecret("explicit"); got != "explicit" {
		t.Errorf("ResolveSecret(explicit) = %q, want explicit", got)
	}
}

func TestResolveTokenPrefersFlag(t *testing.T) {
	got, err := clientauth.ResolveToken("flagtok", "cfgtok", "", "", "")
	if err != nil || got != "flagtok" {
		t.Fatalf("ResolveToken = %q, %v; want flagtok", got, err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `cli/`): `go test ./clientauth/`
Expected: FAIL — `no required module provides package .../cli/clientauth` (package doesn't exist yet).

- [ ] **Step 3a: Create the package**

```go
// cli/clientauth/clientauth.go

// Package clientauth resolves and mints bearer tokens for the CLI binaries.
package clientauth

import (
	"fmt"
	"os"
	"time"

	jwtlib "github.com/golang-jwt/jwt/v5"
)

// ResolveToken picks a bearer token: an explicit flag, then a config token,
// then a freshly minted JWT for userID. Returns an error if none is available.
func ResolveToken(flagToken, configToken, userID, email, secret string) (string, error) {
	if flagToken != "" {
		return flagToken, nil
	}
	if configToken != "" {
		return configToken, nil
	}
	if userID != "" {
		return MintJWT(userID, email, ResolveSecret(secret))
	}
	return "", fmt.Errorf("provide -token, set AGENTIC_SEARCH_PAT, or pass -user-id to authenticate")
}

// ResolveSecret returns the JWT signing secret: an explicit value, else
// AGENTIC_SEARCH_AUTH_SECRET (what the backend verifies with), else AUTH_SECRET.
func ResolveSecret(s string) string {
	if s != "" {
		return s
	}
	if v := os.Getenv("AGENTIC_SEARCH_AUTH_SECRET"); v != "" {
		return v
	}
	return os.Getenv("AUTH_SECRET")
}

// MintJWT mints an HS256 token with sub/iat (and email when present).
func MintJWT(userID, email, secret string) (string, error) {
	claims := jwtlib.MapClaims{
		"sub": userID,
		"iat": time.Now().Unix(),
	}
	if email != "" {
		claims["email"] = email
	}
	tok := jwtlib.NewWithClaims(jwtlib.SigningMethodHS256, claims)
	return tok.SignedString([]byte(secret))
}
```

- [ ] **Step 3b: Repoint the query binary**

In `cli/cmd/query/main.go`: delete the local `resolveToken`, `resolveSecret`, and `mintJWT` functions; add `"github.com/lingduoduo/Agentic-Search/cli/clientauth"` to the import block and remove the now-unused `jwtlib`/`time` imports if they are no longer referenced elsewhere in the file; change the call site `token, err := resolveToken(*tokenFlag, cfg.APIKey, *userIDFlag, *emailFlag, *secretFlag)` to `token, err := clientauth.ResolveToken(*tokenFlag, cfg.APIKey, *userIDFlag, *emailFlag, *secretFlag)`.

- [ ] **Step 4: Run tests + build to verify**

Run (from `cli/`):
```bash
go test ./clientauth/
go build ./...
go vet ./cmd/query/
```
Expected: clientauth tests PASS; build succeeds (query binary compiles against clientauth).

- [ ] **Step 5: Commit**

```bash
git add cli/clientauth/ cli/cmd/query/main.go cli/go.mod cli/go.sum
git commit -m "refactor: extract CLI token helpers into cli/clientauth (shared by binaries)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CLI memory models + client methods

**Files:**
- Modify: `cli/models/models.go` (append memory structs)
- Modify: `cli/api/client.go` (append methods + extend `ClientAPI`)
- Test: `cli/api/memory_test.go` (create)

**Interfaces:**
- Produces `Client` methods (all via `doJSON`, paths relative to the `/api` baseURL): `SaveMemory`, `ListMemories`, `SearchMemories`, `ConsolidateMemories`, `GetMemoryProfile`, `GenerateMemoryProfile`, `CurateMemory` — each added to the `ClientAPI` interface.

- [ ] **Step 1: Write the failing test**

```go
// cli/api/memory_test.go
package api_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lingduoduo/Agentic-Search/cli/models"
	"github.com/lingduoduo/Agentic-Search/cli/testutil"
)

func TestSaveMemory_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || !strings.HasSuffix(r.URL.Path, "/memory/save") {
			t.Errorf("got %s %s, want POST /api/memory/save", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memory_id": "mem_123"}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).SaveMemory(t.Context(), "likes tea")
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
	if resp.MemoryID == nil || *resp.MemoryID != "mem_123" {
		t.Errorf("memory_id = %v, want mem_123", resp.MemoryID)
	}
}

func TestListMemories_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || !strings.HasSuffix(r.URL.Path, "/memory/list") {
			t.Errorf("got %s %s, want GET /api/memory/list", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories": [{"id": "m1", "text": "hi", "updated_at": null}]}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).ListMemories(t.Context())
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(resp.Memories) != 1 || resp.Memories[0].ID != "m1" {
		t.Errorf("memories = %+v", resp.Memories)
	}
}

func TestCurateMemory_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/memory/curate") {
			t.Errorf("path = %s, want /api/memory/curate", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","trajectory_id":"t1","counts":{"add":1},"memory_count":1}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).CurateMemory(t.Context(), nil)
	if err != nil {
		t.Fatalf("CurateMemory: %v", err)
	}
	if resp.Status != "ok" || resp.Counts["add"] != 1 {
		t.Errorf("resp = %+v", resp)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `cli/`): `go test ./api/ -run TestSaveMemory`
Expected: FAIL — `client.SaveMemory undefined` (compile error).

- [ ] **Step 3a: Add models**

Append to `cli/models/models.go`:

```go
// --- Memory ---

// MemorySaveRequest is the body for POST /api/memory/save.
type MemorySaveRequest struct {
	Text string `json:"text"`
}

// MemorySaveResponse is the response from POST /api/memory/save.
type MemorySaveResponse struct {
	MemoryID *string `json:"memory_id"`
}

// MemoryRecord is one stored memory.
type MemoryRecord struct {
	ID        string  `json:"id"`
	Text      string  `json:"text"`
	UpdatedAt *string `json:"updated_at"`
}

// MemoryListResponse is the response from GET /api/memory/list.
type MemoryListResponse struct {
	Memories []MemoryRecord `json:"memories"`
}

// MemorySearchRequest is the body for POST /api/memory/search.
type MemorySearchRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// MemorySearchHit is one search result.
type MemorySearchHit struct {
	ID    string  `json:"id"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

// MemorySearchResponse is the response from POST /api/memory/search.
type MemorySearchResponse struct {
	Results []MemorySearchHit `json:"results"`
}

// MemoryConsolidateRequest is the body for POST /api/memory/consolidate.
type MemoryConsolidateRequest struct {
	ResolveConflicts bool `json:"resolve_conflicts"`
}

// MemoryConflict records one resolved attribute conflict.
type MemoryConflict struct {
	Attribute  string   `json:"attribute"`
	Kept       string   `json:"kept"`
	Superseded []string `json:"superseded"`
}

// MemoryConsolidateReport is the consolidate report.
type MemoryConsolidateReport struct {
	Initial           int              `json:"initial"`
	DuplicatesRemoved int              `json:"duplicates_removed"`
	ConflictsResolved []MemoryConflict `json:"conflicts_resolved"`
	Final             int              `json:"final"`
}

// MemoryConsolidateResponse is the response from POST /api/memory/consolidate.
type MemoryConsolidateResponse struct {
	Report MemoryConsolidateReport `json:"report"`
}

// MemoryProfileEntry is one profile entry (extra JSON fields are ignored).
type MemoryProfileEntry struct {
	Topic    string `json:"topic"`
	Subtopic string `json:"subtopic"`
	Content  string `json:"content"`
}

// MemoryProfileResponse is the response from the profile endpoints.
type MemoryProfileResponse struct {
	Profile []MemoryProfileEntry `json:"profile"`
}

// MemoryCurateRequest is the body for POST /api/memory/curate.
type MemoryCurateRequest struct {
	SessionID *string `json:"session_id,omitempty"`
}

// MemoryCurateResponse is the response from POST /api/memory/curate.
type MemoryCurateResponse struct {
	Status       string         `json:"status"`
	TrajectoryID string         `json:"trajectory_id"`
	Counts       map[string]int `json:"counts"`
	MemoryCount  int            `json:"memory_count"`
}
```

- [ ] **Step 3b: Add client methods + extend the interface**

Append the methods to `cli/api/client.go` (before the `ClientAPI` interface block), then add each to the `ClientAPI` interface:

```go
// SaveMemory calls POST /api/memory/save.
func (c *Client) SaveMemory(ctx context.Context, text string) (*models.MemorySaveResponse, error) {
	var resp models.MemorySaveResponse
	if err := c.doJSON(ctx, "POST", "/memory/save", models.MemorySaveRequest{Text: text}, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListMemories calls GET /api/memory/list.
func (c *Client) ListMemories(ctx context.Context) (*models.MemoryListResponse, error) {
	var resp models.MemoryListResponse
	if err := c.doJSON(ctx, "GET", "/memory/list", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// SearchMemories calls POST /api/memory/search.
func (c *Client) SearchMemories(ctx context.Context, query string, maxResults int) (*models.MemorySearchResponse, error) {
	var resp models.MemorySearchResponse
	req := models.MemorySearchRequest{Query: query, MaxResults: maxResults}
	if err := c.doJSON(ctx, "POST", "/memory/search", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ConsolidateMemories calls POST /api/memory/consolidate.
func (c *Client) ConsolidateMemories(ctx context.Context, resolveConflicts bool) (*models.MemoryConsolidateResponse, error) {
	var resp models.MemoryConsolidateResponse
	req := models.MemoryConsolidateRequest{ResolveConflicts: resolveConflicts}
	if err := c.doJSON(ctx, "POST", "/memory/consolidate", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// GetMemoryProfile calls GET /api/memory/profile.
func (c *Client) GetMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error) {
	var resp models.MemoryProfileResponse
	if err := c.doJSON(ctx, "GET", "/memory/profile", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// GenerateMemoryProfile calls POST /api/memory/profile/generate.
func (c *Client) GenerateMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error) {
	var resp models.MemoryProfileResponse
	if err := c.doJSON(ctx, "POST", "/memory/profile/generate", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// CurateMemory calls POST /api/memory/curate.
func (c *Client) CurateMemory(ctx context.Context, sessionID *string) (*models.MemoryCurateResponse, error) {
	var resp models.MemoryCurateResponse
	req := models.MemoryCurateRequest{SessionID: sessionID}
	if err := c.doJSONWith(ctx, c.searchHTTPClient, "POST", "/memory/curate", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}
```

Add these lines to the `ClientAPI` interface (before the closing `}`):

```go
	SaveMemory(ctx context.Context, text string) (*models.MemorySaveResponse, error)
	ListMemories(ctx context.Context) (*models.MemoryListResponse, error)
	SearchMemories(ctx context.Context, query string, maxResults int) (*models.MemorySearchResponse, error)
	ConsolidateMemories(ctx context.Context, resolveConflicts bool) (*models.MemoryConsolidateResponse, error)
	GetMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error)
	GenerateMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error)
	CurateMemory(ctx context.Context, sessionID *string) (*models.MemoryCurateResponse, error)
```

- [ ] **Step 4: Run tests + build**

Run (from `cli/`):
```bash
go test ./api/ -run TestSaveMemory
go test ./api/ -run TestListMemories
go test ./api/ -run TestCurateMemory
go build ./...
```
Expected: PASS; build succeeds (the `var _ ClientAPI = (*Client)(nil)` assertion still holds).

- [ ] **Step 5: Commit**

```bash
git add cli/models/models.go cli/api/client.go cli/api/memory_test.go
git commit -m "feat: cli/api memory client methods + models

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `cli/cmd/memory` binary

**Files:**
- Create: `cli/cmd/memory/main.go`
- Test: `cli/cmd/memory/main_test.go`

**Interfaces:**
- Consumes: `config.Load`, `api.NewClient`, `clientauth.ResolveToken`, the Task-4 client methods.
- Produces: a `run([]string) int` function (so tests can drive it) and `main()` calling `os.Exit(run(os.Args[1:]))`. Subcommands: `add`, `list`, `search`, `consolidate`, `profile`, `curate`.

- [ ] **Step 1: Write the failing test**

```go
// cli/cmd/memory/main_test.go
package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRunUnknownSubcommand(t *testing.T) {
	if code := run([]string{"bogus"}); code == 0 {
		t.Fatalf("unknown subcommand should exit non-zero")
	}
}

func TestRunNoArgsShowsUsage(t *testing.T) {
	if code := run(nil); code == 0 {
		t.Fatalf("no subcommand should exit non-zero")
	}
}

func TestRunAddHitsSaveEndpoint(t *testing.T) {
	var gotPath, gotMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotMethod = r.URL.Path, r.Method
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memory_id":"mem_1"}`))
	}))
	defer srv.Close()
	t.Setenv("AGENTIC_SEARCH_URL", srv.URL)

	if code := run([]string{"add", "likes", "tea"}); code != 0 {
		t.Fatalf("add exited %d", code)
	}
	if gotMethod != "POST" || !strings.HasSuffix(gotPath, "/memory/save") {
		t.Errorf("got %s %s, want POST .../memory/save", gotMethod, gotPath)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `cli/`): `go test ./cmd/memory/`
Expected: FAIL — `undefined: run` / package has no `main.go` yet.

- [ ] **Step 3: Implement the binary**

```go
// cli/cmd/memory/main.go

// Command memory manages a user's long-term memory via the Agentic Search
// backend /api/memory endpoints.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/lingduoduo/Agentic-Search/cli/api"
	"github.com/lingduoduo/Agentic-Search/cli/clientauth"
	"github.com/lingduoduo/Agentic-Search/cli/config"
	"github.com/lingduoduo/Agentic-Search/cli/models"
)

func main() { os.Exit(run(os.Args[1:])) }

func usage() {
	fmt.Fprintln(os.Stderr, `memory: manage user memory via the Agentic Search backend

Usage:
  memory <command> [flags]

Commands:
  add <text...>            Save a memory
  list                     List stored memories
  search <query...>        Search memories (--top-k N)
  consolidate              Deduplicate + resolve conflicts (--no-conflict)
  profile                  Show the user profile (--generate to rebuild via LLM)
  curate                   Reconcile memories from conversation (--session-id S)

Common flags: --url, --token, --user-id, --email, --secret`)
}

func run(args []string) int {
	if len(args) == 0 {
		usage()
		return 2
	}
	cmd, rest := args[0], args[1:]

	fs := flag.NewFlagSet(cmd, flag.ContinueOnError)
	urlFlag := fs.String("url", "", "Backend URL (overrides AGENTIC_SEARCH_URL)")
	tokenFlag := fs.String("token", "", "Bearer token / JWT (overrides AGENTIC_SEARCH_PAT)")
	userIDFlag := fs.String("user-id", "", "User ID — mint a JWT when no token is given")
	emailFlag := fs.String("email", "", "Email embedded in the minted JWT")
	secretFlag := fs.String("secret", "", "JWT signing secret (else AGENTIC_SEARCH_AUTH_SECRET / AUTH_SECRET)")
	topK := fs.Int("top-k", 5, "search: max results")
	noConflict := fs.Bool("no-conflict", false, "consolidate: dedup only")
	generate := fs.Bool("generate", false, "profile: rebuild via the LLM")
	sessionID := fs.String("session-id", "", "curate: restrict to one session")

	switch cmd {
	case "add", "list", "search", "consolidate", "profile", "curate":
	case "help", "-h", "--help":
		usage()
		return 0
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		usage()
		return 2
	}

	if err := fs.Parse(rest); err != nil {
		return 2
	}

	cfg := config.Load()
	if *urlFlag != "" {
		cfg.ServerURL = *urlFlag
	}
	// Token is optional: without one the backend uses the default user.
	if tok, err := clientauth.ResolveToken(*tokenFlag, cfg.APIKey, *userIDFlag, *emailFlag, *secretFlag); err == nil {
		cfg.APIKey = tok
	}
	client := api.NewClient(cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	if err := dispatch(ctx, client, cmd, fs, *topK, *noConflict, *generate, *sessionID); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	return 0
}

func dispatch(ctx context.Context, client *api.Client, cmd string, fs *flag.FlagSet, topK int, noConflict, generate bool, sessionID string) error {
	switch cmd {
	case "add":
		text := strings.TrimSpace(strings.Join(fs.Args(), " "))
		if text == "" {
			return fmt.Errorf("add requires memory text")
		}
		resp, err := client.SaveMemory(ctx, text)
		if err != nil {
			return err
		}
		id := "(empty, not saved)"
		if resp.MemoryID != nil {
			id = *resp.MemoryID
		}
		fmt.Printf("saved: %s\n", id)
	case "list":
		resp, err := client.ListMemories(ctx)
		if err != nil {
			return err
		}
		if len(resp.Memories) == 0 {
			fmt.Println("(no memories)")
		}
		for _, m := range resp.Memories {
			fmt.Printf("- (%s) %s\n", m.ID, m.Text)
		}
	case "search":
		query := strings.TrimSpace(strings.Join(fs.Args(), " "))
		if query == "" {
			return fmt.Errorf("search requires a query")
		}
		resp, err := client.SearchMemories(ctx, query, topK)
		if err != nil {
			return err
		}
		if len(resp.Results) == 0 {
			fmt.Printf("no memories matched %q\n", query)
		}
		for _, r := range resp.Results {
			fmt.Printf("- [%.3f] (%s) %s\n", r.Score, r.ID, r.Text)
		}
	case "consolidate":
		resp, err := client.ConsolidateMemories(ctx, !noConflict)
		if err != nil {
			return err
		}
		rep := resp.Report
		fmt.Printf("initial=%d duplicates_removed=%d conflicts_resolved=%d final=%d\n",
			rep.Initial, rep.DuplicatesRemoved, len(rep.ConflictsResolved), rep.Final)
		for _, c := range rep.ConflictsResolved {
			fmt.Printf("  conflict[%s]: kept %q, dropped %v\n", c.Attribute, c.Kept, c.Superseded)
		}
	case "profile":
		var resp *models.MemoryProfileResponse
		var err error
		if generate {
			resp, err = client.GenerateMemoryProfile(ctx)
		} else {
			resp, err = client.GetMemoryProfile(ctx)
		}
		if err != nil {
			return err
		}
		if len(resp.Profile) == 0 {
			fmt.Println("(empty profile)")
		}
		for _, e := range resp.Profile {
			fmt.Printf("- %s / %s: %s\n", e.Topic, e.Subtopic, e.Content)
		}
	case "curate":
		var sid *string
		if sessionID != "" {
			sid = &sessionID
		}
		resp, err := client.CurateMemory(ctx, sid)
		if err != nil {
			return err
		}
		fmt.Printf("status=%s trajectory=%s counts=%v memories=%d\n",
			resp.Status, resp.TrajectoryID, resp.Counts, resp.MemoryCount)
	}
	return nil
}
```

The `profile` branch above uses `*models.MemoryProfileResponse` directly (both `GetMemoryProfile` and `GenerateMemoryProfile` return that type), so the `cli/models` import in the block is required. No adapter types are needed.

- [ ] **Step 4: Run tests + build + vet**

Run (from `cli/`):
```bash
go test ./cmd/memory/
go build ./...
go vet ./...
```
Expected: the 3 dispatch tests PASS; build + vet clean.

- [ ] **Step 5: Commit**

```bash
git add cli/cmd/memory/
git commit -m "feat: cli/cmd/memory binary (add/list/search/consolidate/profile/curate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full verification

**Files:** none.

- [ ] **Step 1: Python suite + lint**

Run:
```bash
pytest tests/unit/memory/ tests/unit/servers/web/test_memory_router.py tests/unit/test_mcp_memory_tools.py -q
ruff check . --fix && ruff format .
```
Expected: all PASS; ruff clean. Re-run the tests if ruff changed files.

- [ ] **Step 2: Go build + vet + test**

Run (from `cli/`):
```bash
go build ./...
go vet ./...
go test ./...
```
Expected: all PASS.

- [ ] **Step 3: Broader Python sweep (catch router registration regressions)**

Run: `pytest tests/unit/servers/web/ -q`
Expected: PASS.

- [ ] **Step 4: Commit any lint fixups**

```bash
git add -A
git commit -m "chore: ruff for memory CLI backend endpoints

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(Skip if nothing changed.)

---

## Post-implementation

Push the branch and open a PR against `main`, titled e.g. *feat: memory CLI + /api/memory backend endpoints*, linking the spec and this plan. Note the auth-secret env alignment (`AGENTIC_SEARCH_AUTH_SECRET`) in the PR body.
