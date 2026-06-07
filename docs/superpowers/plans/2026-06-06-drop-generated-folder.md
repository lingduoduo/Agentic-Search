# Drop `generated/` Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the repo-root `generated/` directory by inlining its three callers to use `requests` directly, then deleting the folder and removing its `pythonpath` entry.

**Architecture:** `generated/` is a 58-line stub of an OpenAPI client that was never actually generated. It provides three stub classes (`Configuration`, `ApiClient`, `ApiException`) used in only three integration test files. `_cc_pair_creator` in `cc_pair.py` is already broken — it calls `api.DefaultApi` and `api.ConnectorCredentialPairMetadata` that don't exist in the stub. The fix is to replace the broken OpenAPI call with a direct `requests.put` matching the REST pattern used everywhere else in the test managers, replace `ApiException` with `requests.HTTPError`, then delete the directory.

**Tech Stack:** Python 3.11, pytest, requests, ruff

---

## Caller Map

| File | What it uses from `generated/` | Replacement |
|------|--------------------------------|-------------|
| `tests/integration/common_utils/config.py` | `Configuration(host=URL)` → `api_config` | Delete entire file (only caller of `api_config` is `cc_pair.py`) |
| `tests/integration/common_utils/managers/cc_pair.py` | `api.ApiClient`, `api.DefaultApi`, `api.ConnectorCredentialPairMetadata`, `api.StatusResponseInt` in `_cc_pair_creator` | Replace with `requests.put` |
| `tests/integration/tests/permissions/test_cc_pair_permissions.py` | `ApiException` (3 `pytest.raises` calls) | Replace with `requests.HTTPError` |

---

## Task 1: Replace generated-client usage with `requests` and delete `config.py`

**Files:**
- Modify: `tests/integration/common_utils/managers/cc_pair.py`
- Delete: `tests/integration/common_utils/config.py`
- Modify: `tests/integration/tests/permissions/test_cc_pair_permissions.py`

### Step 1a — Rewrite `_cc_pair_creator` in `cc_pair.py`

Read the file first: `tests/integration/common_utils/managers/cc_pair.py`

Find `_cc_pair_creator` (starts around line 21) and the two `generated`-related imports (lines 7–8):
```python
import generated.agentic_search_openapi_client.agentic_search_openapi_client as api  # type: ignore[unresolved-import]
...
from tests.integration.common_utils.config import api_config
```

**Remove** both import lines.

**Replace** the entire `_cc_pair_creator` function body:

Current broken body (uses non-existent stub types):
```python
def _cc_pair_creator(
    connector_id: int,
    credential_id: int,
    user_performing_action: DATestUser,
    name: str | None = None,
    access_type: AccessType = AccessType.PUBLIC,
    groups: list[int] | None = None,
) -> DATestCCPair:
    name = f"{name}-cc-pair" if name else f"test-cc-pair-{uuid4()}"

    with api.ApiClient(api_config) as api_client:
        api_instance = api.DefaultApi(api_client)
        connector_credential_pair_metadata = api.ConnectorCredentialPairMetadata(
            name=name, access_type=access_type, groups=groups or []
        )
        api_response: api.StatusResponseInt = (
            api_instance.associate_credential_to_connector(
                connector_id,
                credential_id,
                connector_credential_pair_metadata,
                _headers=user_performing_action.headers,
            )
        )

    return DATestCCPair(
        id=int(api_response.data),
        name=name,
        connector_id=connector_id,
        credential_id=credential_id,
        access_type=access_type,
        groups=groups or [],
    )
```

Replace with:
```python
def _cc_pair_creator(
    connector_id: int,
    credential_id: int,
    user_performing_action: DATestUser,
    name: str | None = None,
    access_type: AccessType = AccessType.PUBLIC,
    groups: list[int] | None = None,
) -> DATestCCPair:
    name = f"{name}-cc-pair" if name else f"test-cc-pair-{uuid4()}"
    response = requests.put(
        url=f"{API_SERVER_URL}/manage/admin/connector/{connector_id}/credential/{credential_id}",
        json={"name": name, "access_type": access_type, "groups": groups or []},
        headers=user_performing_action.headers,
    )
    response.raise_for_status()
    return DATestCCPair(
        id=int(response.json()["data"]),
        name=name,
        connector_id=connector_id,
        credential_id=credential_id,
        access_type=access_type,
        groups=groups or [],
    )
```

- [ ] **Step 1a: Edit `cc_pair.py`** — remove 2 imports, replace `_cc_pair_creator` body

### Step 1b — Delete `config.py`

After removing the `api_config` import from `cc_pair.py`, `config.py` has no callers. Delete it:

```bash
git rm tests/integration/common_utils/config.py
```

- [ ] **Step 1b: Delete `config.py`**

### Step 1c — Replace `ApiException` with `HTTPError` in test

In `tests/integration/tests/permissions/test_cc_pair_permissions.py`:

Replace the import:
```python
from generated.agentic_search_openapi_client.exceptions import ApiException  # ty: ignore[unresolved-import]
```
with:
```python
from requests import HTTPError
```

Replace all three `pytest.raises(ApiException)` with `pytest.raises(HTTPError)`:
- Line ~96: `with pytest.raises(ApiException):` → `with pytest.raises(HTTPError):`
- Line ~108: `with pytest.raises(ApiException):` → `with pytest.raises(HTTPError):`
- Line ~134: `with pytest.raises(ApiException):` → `with pytest.raises(HTTPError):`

- [ ] **Step 1c: Edit `test_cc_pair_permissions.py`** — replace import and 3 `pytest.raises` calls

### Step 1d — Verify no remaining `generated` references in Python files

```bash
grep -rn "from generated\|import generated" /Users/linghuang/Git/Agentic-Search --include="*.py" | grep -v "__pycache__"
```

Expected: **no output**.

- [ ] **Step 1d: Verify no remaining `generated` Python imports**

### Step 1e — Lint

```bash
cd /Users/linghuang/Git/Agentic-Search && ruff check tests/integration/ --fix && ruff format tests/integration/
```

Expected: no errors.

- [ ] **Step 1e: Lint**

### Step 1f — Run unit tests

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/ -x -q
```

Expected: all pass. (Integration tests are skipped — they require a live server.)

- [ ] **Step 1f: Run unit tests**

### Step 1g — Commit

```bash
cd /Users/linghuang/Git/Agentic-Search && git add \
  tests/integration/common_utils/managers/cc_pair.py \
  tests/integration/tests/permissions/test_cc_pair_permissions.py && \
git commit -m "refactor(tests): replace OpenAPI client stub with direct requests in cc_pair manager"
```

- [ ] **Step 1g: Commit**

---

## Task 2: Delete `generated/` directory and update `pyproject.toml`

**Files:**
- Delete: `generated/` directory (entire tree)
- Modify: `pyproject.toml` — remove `"generated"` from `pythonpath`

### Step 2a — Verify no remaining `generated/` imports anywhere

```bash
grep -rn "from generated\|import generated\|generated\." \
  /Users/linghuang/Git/Agentic-Search --include="*.py" --include="*.toml" --include="*.cfg" \
  | grep -v "__pycache__" | grep -v "docs/superpowers/plans/"
```

Expected: only the `pyproject.toml` line `pythonpath = ["src", "generated"]`. If any `.py` file appears, stop and report BLOCKED.

- [ ] **Step 2a: Confirm only pyproject.toml references `generated`**

### Step 2b — Delete the directory

```bash
git rm -r /Users/linghuang/Git/Agentic-Search/generated/
```

Expected output shows deletion of 6 files.

- [ ] **Step 2b: Delete `generated/` directory**

### Step 2c — Update `pyproject.toml`

In `pyproject.toml`, find line 35:
```toml
pythonpath = ["src", "generated"]
```
Replace with:
```toml
pythonpath = ["src"]
```

- [ ] **Step 2c: Remove `"generated"` from pythonpath in `pyproject.toml`**

### Step 2d — Verify `generated/` is gone

```bash
ls /Users/linghuang/Git/Agentic-Search/generated/ 2>&1
```

Expected: `ls: cannot access .../generated/: No such file or directory`

- [ ] **Step 2d: Confirm directory is deleted**

### Step 2e — Run unit tests

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/ -x -q
```

Expected: all pass.

- [ ] **Step 2e: Run unit tests**

### Step 2f — Lint

```bash
cd /Users/linghuang/Git/Agentic-Search && ruff check . --fix && ruff format .
```

- [ ] **Step 2f: Lint**

### Step 2g — Commit

```bash
cd /Users/linghuang/Git/Agentic-Search && git add pyproject.toml && git commit -m "chore: delete generated/ OpenAPI stub directory and remove from pythonpath"
```

- [ ] **Step 2g: Commit**

---

## Self-Review

### Spec coverage
- ✅ `_cc_pair_creator` replaced with direct `requests.put` — Task 1a
- ✅ `config.py` deleted (only callers: `cc_pair.py` which no longer needs it) — Task 1b
- ✅ `ApiException` replaced with `requests.HTTPError` in test — Task 1c
- ✅ `generated/` directory deleted — Task 2b
- ✅ `pyproject.toml` pythonpath updated — Task 2c

### Scope note on `access_type` serialization
`AccessType` is imported from `tests.integration.common_utils.types` — it's a string enum whose `.value` will be serialized correctly by `requests` when passed in `json=`. No additional serialization needed.

### Scope note on the REST endpoint
`PUT /manage/admin/connector/{connector_id}/credential/{credential_id}` follows the REST convention for this codebase. The response is expected to be `{"data": <cc_pair_id>}`. If the backend returns a different shape, the integration test will fail with a clear `KeyError` rather than an `AttributeError` from a missing stub class — a strict improvement.

### Placeholder scan
No TBD or vague steps. All code is complete.
