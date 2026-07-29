# MCP Document Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one native Agentic Search MCP tool that safely extracts bounded content from base64-encoded PDF, DOCX, PPTX, CSV, and TXT documents.

**Architecture:** A focused `documents.py` MCP module owns request validation, strict decoding, format dispatch, extraction helpers, response normalization, and FastMCP registration. It accepts document bytes rather than local paths or URLs, runs synchronous work off the event loop, and places untrusted PDF/Office parsers in concurrency-limited disposable processes with ZIP preflight, wall-time, CPU, and memory limits. It returns ordinary dictionaries and imports format libraries only when the selected format needs them.

**Tech Stack:** Python 3.10+, FastMCP, pytest, standard-library `base64`/`csv`/`multiprocessing`/`resource`/`zipfile`, psutil watchdog support, and optional pypdf, python-docx, and python-pptx.

## Global Constraints

- Expose exactly one authenticated MCP tool named `extract_document`.
- Accept `file_name: str`, `content_base64: str`, `page_range: str | None = None`, and `max_rows: int = 1000`.
- Support `.pdf`, `.docx`, `.pptx`, `.csv`, and `.txt`; reject other or missing extensions.
- Never accept or dereference a server-local path or remote URL.
- Return `{"document": {...}}` on success and `{"error": str, "document": None}` on failure.
- Do not expose tracebacks, temporary paths, document bytes, or host details in errors.
- Enforce `MAX_INPUT_BYTES = 20 * 1024 * 1024`, `MAX_RESPONSE_CHARS = 50_000`, and `MAX_CSV_ROWS = 10_000`.
- Parse PDF page ranges as one-based, deduplicated, sorted, bounded selections; reject malformed and descending ranges.
- Run synchronous dispatch with `asyncio.to_thread`; run PDF/DOCX/PPTX parsing in killable, concurrency-limited child processes. Enforce CPU/memory with Linux kernel limits and a sampled psutil parent watchdog on macOS/Windows.
- Keep parser inputs memory-backed; if a future parser requires a temporary file, delete it in a `finally` block.
- Importing the MCP server must succeed without optional document parser libraries.
- Do not modify or delete the user-supplied reference files `src/internal/document_index/base.py` and `src/internal/document_index/document_processing.py`.
- Any `.superpowers/sdd/*report.md` produced for TDD implementation work must follow `docs/development/self-review-reports.md` and pass `python examples/validate_task_report.py --require-tdd REPORT_FILE` before review. A report produced only for Task 6 verification uses normal canonical validation without `--require-tdd`; do not fabricate RED/GREEN evidence for verification-only work.

---

## File Structure

- Create `src/internal/mcp_server/tools/documents.py`: MCP registration, validation, dispatch, bounded result builders, and format extractors.
- Create `tests/unit/test_mcp_document_tools.py`: focused behavior, safety, cleanup, parser, and registration tests.
- Modify `src/internal/mcp_server/api.py`: import the new tool module after `mcp_server` construction.
- Modify `src/internal/mcp_server/tools/__init__.py`: include the new registration module in package imports.
- Modify `pyproject.toml`: define the optional `mcp-documents` dependency extra.
- Modify `src/internal/mcp_server/README.md`: document the new tool, safe input model, installation extra, and example.

### Task 1: Input validation, TXT extraction, and response bounds

**Files:**
- Create: `src/internal/mcp_server/tools/documents.py`
- Create: `tests/unit/test_mcp_document_tools.py`

**Interfaces:**
- Produces: `parse_page_range(page_range: str, total_pages: int) -> list[int]`
- Produces: `_decode_document(content_base64: str) -> bytes`
- Produces: `_extract_document_sync(file_name: str, data: bytes, page_range: str | None, max_rows: int) -> dict[str, Any]`
- Produces: `async extract_document(file_name: str, content_base64: str, page_range: str | None = None, max_rows: int = 1000) -> dict[str, Any]`
- Response invariant: success contains only `document`; failure contains `error` and `document=None`.

- [ ] **Step 1: Write failing tests for TXT success and strict request validation**

Add tests that encode payloads with `base64.b64encode`, call
`extract_document`, and assert:

```python
@pytest.mark.asyncio
async def test_extract_document_decodes_txt():
    result = await documents.extract_document(
        "notes.TXT", base64.b64encode(b"alpha\\nbeta").decode()
    )
    assert result == {
        "document": {
            "file_name": "notes.TXT",
            "file_type": "txt",
            "text": "alpha\\nbeta",
            "text_length": 10,
            "truncated": False,
        }
    }


@pytest.mark.parametrize(
    ("file_name", "payload", "error_fragment"),
    [
        ("", "YQ==", "file name"),
        ("no-extension", "YQ==", "extension"),
        ("data.bin", "YQ==", "unsupported"),
        ("data.txt", "%%%", "base64"),
        ("data.txt", "", "empty"),
    ],
)
@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_input(
    file_name, payload, error_fragment
):
    result = await documents.extract_document(file_name, payload)
    assert result["document"] is None
    assert error_fragment in result["error"].lower()
```

Also cover invalid UTF-8, decoded input over `MAX_INPUT_BYTES`, and output over
`MAX_OUTPUT_CHARS` by monkeypatching the constants to small values.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py
```

Expected: collection fails because
`src.internal.mcp_server.tools.documents` does not exist.

- [ ] **Step 3: Implement the minimal TXT path and shared validation**

Create `documents.py` with:

```python
from __future__ import annotations

import asyncio
import base64
import binascii
from pathlib import PurePath
from typing import Any

from ..api import mcp_server

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_CHARS = 50_000
MAX_CSV_ROWS = 10_000
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv", ".txt"}


def _error(message: str) -> dict[str, Any]:
    return {"error": message, "document": None}


def _decode_document(content_base64: str) -> bytes:
    if not content_base64:
        raise ValueError("Document content must not be empty.")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Document content is not valid base64.") from exc
    if not data:
        raise ValueError("Decoded document must not be empty.")
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("Decoded document exceeds the 20 MiB input limit.")
    return data


def _bounded_text(text: str) -> tuple[str, int, bool]:
    return text[:MAX_OUTPUT_CHARS], len(text), len(text) > MAX_OUTPUT_CHARS
```

Validate that `PurePath(file_name).name == file_name`, derive a lowercase
suffix, reject unsupported formats, strictly decode TXT as UTF-8, and use
`_bounded_text` for its result. Decorate only the public async function:

```python
@mcp_server.tool()
async def extract_document(
    file_name: str,
    content_base64: str,
    page_range: str | None = None,
    max_rows: int = 1000,
) -> dict[str, Any]:
    try:
        data = _decode_document(content_base64)
        return await asyncio.to_thread(
            _extract_document_sync, file_name, data, page_range, max_rows
        )
    except (ValueError, ImportError) as exc:
        return _error(str(exc))
    except Exception:
        return _error("Document extraction failed.")
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the validated TXT slice**

```bash
git add src/internal/mcp_server/tools/documents.py tests/unit/test_mcp_document_tools.py
git commit -m "feat: add safe MCP text document extraction"
```

### Task 2: PDF extraction and page-range semantics

**Files:**
- Modify: `src/internal/mcp_server/tools/documents.py`
- Modify: `tests/unit/test_mcp_document_tools.py`

**Interfaces:**
- Consumes: `_bounded_text(text: str) -> tuple[str, int, bool]`
- Produces: `parse_page_range(page_range: str, total_pages: int) -> list[int]`
- Produces: `_extract_pdf(data: bytes, file_name: str, page_range: str | None) -> dict[str, Any]`

- [ ] **Step 1: Write failing page-range and PDF tests**

Add parameterized range expectations:

```python
@pytest.mark.parametrize(
    ("raw", "total", "expected"),
    [
        ("1", 5, [0]),
        ("1-3,5", 5, [0, 1, 2, 4]),
        ("3,1,3", 5, [0, 2]),
        ("1-99", 3, [0, 1, 2]),
        ("99", 3, []),
    ],
)
def test_parse_page_range(raw, total, expected):
    assert documents.parse_page_range(raw, total) == expected
```

Assert `ValueError` for `""`, `"0"`, `"-1"`, `"3-1"`, `"1-"`, `"1--2"`,
`"a"`, and total pages less than one. Generate a two-page PDF using pypdf
and assert selected page
headers, total pages, extracted pages, and text. Monkeypatch `__import__` or the
module import helper to assert a missing pypdf returns an actionable
`mcp-documents` installation error.

- [ ] **Step 2: Run PDF tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py -k "page_range or pdf"
```

Expected: failures because page-range validation and PDF dispatch are absent.

- [ ] **Step 3: Implement PDF parsing with lazy imports**

Implement `parse_page_range` by splitting comma segments, requiring positive
decimal one-based bounds, rejecting descending ranges, clamping range ends,
then returning `sorted(set(pages))`. Add:

```python
def _require_module(module_name: str, distribution_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{distribution_name} is required; install agentic-search[mcp-documents]."
        ) from exc
```

Use `io.BytesIO(data)` with `pypdf.PdfReader`, extract selected pages, convert
`None` page text to `""`, add `--- Page N ---` labels, and bound the combined
text. Reject a non-`None` `page_range` for every non-PDF format.

- [ ] **Step 4: Run PDF tests and verify GREEN**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py -k "page_range or pdf"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the PDF slice**

```bash
git add src/internal/mcp_server/tools/documents.py tests/unit/test_mcp_document_tools.py
git commit -m "feat: extract bounded PDF content over MCP"
```

### Task 3: DOCX, PPTX, and CSV extraction

**Files:**
- Modify: `src/internal/mcp_server/tools/documents.py`
- Modify: `tests/unit/test_mcp_document_tools.py`

**Interfaces:**
- Produces: `_extract_docx(data: bytes, file_name: str) -> dict[str, Any]`
- Produces: `_extract_pptx(data: bytes, file_name: str) -> dict[str, Any]`
- Produces: `_extract_csv(data: bytes, file_name: str, max_rows: int) -> dict[str, Any]`
- Nested-data invariant: tables, slides, and CSV records stop growing when the shared output-character budget is exhausted.

- [ ] **Step 1: Write failing format tests**

Generate minimal DOCX and PPTX documents into `io.BytesIO` when their libraries
are present. Assert paragraph/table and slide extraction. Add CSV tests using
only encoded byte literals:

```python
@pytest.mark.asyncio
async def test_csv_exact_row_limit_is_not_truncated():
    payload = base64.b64encode(b"name,value\\na,1\\nb,2\\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)
    assert result["document"]["rows"] == 2
    assert result["document"]["truncated"] is False


@pytest.mark.asyncio
async def test_csv_over_row_limit_is_truncated():
    payload = base64.b64encode(b"name,value\\na,1\\nb,2\\nc,3\\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)
    assert result["document"]["rows"] == 2
    assert result["document"]["truncated"] is True
```

Cover `max_rows` values `0`, `-1`, and `MAX_CSV_ROWS + 1`; malformed UTF-8;
duplicate CSV headings; nested-data bounding; and missing python-docx or
python-pptx installation errors.

- [ ] **Step 2: Run format tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py -k "docx or pptx or csv"
```

Expected: failures because these dispatch branches are absent.

- [ ] **Step 3: Implement bounded format extractors**

For DOCX and PPTX, lazily import `docx` and `pptx`, use `io.BytesIO(data)`, and
build JSON-compatible dictionaries. Track the serialized character cost while
adding table cells and slide objects; stop adding nested entries at
`MAX_OUTPUT_CHARS`.

For CSV, use standard-library `csv.DictReader` over
`io.StringIO(data.decode("utf-8"))`, reject absent or duplicate headings, read
at most `max_rows + 1` records, return only the first `max_rows`, and set
`truncated = len(rows) > max_rows`. Validate
`1 <= max_rows <= MAX_CSV_ROWS` before parsing.

- [ ] **Step 4: Run the complete focused suite and verify GREEN**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit all document formats**

```bash
git add src/internal/mcp_server/tools/documents.py tests/unit/test_mcp_document_tools.py
git commit -m "feat: extract office and CSV documents over MCP"
```

### Task 4: MCP registration and event-loop/cleanup guarantees

**Files:**
- Modify: `src/internal/mcp_server/api.py`
- Modify: `src/internal/mcp_server/tools/__init__.py`
- Modify: `src/internal/mcp_server/tools/documents.py`
- Modify: `tests/unit/test_mcp_document_tools.py`

**Interfaces:**
- Consumes: the decorated `extract_document` function from Task 1.
- Produces: an `extract_document` entry in `await mcp_server.list_tools()`.

- [ ] **Step 1: Write failing registration and execution-boundary tests**

Add a static startup-wiring test that reads `api.py` and
`tools/__init__.py`, parses their imports, and requires both modules to import
`documents`. This is the RED gate: importing `documents` directly in earlier
tests already executes its decorator, so a runtime-only discovery assertion
could pass even when production startup never imports the module.

Also retain runtime discovery as a GREEN integration assertion:

```python
@pytest.mark.asyncio
async def test_extract_document_is_registered():
    from src.internal.mcp_server.api import mcp_server

    names = {tool.name for tool in await mcp_server.list_tools()}
    assert "extract_document" in names
```

Monkeypatch `asyncio.to_thread` with an async spy and assert the public tool
uses it. If a parser path uses a temporary file, monkeypatch
`tempfile.NamedTemporaryFile`, force both success and parser failure, and
assert the created path no longer exists after each call. If every selected
library accepts `BytesIO`, assert no temporary file is created and remove the
unused cleanup path from production code.

- [ ] **Step 2: Run registration/boundary tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py -k "registered or thread or temporary"
```

Expected: the static startup-wiring test fails until both production modules
explicitly import `documents`; any missing execution-boundary behavior also
fails.

- [ ] **Step 3: Wire MCP startup imports and finalize cleanup behavior**

Import `documents` beside the other tool modules in both
`src/internal/mcp_server/api.py` and
`src/internal/mcp_server/tools/__init__.py`. Preserve the import-after-server
ordering in `api.py`. Keep `asyncio.to_thread` in the public entry point.
Implement a `NamedTemporaryFile` plus `os.unlink` in `finally` only if a parser
demonstrably cannot consume `BytesIO`; otherwise keep extraction memory-backed
and omit filesystem code.

- [ ] **Step 4: Run focused and existing MCP tests**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py tests/unit/test_mcp_server.py tests/unit/test_mcp_auth.py tests/unit/test_mcp_dynamic_bridge.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit MCP integration**

```bash
git add src/internal/mcp_server/api.py src/internal/mcp_server/tools/__init__.py src/internal/mcp_server/tools/documents.py tests/unit/test_mcp_document_tools.py
git commit -m "feat: register document extraction MCP tool"
```

### Task 5: Dependency declaration and MCP documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/internal/mcp_server/README.md`
- Modify: `tests/unit/test_mcp_document_tools.py`

**Interfaces:**
- Produces: `agentic-search[mcp-documents]` with `pypdf>=6.12.2`, `psutil`, `python-docx`, and `python-pptx`.
- Documents: tool schema and safe direct-content boundary.

- [ ] **Step 1: Write failing dependency-contract test**

Parse `pyproject.toml` with `tomllib` and assert:

```python
def test_document_parser_extra_declares_all_optional_parsers():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    requirements = project["optional-dependencies"]["mcp-documents"]
    assert any(item.startswith("pypdf>=6.12.2") for item in requirements)
    assert any(item.startswith("psutil") for item in requirements)
    assert any(item.startswith("python-docx") for item in requirements)
    assert any(item.startswith("python-pptx") for item in requirements)
```

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py -k parser_extra
```

Expected: failure because `mcp-documents` is not declared.

- [ ] **Step 3: Declare dependencies and document the tool**

Add:

```toml
mcp-documents = [
    "pypdf>=6.12.2",
    "psutil>=5.9.0",
    "python-docx>=1.1.0",
    "python-pptx>=1.0.0",
]
```

Update the MCP README capabilities list and add:

```json
{
  "file_name": "report.pdf",
  "content_base64": "<base64 document bytes>",
  "page_range": "1-3"
}
```

State the 20 MiB decoded-input limit, 50,000-character output limit, supported
extensions, CSV row limit, optional-extra installation command, and that paths
and URLs are intentionally unsupported.

- [ ] **Step 4: Run the focused suite and validate documentation diff**

Run:

```bash
pytest -q tests/unit/test_mcp_document_tools.py
git diff --check
```

Expected: tests pass and `git diff --check` prints no diagnostics.

- [ ] **Step 5: Commit dependencies and docs**

```bash
git add pyproject.toml src/internal/mcp_server/README.md tests/unit/test_mcp_document_tools.py
git commit -m "docs: describe MCP document extraction"
```

### Task 6: End-to-end MCP validation and completion evidence

**Files:**
- Modify only if verification reveals a defect in an already scoped file.

**Interfaces:**
- Verifies: FastMCP discovery and direct invocation of `extract_document`.

- [ ] **Step 1: Run the complete focused MCP test set**

```bash
pytest -q tests/unit/test_mcp_document_tools.py tests/unit/test_mcp_server.py tests/unit/test_mcp_auth.py tests/unit/test_mcp_dynamic_bridge.py
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 2: Run the default repository suite**

```bash
pytest -q
```

Expected: all default unit and regression tests pass. If an unrelated
pre-existing failure occurs, record the exact command and output separately
without weakening focused acceptance criteria.

- [ ] **Step 3: Validate FastMCP tool discovery and invocation**

Run a short Python command that:

```python
import asyncio
import base64
from src.internal.mcp_server.api import mcp_server
from src.internal.mcp_server.tools.documents import extract_document

async def main():
    names = {tool.name for tool in await mcp_server.list_tools()}
    assert "extract_document" in names
    result = await extract_document(
        "smoke.txt", base64.b64encode(b"MCP document smoke test").decode()
    )
    assert result["document"]["text"] == "MCP document smoke test"

asyncio.run(main())
```

Expected: exit code 0 with no assertion failure.

- [ ] **Step 4: Inspect the final scope and safety diff**

Run:

```bash
git status --short
git diff --check HEAD~5..HEAD
git diff --stat HEAD~5..HEAD
rg -n "requests|https?://|validate_file_path|TextContent|ActionResponse|traceback" src/internal/mcp_server/tools/documents.py
```

Expected: only planned files plus the user's two untouched reference files are
present; diff checks pass; the safety scan returns no production matches.

- [ ] **Step 5: Apply verification-before-completion**

Read and follow `superpowers:verification-before-completion`. Report the exact
commands, current passing counts, skipped optional-parser tests, and any
pre-existing failures. Do not claim completion based on earlier test output.
