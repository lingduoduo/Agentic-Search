# Corpus Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a corpus registry (manifest + resolver) so the demo/hybrid retrieval servers can load a named corpus or a union of corpora, extending search coverage; fold in zero-reference `data/` cleanup.

**Architecture:** A `data/corpora.json` manifest names each corpus. A new `corpus_registry.py` resolves a spec (name / `all` / comma-list / path) to a `list[dict]` of docs. `TfidfRetriever` gains a `from_docs()` classmethod so both servers can build from resolved docs while keeping their existing path-based constructor. A `--corpus` flag is added alongside the existing `--corpus_path`.

**Tech Stack:** Python, argparse, scikit-learn TF-IDF, pytest.

## Global Constraints

- All corpora share the schema `{id, title, contents, metadata}`.
- `--corpus_path <file>` must keep working exactly as today (back-compat); the default local demo run is unchanged.
- Do not touch BEIR eval (`beir_eval.py` reads `data/beir/<dataset>/` raw), the `vocabulary_corpus.json` artifact, or the frontend example chips.
- `TfidfRetriever(corpus_path)` signature and its 5 existing call sites stay unchanged.
- Never commit to `main`; work on branch `feat/corpus-registry` (already created).

---

### Task 1: Corpus registry resolver + manifest

**Files:**
- Create: `data/corpora.json`
- Create: `src/internal/servers/retrieval/corpus_registry.py`
- Test: `tests/unit/servers/retrieval/test_corpus_registry.py`

**Interfaces:**
- Consumes: `demo._load_corpus(path) -> list[dict]` (existing jsonl loader).
- Produces:
  - `load_manifest(path: str = "data/corpora.json") -> dict[str, dict]`
  - `resolve_corpus_docs(spec: str, manifest: dict | None = None) -> list[dict]`

- [ ] **Step 1: Create the manifest**

Create `data/corpora.json`:

```json
{
  "demo": {"path": "data/corpus.jsonl", "domain": "retrieval/ML", "docs": 30},
  "scifact": {"path": "data/corpus_scifact.jsonl", "domain": "scientific", "docs": 5183},
  "nfcorpus": {"path": "data/corpus_nfcorpus.jsonl", "domain": "medical", "docs": 3633}
}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/servers/retrieval/test_corpus_registry.py`:

```python
import json

import pytest

from src.internal.servers.retrieval.corpus_registry import (
    load_manifest,
    resolve_corpus_docs,
)


def _write_corpus(path, docs):
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


@pytest.fixture
def manifest(tmp_path):
    _write_corpus(tmp_path / "a.jsonl", [{"id": "a1", "title": "A1", "contents": "alpha"}])
    _write_corpus(
        tmp_path / "b.jsonl",
        [
            {"id": "b1", "title": "B1", "contents": "beta"},
            {"id": "a1", "title": "dup", "contents": "duplicate of a1"},
        ],
    )
    return {
        "a": {"path": str(tmp_path / "a.jsonl")},
        "b": {"path": str(tmp_path / "b.jsonl")},
    }


def test_resolve_by_name(manifest):
    docs = resolve_corpus_docs("a", manifest)
    assert [d["id"] for d in docs] == ["a1"]


def test_resolve_all_unions_and_dedupes_by_id(manifest):
    docs = resolve_corpus_docs("all", manifest)
    # a1 from "a" wins; a1 duplicate in "b" is dropped; b1 kept.
    assert [d["id"] for d in docs] == ["a1", "b1"]


def test_resolve_comma_list(manifest):
    docs = resolve_corpus_docs("b,a", manifest)
    # b first (b1, a1-dup), then a's a1 is a dup and dropped.
    assert [d["id"] for d in docs] == ["b1", "a1"]


def test_resolve_path_backcompat(tmp_path, manifest):
    p = tmp_path / "direct.jsonl"
    _write_corpus(p, [{"id": "z", "title": "Z", "contents": "zeta"}])
    docs = resolve_corpus_docs(str(p), manifest)
    assert [d["id"] for d in docs] == ["z"]


def test_resolve_unknown_spec_raises(manifest):
    with pytest.raises(ValueError, match="Unknown corpus spec"):
        resolve_corpus_docs("nope", manifest)


def test_load_manifest_missing_file_returns_empty(tmp_path):
    assert load_manifest(str(tmp_path / "absent.json")) == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/retrieval/test_corpus_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: ... corpus_registry`.

- [ ] **Step 4: Implement the resolver**

Create `src/internal/servers/retrieval/corpus_registry.py`:

```python
"""Corpus registry: resolve a corpus spec to a list of documents.

A spec is one of:
  - a registered name from data/corpora.json (e.g. "demo")
  - "all" — the union of every registered corpus, deduped by id
  - a comma-separated list of names (e.g. "demo,scifact")
  - a filesystem path to a .jsonl corpus (back-compat with --corpus_path)
"""

from __future__ import annotations

import json
import logging
import os

from src.internal.servers.retrieval.demo import _load_corpus

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "data/corpora.json"


def load_manifest(path: str = DEFAULT_MANIFEST_PATH) -> dict[str, dict]:
    """Load the corpus manifest; a missing file yields an empty manifest so
    path-only (--corpus_path) usage keeps working with no manifest present."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _dedupe_by_id(docs: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for d in docs:
        doc_id = d.get("id")
        if doc_id is not None and doc_id in seen:
            logger.warning("Dropping duplicate corpus id %r", doc_id)
            continue
        if doc_id is not None:
            seen.add(doc_id)
        out.append(d)
    return out


def resolve_corpus_docs(spec: str, manifest: dict | None = None) -> list[dict]:
    if manifest is None:
        manifest = load_manifest()

    if spec == "all":
        names = list(manifest.keys())
        if not names:
            raise ValueError("No corpora registered in the manifest for 'all'.")
    else:
        candidate = [s.strip() for s in spec.split(",") if s.strip()]
        if candidate and all(n in manifest for n in candidate):
            names = candidate
        elif os.path.exists(spec):
            docs = _load_corpus(spec)
            logger.info("Loaded corpus from path %s (%d docs)", spec, len(docs))
            return docs
        else:
            available = ", ".join(sorted(manifest)) or "(none)"
            raise ValueError(
                f"Unknown corpus spec {spec!r}. Available: {available}, or a file path."
            )

    docs: list[dict] = []
    for name in names:
        entry = manifest[name]
        path = entry["path"] if isinstance(entry, dict) else entry
        docs.extend(_load_corpus(path))
    docs = _dedupe_by_id(docs)
    logger.info("Loaded corpora %s (%d docs after dedupe)", names, len(docs))
    return docs
```

Note: `corpus_registry` imports `_load_corpus` from `demo`. `demo` must NOT import
`corpus_registry` at module top (that would be a cycle); Task 3 imports it inside
`demo.main()` instead.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/servers/retrieval/test_corpus_registry.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add data/corpora.json src/internal/servers/retrieval/corpus_registry.py tests/unit/servers/retrieval/test_corpus_registry.py
git commit -m "feat: add corpus registry resolver + data/corpora.json manifest"
```

---

### Task 2: `TfidfRetriever.from_docs` classmethod

**Files:**
- Modify: `src/internal/servers/retrieval/demo.py` (`TfidfRetriever`, lines 42-50)
- Test: `tests/unit/servers/retrieval/test_demo_retrieval.py`

**Interfaces:**
- Produces: `TfidfRetriever.from_docs(docs: list[dict]) -> TfidfRetriever`, building the
  same TF-IDF matrix as `__init__(corpus_path)` but from pre-loaded docs.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/servers/retrieval/test_demo_retrieval.py`:

```python
def test_from_docs_builds_retriever_without_a_file():
    docs = [
        {"id": "a", "title": "Cats", "contents": "feline animals purr"},
        {"id": "b", "title": "Dogs", "contents": "canine animals bark"},
    ]
    retriever = TfidfRetriever.from_docs(docs)
    rows = retriever.retrieve(["feline purr"], topk=5)
    assert rows[0][0]["document"]["id"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/retrieval/test_demo_retrieval.py::test_from_docs_builds_retriever_without_a_file -v`
Expected: FAIL with `AttributeError: ... has no attribute 'from_docs'`.

- [ ] **Step 3: Refactor `TfidfRetriever` to share a builder**

In `src/internal/servers/retrieval/demo.py`, replace the class body's `__init__`
(lines 43-50) with an `__init__` that delegates to a `_build` method, and add the
`from_docs` classmethod:

```python
class TfidfRetriever:
    def __init__(self, corpus_path: str) -> None:
        self._build(_load_corpus(corpus_path))

    @classmethod
    def from_docs(cls, docs: list[dict]) -> "TfidfRetriever":
        obj = cls.__new__(cls)
        obj._build(docs)
        return obj

    def _build(self, docs: list[dict]) -> None:
        self._docs = docs
        texts = [
            f"{d.get('title', '')} {d.get('contents', d.get('text', ''))}"
            for d in self._docs
        ]
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(texts)
```

Leave the `retrieve` method unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/servers/retrieval/test_demo_retrieval.py -v`
Expected: PASS (3 tests — the 2 existing path-based + the new `from_docs`).

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/retrieval/demo.py tests/unit/servers/retrieval/test_demo_retrieval.py
git commit -m "feat: add TfidfRetriever.from_docs classmethod"
```

---

### Task 3: Wire `--corpus` into demo.py + hybrid.py

**Files:**
- Modify: `src/internal/servers/retrieval/demo.py` (`parse_args` lines 113-126, `main` lines 129-134)
- Modify: `src/internal/servers/retrieval/hybrid.py` (`parse_args` 135-156, `_build_dense` 159-172, `main` 175-181)
- Modify: `README.md` (retrieval server run section)
- Test: `tests/unit/servers/retrieval/test_corpus_registry.py` (add a wiring/integration test)

**Interfaces:**
- Consumes: `resolve_corpus_docs` (Task 1), `TfidfRetriever.from_docs` (Task 2).
- Produces: both servers accept exactly one of `--corpus` (name/`all`/comma-list) or
  `--corpus_path` (file path).

- [ ] **Step 1: Write the failing integration test**

Add to `tests/unit/servers/retrieval/test_corpus_registry.py`:

```python
from src.internal.servers.retrieval.demo import TfidfRetriever


def test_union_docs_feed_from_docs_retriever(manifest):
    docs = resolve_corpus_docs("all", manifest)
    retriever = TfidfRetriever.from_docs(docs)
    rows = retriever.retrieve(["beta"], topk=5)
    assert rows[0][0]["document"]["id"] == "b1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/servers/retrieval/test_corpus_registry.py::test_union_docs_feed_from_docs_retriever -v`
Expected: FAIL — at this point `resolve_corpus_docs` + `from_docs` exist (Tasks 1-2), so this test should actually PASS already. If it passes, that confirms the integration; proceed to wire the CLIs. (This step documents the integration contract; no code change needed if green.)

- [ ] **Step 3: Wire `demo.py`**

Replace `parse_args` (lines 113-126) so `--corpus` and `--corpus_path` are mutually
exclusive and one is required:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo TF-IDF retrieval server")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--corpus_path", type=str, help="Path to a corpus .jsonl file"
    )
    source.add_argument(
        "--corpus",
        type=str,
        help="Registered corpus name, comma-list, or 'all' (see data/corpora.json)",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    add_host_port_args(
        parser,
        "DEMO_RETRIEVAL_HOST",
        "DEMO_RETRIEVAL_PORT",
        default_host=DEFAULT_HOST,
        default_port=DEFAULT_PORT,
    )
    return parser.parse_args()
```

Replace `main` (lines 129-134):

```python
def main() -> None:
    load_environment()
    args = parse_args()
    # Imported here (not at module top) to avoid a circular import:
    # corpus_registry imports _load_corpus from this module.
    from src.internal.servers.retrieval.corpus_registry import resolve_corpus_docs

    docs = resolve_corpus_docs(args.corpus or args.corpus_path)
    retriever = TfidfRetriever.from_docs(docs)
    app = create_app(retriever)
    run_uvicorn_app(app, host=args.host, port=args.port)
```

- [ ] **Step 4: Wire `hybrid.py`**

Replace `parse_args` source args (lines 139-141) with the same mutually-exclusive
group. Replace lines 139-141:

```python
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Path to corpus.jsonl"
    )
```

with:

```python
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--corpus_path", type=str, help="Path to a corpus .jsonl file"
    )
    source.add_argument(
        "--corpus",
        type=str,
        help="Registered corpus name, comma-list, or 'all' (see data/corpora.json)",
    )
```

Change `_build_dense` (lines 159-163) to take pre-loaded docs instead of a path:

```python
def _build_dense(docs: list[dict], device: str) -> DenseEmbeddingRetriever | None:
    try:
        encoder = build_e5_encoder(device=device)
        return DenseEmbeddingRetriever(docs, encoder=encoder)
    except Exception as exc:  # missing deps, model download, MPS unavailable
        logger.warning(
            "Dense leg unavailable, serving TF-IDF only. This usually means the "
            "embedding stack is misconfigured: sentence-transformers < 3.0, or a "
            "torch/torchvision version mismatch from a shared env. Install into a "
            "clean venv per docs/hybrid-dense-setup.md. Cause: %s",
            exc,
        )
        return None
```

Replace `main` (lines 175-181):

```python
def main() -> None:
    load_environment()
    args = parse_args()
    from src.internal.servers.retrieval.corpus_registry import resolve_corpus_docs

    docs = resolve_corpus_docs(args.corpus or args.corpus_path)
    sparse = TfidfRetriever.from_docs(docs)
    dense = None if args.no_dense else _build_dense(docs, args.device)
    app = create_app(dense=dense, sparse=sparse)
    run_uvicorn_app(app, host=args.host, port=args.port)
```

The now-unused `_load_corpus` import in `hybrid.py` (line 26) becomes dead — remove
`_load_corpus` from the `from ...demo import (...)` block, keeping `TfidfRetriever`
and `DEFAULT_TOPK`.

- [ ] **Step 5: Verify imports resolve and no cycle**

Run: `python -c "import src.internal.servers.retrieval.demo, src.internal.servers.retrieval.hybrid, src.internal.servers.retrieval.corpus_registry; print('imports OK')"`
Expected: prints `imports OK` (no ImportError / circular import).

- [ ] **Step 6: Update README**

In `README.md`, find the retrieval-server run section (the `--corpus_path data/corpus.jsonl` demo command). Add, right after the existing demo command block:

```markdown
# Named corpora / union via the registry (data/corpora.json):
python3 -m src.internal.servers.retrieval.demo --corpus demo   # curated 30-doc demo (default)
python3 -m src.internal.servers.retrieval.demo --corpus all    # union of all registered corpora
```

- [ ] **Step 7: Run the retrieval test suite**

Run: `python -m pytest tests/unit/servers/retrieval/ -v`
Expected: PASS. If a hybrid test constructed `_build_dense(path, ...)` with a path,
update it to pass a docs list (grep the failure).

- [ ] **Step 8: Commit**

```bash
git add src/internal/servers/retrieval/demo.py src/internal/servers/retrieval/hybrid.py tests/unit/servers/retrieval/test_corpus_registry.py README.md
git commit -m "feat: add --corpus name/union flag to demo + hybrid servers"
```

---

### Task 4: Cleanup zero-reference `data/` files

**Files:**
- Delete: `data/case.txt`, `data/intent_examples.sample.json`, `data/vocabulary_corpus.json`, `data/nq_rag/`

**Interfaces:** none (deletions only).

- [ ] **Step 1: Re-verify each path is unreferenced**

Run:
```bash
grep -rn "case.txt\|intent_examples.sample\|nq_rag" --include="*.py" --include="*.ts" --include="*.tsx" --include="*.sh" --include="*.yml" --include="*.yaml" --include="*.toml" src tests examples web docker
grep -rn "data/vocabulary_corpus.json" --include="*.py" src tests examples
```
Expected: no non-doc code references to `case.txt`, `intent_examples.sample`, or the
root `data/vocabulary_corpus.json`. (`document_index/cli.py` writes to
`save_dir/vocabulary_corpus.json` under `data/indexes/`, a different path — leave it.)
If any code reference appears, STOP and report it instead of deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm data/case.txt data/intent_examples.sample.json data/vocabulary_corpus.json
rmdir data/nq_rag 2>/dev/null || true
```

(`data/nq_rag/` is an empty untracked dir; `rmdir` removes it if present. If it is
tracked or non-empty, skip it and note so.)

- [ ] **Step 3: Verify the suite still passes**

Run: `python -m pytest tests/unit/servers/retrieval/ tests/unit/test_intent_classifier.py -q`
Expected: PASS (nothing depended on the deleted files).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove zero-reference data/ dead files (case.txt, dup intent examples, stray vocab, empty nq_rag)"
```

---

## Final verification

- [ ] `python -m pytest tests/unit/servers/retrieval/ -q` — green.
- [ ] `python -c "import src.internal.servers.retrieval.hybrid"` — no circular import.
- [ ] `ruff check src/internal/servers/retrieval/ tests/unit/servers/retrieval/` — clean.
- [ ] Manual smoke (optional): `python3 -m src.internal.servers.retrieval.demo --corpus all` starts and serves; `--corpus bogus` exits with the available-names error; `--corpus_path data/corpus.jsonl` still works.
