# Semantic Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in semantic chunking (embedding-similarity, per-document percentile breakpoint) to `document_index/chunking.py`, default-off, wired through the indexing pipeline.

**Architecture:** Pure-logic TDD. New `_split_text_semantic` in `chunking.py`; three new `ChunkingConfig` fields; an optional `embedding_fn` threaded through `chunk_document`/`chunk_documents` and passed by `pipeline.py`. `chunking.py` stays independent of `embedding.py` (structural `Callable` + direct `numpy`). Spec: `docs/superpowers/specs/2026-07-24-semantic-chunking-design.md`.

**Tech Stack:** Python, numpy, pytest, ruff.

## Global Constraints

- Branch: `feat/semantic-chunking`, off `main`. Never commit to `main`.
- `semantic_chunking=False` (default) ⇒ byte-identical output to today.
- `chunking.py` must NOT import from `src.internal.document_index.embedding`. The embedder enters only as an injected `Callable[[list[str]], "np.ndarray"] | None`.
- All fallbacks return `_split_text_paragraphs(...)` and never raise.
- `ruff check .` + `pytest` green at the end of every task.
- Tests use the existing `deterministic_embedding_fn` (`embedding.py:392`) — assert mechanics (boundary at a controlled distance spike), not semantic quality.

---

### Task 1: ChunkingConfig fields + validation

**Files:**
- Modify: `src/internal/document_index/models.py` (`ChunkingConfig`)
- Test: `tests/unit/document_index/test_chunking_config.py` (create; or add to an existing config test file if one exists — grep first)

**Interfaces:**
- Produces: `ChunkingConfig.semantic_chunking: bool`, `.semantic_breakpoint_percentile: float`, `.semantic_buffer_size: int`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from src.internal.document_index.models import ChunkingConfig


def test_semantic_defaults_are_off():
    c = ChunkingConfig()
    assert c.semantic_chunking is False
    assert c.semantic_breakpoint_percentile == 95.0
    assert c.semantic_buffer_size == 1
    c.validate()  # defaults are valid


@pytest.mark.parametrize("pct", [0.0, 100.0, -1.0, 150.0])
def test_validate_rejects_bad_percentile(pct):
    with pytest.raises(ValueError):
        ChunkingConfig(semantic_breakpoint_percentile=pct).validate()


def test_validate_rejects_bad_buffer():
    with pytest.raises(ValueError):
        ChunkingConfig(semantic_buffer_size=0).validate()
```

- [ ] **Step 2: Run — expect failure** (`AttributeError`/no such field)

Run: `pytest tests/unit/document_index/test_chunking_config.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `ChunkingConfig` add after `min_content_tokens`:

```python
    semantic_chunking: bool = False
    semantic_breakpoint_percentile: float = 95.0
    semantic_buffer_size: int = 1
```

In `validate()` add before the final line:

```python
        if not 0 < self.semantic_breakpoint_percentile < 100:
            raise ValueError(
                "semantic_breakpoint_percentile must be between 0 and 100 (exclusive)."
            )
        if self.semantic_buffer_size < 1:
            raise ValueError("semantic_buffer_size must be at least 1.")
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/document_index/test_chunking_config.py -q && ruff check src/internal/document_index/models.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(chunking): add semantic_chunking config fields + validation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_split_text_semantic` core algorithm

**Files:**
- Modify: `src/internal/document_index/chunking.py`
- Test: `tests/unit/document_index/test_semantic_chunking.py` (create)

**Interfaces:**
- Consumes: `_split_paragraphs`, `_split_sentences_in_paragraph`, `_split_text_paragraphs`, `_token_count` (existing in chunking.py).
- Produces: `_split_text_semantic(text: str, chunk_size: int, chunk_overlap: int, embedding_fn, percentile: float, buffer_size: int) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.internal.document_index import chunking
from src.internal.document_index.embedding import deterministic_embedding_fn

EMB = deterministic_embedding_fn(dim=8)


def _semantic(text, chunk_size=900, overlap=0, embedding_fn=EMB, pct=95.0, buf=1):
    return chunking._split_text_semantic(
        text, chunk_size, overlap, embedding_fn, pct, buf
    )


def test_boundary_at_topic_shift():
    # three identical "cats" sentences then two identical "dogs" sentences:
    # within-topic distance ~0, one large spike at the cats->dogs seam.
    text = (
        "cats cats cats. cats cats cats. cats cats cats. "
        "dogs dogs dogs. dogs dogs dogs."
    )
    chunks = _semantic(text)
    assert len(chunks) == 2
    assert "cats" in chunks[0] and "dogs" not in chunks[0]
    assert "dogs" in chunks[1] and "cats" not in chunks[1]


def test_flat_document_stays_one_chunk():
    # all-identical sentences → all distances equal → strict > yields no boundary.
    text = "same same same. same same same. same same same."
    chunks = _semantic(text)
    assert len(chunks) == 1


def test_size_cap_never_exceeds_chunk_size():
    # one topic, many sentences, tiny chunk_size → size cap must re-split.
    text = " ".join(["alpha beta gamma delta." for _ in range(30)])
    chunks = _semantic(text, chunk_size=5, overlap=1)
    assert chunks
    for c in chunks:
        assert chunking._token_count(c) <= 5


def test_fallback_when_no_embedder():
    text = "one one one. two two two. three three three."
    assert _semantic(text, embedding_fn=None) == chunking._split_text_paragraphs(
        text, 900, 0
    )


def test_fallback_single_sentence():
    text = "only one sentence here."
    assert _semantic(text) == chunking._split_text_paragraphs(text, 900, 0)


def test_fallback_on_embedder_error():
    def boom(_):
        raise RuntimeError("embedder down")

    text = "one one one. two two two. three three three."
    assert _semantic(text, embedding_fn=boom) == chunking._split_text_paragraphs(
        text, 900, 0
    )
```

- [ ] **Step 2: Run — expect failure** (`AttributeError: _split_text_semantic`)

Run: `pytest tests/unit/document_index/test_semantic_chunking.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `_split_text_semantic` + helpers**

Add near the top of `chunking.py` (after the existing stdlib imports):

```python
import logging

import numpy as np

logger = logging.getLogger(__name__)
```

Add these functions to `chunking.py`:

```python
def _document_sentences(text: str) -> list[str]:
    """Flatten a document into an ordered sentence list (paragraph then sentence)."""
    sentences: list[str] = []
    for para in _split_paragraphs(text):
        sentences.extend(_split_sentences_in_paragraph(para))
    return sentences


def _buffered_sentences(sentences: list[str], buffer_size: int) -> list[str]:
    """Combine each sentence with its neighbors to denoise the similarity signal."""
    if buffer_size <= 1:
        return sentences
    half = buffer_size - 1
    combined: list[str] = []
    for i in range(len(sentences)):
        lo = max(0, i - half)
        hi = min(len(sentences), i + half + 1)
        combined.append(" ".join(sentences[lo:hi]))
    return combined


def _adjacent_distances(vectors: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos) between each adjacent pair of row vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = vectors / safe
    sims = np.sum(unit[:-1] * unit[1:], axis=1)
    return 1.0 - sims


def _split_text_semantic(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_fn,
    percentile: float,
    buffer_size: int,
) -> list[str]:
    """Chunk at points where adjacent-sentence cosine distance spikes.

    Falls back to the paragraph splitter when embeddings are unavailable, the
    document has fewer than two sentences, or embedding fails.
    """
    if embedding_fn is None:
        return _split_text_paragraphs(text, chunk_size, chunk_overlap)

    sentences = _document_sentences(text)
    if len(sentences) < 2:
        return _split_text_paragraphs(text, chunk_size, chunk_overlap)

    try:
        vectors = np.asarray(embedding_fn(_buffered_sentences(sentences, buffer_size)))
        if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
            raise ValueError("embedding output shape does not match sentence count")
    except Exception as exc:  # noqa: BLE001 — degrade to structural chunking
        logger.warning("Semantic chunking embedding failed (%s); using paragraphs.", exc)
        return _split_text_paragraphs(text, chunk_size, chunk_overlap)

    distances = _adjacent_distances(vectors)
    threshold = float(np.percentile(distances, percentile))

    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold:
            groups.append(current)
            current = []
        current.append(sentences[i + 1])
    if current:
        groups.append(current)

    chunks: list[str] = []
    for group in groups:
        group_text = " ".join(group).strip()
        if not group_text:
            continue
        if _token_count(group_text) > chunk_size:
            chunks.extend(
                _split_text_paragraphs(group_text, chunk_size, chunk_overlap)
            )
        else:
            chunks.append(group_text)
    return [c for c in chunks if c]
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/document_index/test_semantic_chunking.py -q`
Expected: PASS (all 6). If `test_boundary_at_topic_shift` is off by the join spacing, confirm the deterministic embedder maps identical strings to identical vectors and distinct strings to a large distance; adjust the fixture sentences (not the algorithm) so the seam is the sole above-percentile gap.

- [ ] **Step 5: ruff + commit**

Run: `ruff check src/internal/document_index/chunking.py`

```bash
git add -A
git commit -m "feat(chunking): _split_text_semantic (percentile breakpoint + size cap + fallbacks)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire `embedding_fn` through chunk_document / chunk_documents / pipeline

**Files:**
- Modify: `src/internal/document_index/chunking.py` (`chunk_document`, `chunk_documents`, `_split_text` routing)
- Modify: `src/internal/document_index/pipeline.py` (pass `embedding_fn` to `chunk_documents`)
- Test: `tests/unit/document_index/test_semantic_chunking.py` (extend)

**Interfaces:**
- Consumes: `_split_text_semantic` (Task 2), `ChunkingConfig.semantic_*` (Task 1).
- Produces: `chunk_document(document, config, *, embedding_fn=None)`, `chunk_documents(documents, config, *, callback=None, embedding_fn=None)`.

- [ ] **Step 1: Write the failing tests**

```python
from src.internal.connectors.models import Document
from src.internal.document_index.models import ChunkingConfig


def _doc(text):
    return Document(id="d1", title="", contents=text, url=None, metadata={}, permissions=[])


def test_chunk_document_routes_to_semantic_when_enabled():
    text = (
        "cats cats cats. cats cats cats. cats cats cats. "
        "dogs dogs dogs. dogs dogs dogs."
    )
    cfg = ChunkingConfig(
        semantic_chunking=True, include_title=False, include_metadata=False
    )
    chunks = chunking.chunk_document(_doc(text), cfg, embedding_fn=EMB)
    assert len(chunks) == 2


def test_chunk_document_semantic_off_matches_today():
    text = "cats cats cats. dogs dogs dogs. birds birds birds."
    cfg = ChunkingConfig(include_title=False, include_metadata=False)  # off
    with_fn = chunking.chunk_document(_doc(text), cfg, embedding_fn=EMB)
    without_fn = chunking.chunk_document(_doc(text), cfg)
    assert [c.text for c in with_fn] == [c.text for c in without_fn]


def test_chunk_documents_threads_embedding_fn():
    text = (
        "cats cats cats. cats cats cats. cats cats cats. "
        "dogs dogs dogs. dogs dogs dogs."
    )
    cfg = ChunkingConfig(
        semantic_chunking=True, include_title=False, include_metadata=False
    )
    chunks = chunking.chunk_documents([_doc(text)], cfg, embedding_fn=EMB)
    assert len(chunks) == 2
```

(Verify the `Document` constructor signature by reading `connectors/models.py`; adjust the `_doc` factory to the real required fields.)

- [ ] **Step 2: Run — expect failure** (`chunk_document() got an unexpected keyword argument 'embedding_fn'`)

Run: `pytest tests/unit/document_index/test_semantic_chunking.py -q`
Expected: the 3 new tests FAIL.

- [ ] **Step 3: Implement the wiring**

In `chunking.py`, change the signatures and the split call:

```python
def chunk_document(
    document: Document,
    config: ChunkingConfig,
    *,
    embedding_fn=None,
) -> list[IndexChunk]:
```

Replace the `chunk_texts = _split_text(...)` line (currently `chunking.py:64`) with:

```python
    if config.semantic_chunking and embedding_fn is not None:
        chunk_texts = _split_text_semantic(
            text,
            content_token_limit,
            config.chunk_overlap,
            embedding_fn,
            config.semantic_breakpoint_percentile,
            config.semantic_buffer_size,
        )
    else:
        chunk_texts = _split_text(text, content_token_limit, config.chunk_overlap)
```

And thread it through `chunk_documents`:

```python
def chunk_documents(
    documents: Iterable[Document],
    config: ChunkingConfig,
    *,
    callback: IndexingHeartbeatInterface | None = None,
    embedding_fn=None,
) -> list[IndexChunk]:
    ...
        document_chunks = chunk_document(document, config, embedding_fn=embedding_fn)
```

- [ ] **Step 4: Pass the pipeline's embedder**

In `src/internal/document_index/pipeline.py`, change the `chunk_documents(...)` call (currently `pipeline.py:51`) to:

```python
    chunks = chunk_documents(
        indexable_docs,
        config.chunking,
        callback=callback,
        embedding_fn=embedding_fn,
    )
```

(`embedding_fn` is already a parameter of `run_indexing_pipeline`.)

- [ ] **Step 5: Run — expect pass + full suite**

Run: `pytest tests/unit/document_index/ -q && ruff check src/internal/document_index/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(chunking): thread embedding_fn through chunk_document(s) + pipeline

Semantic routing engages only when config.semantic_chunking and an embedding_fn is
supplied; otherwise byte-identical to today.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: End-to-end pipeline test + regression + final gate + PR

**Files:**
- Test: `tests/unit/document_index/test_semantic_chunking.py` (extend) or the existing pipeline test file (grep for `run_indexing_pipeline`)

- [ ] **Step 1: Write the end-to-end + flag-off regression tests**

```python
def test_pipeline_builds_index_with_semantic_chunking(tmp_path):
    from src.internal.document_index.pipeline import run_indexing_pipeline
    from src.internal.document_index.models import (
        IndexingPipelineConfig,
        ChunkingConfig,
    )
    # Build a minimal IndexingPipelineConfig with semantic_chunking=True and a
    # save_dir of tmp_path (read the config's real required fields first).
    # Assert the pipeline returns a result and writes corpus/embeddings without error.
    ...
```

(Read `IndexingPipelineConfig` and an existing pipeline test to copy the exact minimal construction + the `deterministic_embedding_fn` wiring; the assertion is that the run succeeds and produces chunks with `semantic_chunking=True`.)

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/unit/document_index/test_semantic_chunking.py -q`
Expected: PASS.

- [ ] **Step 3: Full gate**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 4: Confirm the flag-off invariant + diff shape**

Run: `git diff --stat main...HEAD`
Expected: `models.py`, `chunking.py`, `pipeline.py` modified; new test files added; spec + plan added. No unrelated files.

- [ ] **Step 5: Push + open PR**

```bash
git push -u origin feat/semantic-chunking
gh pr create --base main --title "feat: opt-in semantic chunking (embedding-similarity, percentile breakpoint)" --body "$(cat <<'EOF'
Adds semantic chunking — the third canonical chunking strategy — as an opt-in
`ChunkingConfig` mode, default-off (byte-identical when off).

`_split_text_semantic` embeds each sentence (via the pipeline's existing
`embedding_fn`), places a boundary wherever the adjacent-sentence cosine distance
exceeds this document's Nth-percentile breakpoint (default p95), caps each chunk at
`chunk_size` (re-splitting oversized semantic regions with the existing paragraph
splitter), and falls back to the structural splitter on any error / `<2` sentences
/ missing embedder — never raising.

- Config: `semantic_chunking` / `semantic_breakpoint_percentile` / `semantic_buffer_size`
- `chunking.py` stays independent of `embedding.py` (injected `Callable` + direct numpy)
- Cost tradeoff (documented): one extra sentence-embedding pass at index time, gated by the flag
- Tests use the deterministic embedder to assert mechanics (boundary at a controlled distance spike), plus every fallback and the flag-off invariant

Spec: `docs/superpowers/specs/2026-07-24-semantic-chunking-design.md`
Plan: `docs/superpowers/plans/2026-07-24-semantic-chunking.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:** config fields + validation (T1); the `_split_text_semantic` algorithm — percentile breakpoint, size cap, all fallbacks, buffer (T2); wiring through `chunk_document`/`chunk_documents`/`pipeline` + the semantic-off invariant (T3); end-to-end + flag-off regression + gate + PR (T4). Every spec success-criterion maps to a test.

**Placeholder scan:** the two `...` blocks (T4 end-to-end construction, and the `_doc` factory caveat) are explicitly "read the real signature first" instructions, not hidden logic — the pipeline/config constructors must be read from source rather than guessed. All algorithm code is given in full.

**Type consistency:** `_split_text_semantic`'s parameter order `(text, chunk_size, chunk_overlap, embedding_fn, percentile, buffer_size)` is identical in the Task 2 test helper, the implementation, and the Task 3 `chunk_document` call site. Config field names (`semantic_chunking`, `semantic_breakpoint_percentile`, `semantic_buffer_size`) match across T1/T3.
