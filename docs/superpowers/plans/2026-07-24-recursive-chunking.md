# Recursive (Structure-Aware) Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in recursive structure-aware chunking to `document_index/chunking.py` — coarse→fine Markdown separator hierarchy with fenced-code-block and table integrity — default-off.

**Architecture:** Pure-logic TDD, built bottom-up: `_segment_blocks` (atomic code/table detection) → `_recursive_split` (separator-hierarchy recursion for prose) → `_split_text_recursive` (orchestrate + merge/overlap) → route in `chunk_document`. Self-contained in `chunking.py` — no `natural_language_processing` import (cycle avoidance); reuses `_token_count`, `_split_token_window`, `_overlap_tail`. Spec: `docs/superpowers/specs/2026-07-24-recursive-chunking-design.md`.

**Tech Stack:** Python (`re`), pytest, ruff.

## Global Constraints

- Branch: `feat/recursive-chunking`, off `main`. Never commit to `main`.
- `recursive_chunking=False` (default) ⇒ byte-identical output to today.
- `chunking.py` must NOT import from `src.internal.natural_language_processing` (would create an import cycle — NLP already imports `document_index`).
- Reuse existing chunking helpers (`_token_count`, `_split_token_window`, `_overlap_tail`, `re`); do not add new deps.
- `ruff check .` + `pytest` green at the end of every task.

> **AS-SHIPPED CORRECTIONS (discovered during Task 4 + review — this plan's Task 4
> snippets below are the pre-pivot design; the shipped code differs):**
> 1. **Atomic blocks are kept WHOLE, never token-windowed.** The Task 4 snippet that
>    token-windows an oversized atomic segment (and the PR-body draft's "token-windowed
>    as a last resort") was dropped: an oversized code block / table larger than
>    `chunk_size` becomes one oversized chunk. Token-windowing it would re-introduce
>    the mid-block splitting this feature exists to prevent. `_split_text_recursive`
>    appends every atomic segment whole.
> 2. **Merge carries overlap from trailing PROSE pieces only.** A confirmed bug (fix
>    commit `94914c7`) was that `_overlap_tail` sliced the tail of an oversized atomic
>    block into the next chunk when `chunk_overlap > 0` (production default 120).
>    `_merge_recursive_pieces` now takes `list[tuple[bool, str]]` (`is_atomic`, text)
>    and builds the overlap tail by walking from the end and stopping at the first
>    atomic piece — so an atomic block is never fragmented across chunks. See the
>    reconciled spec for the authoritative behavior.

---

### Task 1: Config field + mutual-exclusivity validation

**Files:**
- Modify: `src/internal/document_index/models.py` (`ChunkingConfig`)
- Test: `tests/unit/document_index/test_chunking_config.py` (extend — it exists)

- [ ] **Step 1: Write the failing tests**

```python
def test_recursive_default_off():
    c = ChunkingConfig()
    assert c.recursive_chunking is False
    c.validate()


def test_recursive_and_semantic_are_mutually_exclusive():
    with pytest.raises(ValueError):
        ChunkingConfig(recursive_chunking=True, semantic_chunking=True).validate()


def test_recursive_alone_is_valid():
    ChunkingConfig(recursive_chunking=True).validate()
```

- [ ] **Step 2: Run — expect fail** (`AttributeError`)

Run: `pytest tests/unit/document_index/test_chunking_config.py -q`

- [ ] **Step 3: Implement**

In `ChunkingConfig`, after `semantic_buffer_size`:

```python
    recursive_chunking: bool = False
```

In `validate()`, after the semantic checks:

```python
        if self.recursive_chunking and self.semantic_chunking:
            raise ValueError(
                "recursive_chunking and semantic_chunking are mutually exclusive."
            )
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/document_index/test_chunking_config.py -q && ruff check src/internal/document_index/models.py`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(chunking): add recursive_chunking config flag (excl. with semantic)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_segment_blocks` — atomic code/table detection

**Files:**
- Modify: `src/internal/document_index/chunking.py`
- Test: `tests/unit/document_index/test_recursive_chunking.py` (create)

**Interfaces:**
- Produces: `_segment_blocks(text: str) -> list[tuple[str, str]]` — ordered `("atomic"|"prose", segment_text)`.

- [ ] **Step 1: Write the failing tests**

```python
from src.internal.document_index import chunking


def test_code_fence_is_one_atomic_segment():
    text = "Intro para.\n\n```python\nx = 1\n\n# not a heading\ny = 2\n```\n\nOutro."
    segs = chunking._segment_blocks(text)
    kinds = [k for k, _ in segs]
    assert kinds == ["prose", "atomic", "prose"]
    atomic = [s for k, s in segs if k == "atomic"][0]
    assert "x = 1" in atomic and "# not a heading" in atomic and "y = 2" in atomic


def test_unterminated_fence_is_atomic_to_eof():
    text = "Para.\n\n```\nunclosed code\nmore code"
    segs = chunking._segment_blocks(text)
    assert segs[-1][0] == "atomic"
    assert "unclosed code" in segs[-1][1] and "more code" in segs[-1][1]


def test_markdown_table_is_atomic():
    text = "Before.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n\nAfter."
    segs = chunking._segment_blocks(text)
    kinds = [k for k, _ in segs]
    assert kinds == ["prose", "atomic", "prose"]
    table = [s for k, s in segs if k == "atomic"][0]
    assert "| 1 | 2 |" in table and "| 3 | 4 |" in table


def test_plain_prose_is_single_prose_segment():
    text = "Just one paragraph.\n\nAnd another."
    segs = chunking._segment_blocks(text)
    assert [k for k, _ in segs] == ["prose"]
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/unit/document_index/test_recursive_chunking.py -q`

- [ ] **Step 3: Implement**

Add module-level constants near the top of `chunking.py` (after existing constants):

```python
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")
```

Add:

```python
def _segment_blocks(text: str) -> list[tuple[str, str]]:
    """Partition text into ordered ("atomic"|"prose", segment) parts.

    Fenced code blocks and Markdown tables are "atomic" (never split internally);
    the text between them is "prose". A missed/ambiguous block simply stays prose.
    """
    lines = text.split("\n")
    segments: list[tuple[str, str]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            joined = "\n".join(prose).strip()
            if joined:
                segments.append(("prose", joined))
            prose.clear()

    i, n = 0, len(lines)
    while i < n:
        fence = _FENCE_RE.match(lines[i])
        if fence:
            marker = fence.group(1)
            flush_prose()
            block = [lines[i]]
            i += 1
            while i < n:
                block.append(lines[i])
                closed = lines[i].strip().startswith(marker)
                i += 1
                if closed:
                    break
            segments.append(("atomic", "\n".join(block).strip()))
            continue
        if "|" in lines[i] and i + 1 < n and _TABLE_DELIM_RE.match(lines[i + 1]):
            flush_prose()
            block = [lines[i], lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                block.append(lines[i])
                i += 1
            segments.append(("atomic", "\n".join(block).strip()))
            continue
        prose.append(lines[i])
        i += 1

    flush_prose()
    return segments
```

- [ ] **Step 4: Run — expect pass + ruff**

Run: `pytest tests/unit/document_index/test_recursive_chunking.py -q && ruff check src/internal/document_index/chunking.py`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(chunking): _segment_blocks — atomic code-fence + table detection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `_recursive_split` — separator-hierarchy recursion for prose

**Files:**
- Modify: `src/internal/document_index/chunking.py`
- Test: `tests/unit/document_index/test_recursive_chunking.py` (extend)

**Interfaces:**
- Consumes: `_token_count` (existing).
- Produces: `_split_on_separator(text, sep) -> list[str]`; `_recursive_split(text, separators, chunk_size, chunk_overlap) -> list[str]`; `_RECURSIVE_SEPARATORS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_short_text_returns_whole():
    assert chunking._recursive_split(
        "small text", chunking._RECURSIVE_SEPARATORS, 900, 0
    ) == ["small text"]


def test_splits_at_heading_boundaries_first():
    text = (
        "# Title\nintro line\n\n## Section A\naaa aaa aaa\n\n## Section B\nbbb bbb bbb"
    )
    pieces = chunking._recursive_split(text, chunking._RECURSIVE_SEPARATORS, 6, 0)
    joined = "\n".join(pieces)
    # each H2 marker stays attached to its section's content
    a = next(p for p in pieces if "Section A" in p)
    b = next(p for p in pieces if "Section B" in p)
    assert "aaa" in a and "bbb" not in a
    assert "bbb" in b and "aaa" not in b


def test_recurses_to_words_and_never_exceeds_size():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    pieces = chunking._recursive_split(text, chunking._RECURSIVE_SEPARATORS, 3, 0)
    assert pieces
    for p in pieces:
        assert chunking._token_count(p) <= 3


def test_spaceless_blob_terminates_as_one_piece():
    # a single whitespace-token (no separators present) is one token -> kept as-is;
    # this guards against infinite recursion at the finest separator.
    text = "x" * 50
    pieces = chunking._recursive_split(text, chunking._RECURSIVE_SEPARATORS, 2, 0)
    assert pieces == [text]
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/unit/document_index/test_recursive_chunking.py -q`

- [ ] **Step 3: Implement**

Add:

```python
# coarse -> fine; heading levels keep their marker attached to the section below.
_RECURSIVE_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, sep: str) -> list[str]:
    if sep.startswith("\n#"):
        marker = sep[1:]  # e.g. "# "
        return re.split(rf"\n(?={re.escape(marker)})", text)
    if sep == ". ":
        return re.split(r"(?<=[.!?。！？])\s+", text)
    return text.split(sep)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split prose along a coarse->fine separator list, recursing only when a piece
    still exceeds chunk_size; the finest level falls back to a token window."""
    text = text.strip()
    if not text:
        return []
    if _token_count(text) <= chunk_size:
        return [text]

    sep = separators[-1]
    rest: list[str] = []
    for idx, candidate in enumerate(separators):
        if candidate in text:
            sep = candidate
            rest = separators[idx + 1 :]
            break

    out: list[str] = []
    for piece in _split_on_separator(text, sep):
        piece = piece.strip()
        if not piece:
            continue
        if _token_count(piece) <= chunk_size or not rest:
            # kept when it fits, or when no finer separator remains (a single
            # whitespace-token cannot be reduced further — the atomic-block path
            # in _split_text_recursive is where genuinely oversized spans are
            # token-windowed).
            out.append(piece)
        else:
            out.extend(_recursive_split(piece, rest, chunk_size, chunk_overlap))
    return out
```

> Note: `_split_token_window` is deliberately NOT used here — under the whitespace
> token model a piece with no remaining separator is a single token (count 1), so
> the finest level always terminates by appending. Oversized *atomic* blocks are the
> only real oversize case and are token-windowed in `_split_text_recursive` (Task 4).

- [ ] **Step 4: Run — expect pass + ruff**

Run: `pytest tests/unit/document_index/test_recursive_chunking.py -q && ruff check src/internal/document_index/chunking.py`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(chunking): _recursive_split — coarse->fine separator recursion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `_split_text_recursive` + merge, and route in `chunk_document`

**Files:**
- Modify: `src/internal/document_index/chunking.py`
- Test: `tests/unit/document_index/test_recursive_chunking.py` (extend)

**Interfaces:**
- Consumes: `_segment_blocks` (T2), `_recursive_split` (T3), `_token_count`, `_split_token_window`, `_overlap_tail`.
- Produces: `_split_text_recursive(text, chunk_size, chunk_overlap) -> list[str]`; routing in `chunk_document`.

- [ ] **Step 1: Write the failing tests**

```python
from src.internal.connectors.models import Document
from src.internal.document_index.models import ChunkingConfig


def _doc(text):
    return Document(
        id="d1", title="", contents=text, url=None, metadata={}, permissions=[]
    )


def test_code_block_never_split_across_chunks():
    code = "```\n" + "\n".join(f"line{i} = {i}" for i in range(20)) + "\n```"
    text = f"Intro.\n\n{code}\n\nOutro."
    chunks = chunking._split_text_recursive(text, chunk_size=8, chunk_overlap=0)
    # the whole fenced block lands inside exactly one chunk (opening + closing fence together)
    holders = [c for c in chunks if "```" in c]
    assert len(holders) == 1
    assert holders[0].count("```") == 2
    assert "line0 = 0" in holders[0] and "line19 = 19" in holders[0]


def test_table_never_split_across_chunks():
    rows = "\n".join(f"| {i} | {i*2} |" for i in range(15))
    text = f"Before.\n\n| a | b |\n| --- | --- |\n{rows}\n\nAfter."
    chunks = chunking._split_text_recursive(text, chunk_size=8, chunk_overlap=0)
    holders = [c for c in chunks if "| --- | --- |" in c]
    assert len(holders) == 1
    assert "| 0 | 0 |" in holders[0] and "| 14 | 28 |" in holders[0]


def test_chunk_document_routes_to_recursive_when_enabled():
    text = "# A\naaa\n\n## B\nbbb\n\n## C\nccc"
    cfg = ChunkingConfig(
        recursive_chunking=True, include_title=False, include_metadata=False, chunk_size=4
    )
    chunks = chunking.chunk_document(_doc(text), cfg)
    assert len(chunks) >= 2  # split along headings


def test_recursive_off_matches_today():
    text = "one one one. two two two. three three three."
    cfg = ChunkingConfig(include_title=False, include_metadata=False)  # off
    got = [c.text for c in chunking.chunk_document(_doc(text), cfg)]
    expected_texts = chunking._split_text(text, cfg.chunk_size, cfg.chunk_overlap)
    assert got == expected_texts
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/unit/document_index/test_recursive_chunking.py -q`

- [ ] **Step 3: Implement `_split_text_recursive` + `_merge_recursive_pieces`**

```python
def _merge_recursive_pieces(
    pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Greedily merge ordered pieces up to chunk_size, carrying overlap. Pieces are
    never split here, so atomic blocks stay intact."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        piece_tokens = _token_count(piece)
        if current and current_tokens + piece_tokens > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = list(_overlap_tail(current, chunk_overlap))
            current_tokens = _token_count("\n\n".join(current)) if current else 0
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        chunks.append("\n\n".join(current).strip())
    return [c for c in chunks if c]


def _split_text_recursive(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Structure-aware recursive chunking: keep code blocks/tables intact, split
    prose along a coarse->fine Markdown separator hierarchy, then merge + overlap."""
    pieces: list[str] = []
    for kind, seg in _segment_blocks(text):
        if kind == "atomic":
            if _token_count(seg) > chunk_size:
                pieces.extend(_split_token_window(seg, chunk_size, chunk_overlap))
            else:
                pieces.append(seg)
        else:
            pieces.extend(
                _recursive_split(seg, _RECURSIVE_SEPARATORS, chunk_size, chunk_overlap)
            )
    return _merge_recursive_pieces(pieces, chunk_size, chunk_overlap)
```

- [ ] **Step 4: Route in `chunk_document`**

Find the split-selection block in `chunk_document` (the `if config.semantic_chunking and embedding_fn is not None:` branch). Add a recursive branch BEFORE it so the selection reads:

```python
    if config.recursive_chunking:
        chunk_texts = _split_text_recursive(
            text, content_token_limit, config.chunk_overlap
        )
    elif config.semantic_chunking and embedding_fn is not None:
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

- [ ] **Step 5: Run — expect pass + full document_index suite**

Run: `pytest tests/unit/document_index/ -q && ruff check src/internal/document_index/`
Expected: PASS (incl. the code/table integrity + heading-routing + off-matches-today tests).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(chunking): _split_text_recursive + chunk_document routing

Opt-in structure-aware recursive chunking: atomic code/table blocks stay intact,
prose recurses coarse->fine, pieces merge with overlap. Off by default -> unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Full gate + push + open PR

**Files:** none (verification + integration).

- [ ] **Step 1: Full gate**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 2: Confirm no NLP import + flag-off invariant + diff shape**

Run: `grep -n "natural_language_processing" src/internal/document_index/chunking.py`
Expected: no output.
Run: `git diff --stat main...HEAD`
Expected: `models.py`, `chunking.py` modified; `test_chunking_config.py` + new `test_recursive_chunking.py`; spec + plan added. No unrelated files; `pipeline.py` NOT changed (recursive needs no embedding_fn).

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin feat/recursive-chunking
gh pr create --base main --title "feat: opt-in recursive (structure-aware) chunking" --body "$(cat <<'EOF'
Adds **recursive (structure-aware) chunking** — the canonical method that splits along a coarse→fine separator hierarchy and falls to finer units only when a piece still exceeds the size limit — as an **opt-in** `ChunkingConfig` mode, default-off (byte-identical when off).

`_split_text_recursive` (in `document_index/chunking.py`):
1. `_segment_blocks` marks fenced code blocks and Markdown tables as **atomic** — never split internally (a lone block larger than `chunk_size` is token-windowed as a last resort)
2. `_recursive_split` splits prose along `["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]`, recursing to the next-finer separator only for pieces still over `chunk_size`
3. `_merge_recursive_pieces` greedily merges pieces up to `chunk_size` with overlap, never splitting an atomic block

**Config:** `recursive_chunking: bool = False`, mutually exclusive with `semantic_chunking`.

**Design:** opt-in/default-off; `chunking.py` stays free of any `natural_language_processing` import (that direction would cycle — NLP already imports `document_index`), reusing the existing `_token_count`/`_split_token_window`/`_overlap_tail`.

**Tests:** code-block integrity (fence with internal blank lines + fake headings stays whole), table integrity, heading-hierarchy splitting, recursion-to-word fallback with a hard size cap, merge+overlap, recursive-off-equals-today, mutual-exclusivity validation, and `_segment_blocks` unit coverage.

Scope: Markdown + code/table; no HTML; default splitter unchanged.

Spec: `docs/superpowers/specs/2026-07-24-recursive-chunking-design.md`
Plan: `docs/superpowers/plans/2026-07-24-recursive-chunking.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(If `gh pr create` fails with a GitHub GraphQL/5xx error, the branch is still pushed — retry until it succeeds.)

---

## Self-Review

**Spec coverage:** config + mutual exclusivity (T1); atomic code/table detection (T2); coarse→fine recursion with size-capped fallback (T3); orchestration + merge/overlap + `chunk_document` routing + flag-off invariant (T4); gate + no-NLP-import check + PR (T5). Every spec success-criterion maps to a test (heading hierarchy, code integrity, table integrity, recursion fallback, merge+overlap, off-equals-today, mutual-exclusivity, no NLP import).

**Placeholder scan:** no vague steps — every function is given in full; test bodies are complete. Routing edit shows the exact final if/elif/else block.

**Type consistency:** `_segment_blocks -> list[tuple[str,str]]` consumed by `_split_text_recursive`; `_recursive_split(text, separators, chunk_size, chunk_overlap)` signature identical across T3 tests, its impl, and the T4 call site; `_RECURSIVE_SEPARATORS` referenced consistently. `recursive_chunking` field name matches across T1/T4. Routing precedence (recursive → semantic → default) is explicit and each branch passes `content_token_limit` (not raw `chunk_size`), matching the existing semantic branch.
