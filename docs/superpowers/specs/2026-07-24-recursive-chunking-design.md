# Recursive (structure-aware) chunking — opt-in

**Date:** 2026-07-24
**Branch:** `feat/recursive-chunking` (off `main`)
**Status:** design approved, pending spec review

## Context

The repo's default splitter (`_split_text_paragraphs`) is a fixed 2-level
paragraph→sentence greedy packer. It covers the *spirit* of structure-aware
chunking for prose but is not a true recursive splitter: no configurable separator
hierarchy, no Markdown heading nesting, and — the biggest practical gap — no
protection for fenced code blocks or Markdown tables (a blank line inside a code
block splits it). Semantic chunking shipped as an opt-in mode; this adds the third
missing piece, **recursive (structure-aware) chunking**, the same way.

## Goal

Add a proper recursive, structure-aware chunker that splits along a coarse→fine
Markdown separator hierarchy, falling to finer units only when a piece still
exceeds `chunk_size`, and never splits a fenced code block or Markdown table
internally. Opt-in behind a `ChunkingConfig` flag; default-off ⇒ byte-identical to
today.

## Scope decisions (approved)

- **Opt-in**, not a replacement for the default (zero risk to existing indexes).
- **Markdown + code/table integrity**; **no HTML** (out of scope this version).
- Oversized *atomic* block (a lone code block / table larger than `chunk_size`) is
  **kept whole** — integrity is absolute; a large code block or table becomes one
  oversized chunk (the embedder truncates it) rather than a fragmented block with a
  dangling fence or orphaned rows. (Decided during implementation: token-windowing
  such a block reintroduces exactly the mid-block splitting this feature exists to
  prevent, for its highest-value case.)
- Separator hierarchy is **hardcoded** (a sensible Markdown default), not a config
  field (YAGNI).

## Design

### Config (`ChunkingConfig`, `models.py`)

- Add `recursive_chunking: bool = False`.
- `validate()`: raise if `recursive_chunking and semantic_chunking` are both True —
  they are alternative top-level strategies, not composable.

### Self-containment (import-cycle avoidance)

`chunking.py` must NOT import from `natural_language_processing` — NLP already
imports `document_index`, so the reverse would create a cycle. The recursive
splitter reuses chunking's own `_token_count` (whitespace tokens) and `_overlap_tail`,
consistent with the existing splitters. (Documented because it constrains the
implementation.)

### Algorithm — `_split_text_recursive(text, chunk_size, chunk_overlap)`

**Step 1 — segment into atomic blocks + prose spans.** A single left-to-right scan
partitions the document into an ordered list of segments, each tagged `atomic` or
`prose`:
- **Fenced code block:** a line whose stripped text starts with ` ``` ` or `~~~`
  opens a fence; everything through the matching closing fence line is one `atomic`
  segment. An unterminated fence is `atomic` to end-of-document (conservative).
- **Markdown table:** a run of consecutive lines that are table rows — a header
  line containing `|` immediately followed by a separator row matching
  `^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$` (the `|---|---|` delimiter) —
  plus the following contiguous `|`-containing rows, is one `atomic` segment.
- Everything else accumulates into `prose` segments (the text between atomic
  blocks), split back out in document order.

**Step 2 — recursively split each prose segment** with an ordered separator list,
coarse→fine:
```
["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]
```
`_recursive_split(text, separators)`:
1. Pick the first separator that occurs in `text` (else the last, `" "`); split on it
   (keeping heading markers attached to their section — split on the newline before
   the marker, mirroring the current `\n(?=#)` behavior).
2. For each piece: if `_token_count(piece) <= chunk_size`, keep it; else recurse
   with the remaining (finer) separators.
3. At the finest separator (`" "`) a piece is a single whitespace token and is kept
   as-is (under the whitespace-token model a lone token is count 1, so this terminates
   without further splitting).

**Step 3 — atomic segments pass through whole**, regardless of size (see the Scope
decision above: integrity is absolute, an oversized code/table becomes one chunk).

**Step 4 — merge + overlap.** Greedily merge the ordered `(is_atomic, text)` pieces
(prose sub-pieces and atomic blocks, in document order) into chunks up to
`chunk_size`, carrying `chunk_overlap` tokens between emitted chunks via
`_overlap_tail` **computed from trailing prose pieces only** — an atomic block is
never sliced into an overlap tail (that would leak a fence/row fragment into the
next chunk). An atomic block
is never merged *into* in a way that would split it; if adding it would exceed
`chunk_size`, flush first and emit the atomic block as its own chunk.

### Integration

`chunk_document` routes to `_split_text_recursive` when `config.recursive_chunking`
(a branch parallel to the existing semantic branch), before the default
`_split_text`. No `embedding_fn` needed. `chunk_documents`/`pipeline` need no new
parameter (recursive is purely lexical).

## Verification / success criteria

1. `recursive_chunking=False` (default) ⇒ `chunk_document` output byte-identical to
   today.
2. **Heading hierarchy:** a doc with `# A … ## B … ## C` splits at heading
   boundaries; higher-level headings are preferred split points.
3. **Code-block integrity:** a fenced block containing blank lines and lines that
   look like headings (`# not a heading`) stays in exactly one chunk — not split at
   the internal `\n\n` or `#`.
4. **Table integrity:** a Markdown table's rows stay together in one chunk.
5. **Recursion fallback:** a heading-less paragraph exceeding `chunk_size` descends
   to sentence then word splitting; no emitted *prose* chunk exceeds `chunk_size`.
   (An atomic code/table block larger than `chunk_size` is the sole exception — kept
   whole by design.)
6. **Merge + overlap:** small adjacent sections merge up to `chunk_size` with the
   configured overlap; overlap is carried from trailing prose only, so an atomic
   block is never fragmented across chunks even when `chunk_overlap > 0`.
7. `validate()` raises when `recursive_chunking and semantic_chunking`.
8. `chunking.py` imports nothing from `natural_language_processing`.
9. `ruff check .` + `pytest` green.

## Risks / tradeoffs

- **Heuristic structure detection.** Code-fence and table regexes are pragmatic, not
  a full Markdown parser: unusual table formats or indented/tilde fences may be
  missed. The failure mode is graceful — a missed block is chunked as prose (today's
  behavior), never a crash. Tests pin the common cases.
- **Token unit stays whitespace-words** (`_token_count`), consistent with the rest of
  chunking and required to avoid the NLP import cycle; `chunk_size` remains ~words,
  not BPE tokens (already documented).
- Purely additive and opt-in: default indexing is unchanged.

## Out of scope

- HTML structure awareness.
- Replacing / upgrading the default splitter.
- Composing recursive + semantic in one pass.
- Making the separator list configurable.
