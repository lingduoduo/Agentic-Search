from src.internal.connectors.models import Document
from src.internal.document_index import chunking
from src.internal.document_index.models import ChunkingConfig


def _doc(text):
    return Document(
        id="d1", title="", contents=text, url=None, metadata={}, permissions=[]
    )


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


def test_short_text_returns_whole():
    assert chunking._recursive_split(
        "small text", chunking._RECURSIVE_SEPARATORS, 900, 0
    ) == ["small text"]


def test_splits_at_heading_boundaries_first():
    text = (
        "# Title\nintro line\n\n## Section A\naaa aaa aaa\n\n## Section B\nbbb bbb bbb"
    )
    pieces = chunking._recursive_split(text, chunking._RECURSIVE_SEPARATORS, 6, 0)
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
    rows = "\n".join(f"| {i} | {i * 2} |" for i in range(15))
    text = f"Before.\n\n| a | b |\n| --- | --- |\n{rows}\n\nAfter."
    chunks = chunking._split_text_recursive(text, chunk_size=8, chunk_overlap=0)
    holders = [c for c in chunks if "| --- | --- |" in c]
    assert len(holders) == 1
    assert "| 0 | 0 |" in holders[0] and "| 14 | 28 |" in holders[0]


def test_atomic_block_intact_with_overlap():
    code = "```\n" + "\n".join(f"line{i} = {i}" for i in range(20)) + "\n```"
    text = f"Intro prose here.\n\n{code}\n\nOutro prose one. Outro prose two."
    chunks = chunking._split_text_recursive(text, chunk_size=8, chunk_overlap=4)
    fence_chunks = [c for c in chunks if "```" in c]
    assert len(fence_chunks) == 1
    assert fence_chunks[0].count("```") == 2
    for c in chunks:
        if c is not fence_chunks[0]:
            assert "```" not in c
            assert "line19 = 19" not in c


def test_chunk_document_routes_to_recursive_when_enabled():
    text = "# A\naaa\n\n## B\nbbb\n\n## C\nccc"
    cfg = ChunkingConfig(
        recursive_chunking=True,
        include_title=False,
        include_metadata=False,
        chunk_size=4,
        chunk_overlap=0,
    )
    chunks = chunking.chunk_document(_doc(text), cfg)
    assert len(chunks) >= 2  # split along headings


def test_recursive_off_matches_today():
    text = "one one one. two two two. three three three."
    cfg = ChunkingConfig(include_title=False, include_metadata=False)  # off
    got = [c.text for c in chunking.chunk_document(_doc(text), cfg)]
    expected_texts = chunking._split_text(text, cfg.chunk_size, cfg.chunk_overlap)
    assert got == expected_texts
