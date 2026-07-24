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


def test_short_text_returns_whole():
    assert chunking._recursive_split(
        "small text", chunking._RECURSIVE_SEPARATORS, 900, 0
    ) == ["small text"]


def test_splits_at_heading_boundaries_first():
    text = (
        "# Title\nintro line\n\n## Section A\naaa aaa aaa\n\n## Section B\nbbb bbb bbb"
    )
    pieces = chunking._recursive_split(text, chunking._RECURSIVE_SEPARATORS, 6, 0)
    joined = "\n".join(pieces)  # noqa: F841 — unused; kept per brief for readability
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
