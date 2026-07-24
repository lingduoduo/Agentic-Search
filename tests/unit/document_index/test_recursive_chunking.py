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
