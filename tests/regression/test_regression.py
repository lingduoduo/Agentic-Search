"""Regression tests: pin observable behaviour so unintended changes surface early.

Each test locks in a specific, computed output for a known input.  If the
output changes the test fails — prompting deliberate review before merging.
"""

import pytest

from src.backend.document_index.index_builder import (
    prepare_texts,
    resolve_pooling_method,
)
from src.backend.servers.rerank import passage_to_string, string_to_document
from src.backend.servers.app import format_document
from src.backend.document_index.text import (
    extract_keywords,
    normalize_text,
    tokenize_text,
)


# ---------------------------------------------------------------------------
# normalize_text — character-level output snapshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", "hello world"),
        ("  leading   trailing  ", "leading trailing"),
        ("café", "cafe"),  # NFD decomposition + ASCII encode drops accent
        ("it's a test", "it s a test"),  # apostrophe removed
        ("", ""),
        ("123 456", "123 456"),  # digits kept
        ("abc...def", "abc def"),  # ellipsis becomes spaces, collapsed to one
        ("UPPER CASE", "upper case"),
        ("tab\there", "tab here"),  # tabs treated as whitespace
    ],
)
def test_normalize_text_snapshot(text, expected):
    assert normalize_text(text) == expected


def test_normalize_text_keep_non_ascii_preserves_accented_char():
    assert "é" in normalize_text("café", keep_non_ascii=True)


# ---------------------------------------------------------------------------
# tokenize_text — token-list snapshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, max_length, expected",
    [
        ("a b c d e", 3, ["a", "b", "c"]),
        ("Hello, World!", None, ["hello", "world"]),
        ("  spaces  ", None, ["spaces"]),
        ("", None, []),
        ("one", None, ["one"]),
        ("a b c", 10, ["a", "b", "c"]),  # limit larger than token count
    ],
)
def test_tokenize_text_snapshot(text, max_length, expected):
    assert tokenize_text(text, max_length=max_length) == expected


# ---------------------------------------------------------------------------
# extract_keywords — ordering and boundary behaviour
# ---------------------------------------------------------------------------


def test_extract_keywords_frequency_order_stable():
    text = "cat cat cat dog dog bird"
    keywords = extract_keywords(text)
    assert keywords[0] == "cat"
    assert keywords[1] == "dog"
    assert keywords[2] == "bird"


def test_extract_keywords_limit_zero_always_empty():
    assert extract_keywords("lots of words here", limit=0) == []


def test_extract_keywords_limit_negative_always_empty():
    assert extract_keywords("lots of words here", limit=-5) == []


def test_extract_keywords_single_word():
    assert extract_keywords("word", limit=5) == ["word"]


# ---------------------------------------------------------------------------
# prepare_texts — prefix format snapshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, is_query, text, expected",
    [
        ("e5-small", False, "doc text", "passage: doc text"),
        ("e5-small", True, "query text", "query: query text"),
        ("bge-m3", False, "document", "document"),  # passage gets no prefix
        ("contriever", True, "find me", "find me"),
        ("dpr", False, "a doc", "a doc"),
    ],
)
def test_prepare_texts_prefix_snapshot(method, is_query, text, expected):
    result = prepare_texts([text], method, is_query=is_query)
    assert result == [expected]


def test_prepare_texts_bge_query_prefix_contains_instruction():
    result = prepare_texts(["search this"], "bge", is_query=True)
    assert result[0].startswith(
        "Represent this sentence for searching relevant passages:"
    )
    assert "search this" in result[0]


# ---------------------------------------------------------------------------
# resolve_pooling_method — model-key → pooling mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, expected",
    [
        ("e5-large", "mean"),
        ("e5-small-v2", "mean"),
        ("bge-base-en", "cls"),
        ("bge-m3", "cls"),
        ("contriever", "mean"),
        ("jina-embeddings-v2-base-en", "mean"),
        ("bert-base-uncased", "mean"),  # unknown → default mean
        ("roberta-large", "mean"),  # unknown → default mean
    ],
)
def test_resolve_pooling_method_model_key_snapshot(method, expected):
    assert resolve_pooling_method(method, None) == expected


@pytest.mark.parametrize("explicit", ["mean", "cls", "pooler"])
def test_resolve_pooling_method_explicit_passthrough(explicit):
    assert resolve_pooling_method("any-model", explicit) == explicit


# ---------------------------------------------------------------------------
# passage_to_string — output format snapshots
# ---------------------------------------------------------------------------


def test_passage_to_string_quoted_title_format():
    doc = {"document": {"contents": '"My Title"\nBody text here'}}
    assert passage_to_string(doc) == "(Title: My Title) Body text here"


def test_passage_to_string_unquoted_title_format():
    doc = {"document": {"contents": "Unquoted Title\nBody text"}}
    assert passage_to_string(doc) == "(Title: Unquoted Title) Body text"


def test_passage_to_string_no_body():
    doc = {"document": {"contents": "Title only"}}
    result = passage_to_string(doc)
    assert result.startswith("(Title: Title only)")


def test_passage_to_string_flat_contents_key():
    doc = {"contents": "Header\nContent"}
    result = passage_to_string(doc)
    assert "(Title: Header)" in result


# ---------------------------------------------------------------------------
# string_to_document — inverse of passage_to_string
# ---------------------------------------------------------------------------


def test_string_to_document_with_title_format():
    result = string_to_document("(Title: My Title) Body content")
    assert result == {"document": {"contents": '"My Title"\nBody content'}}


def test_string_to_document_plain_text_fallback():
    result = string_to_document("no title here")
    assert result == {"document": {"contents": "no title here"}}


# ---------------------------------------------------------------------------
# format_document — output structure snapshot
# ---------------------------------------------------------------------------


def test_format_document_full_output_snapshot():
    result = format_document("My Title", "My Content")
    assert result == {
        "document": {"title": "My Title", "contents": '"My Title"\nMy Content'}
    }


def test_format_document_none_title_snapshot():
    result = format_document(None, "content")
    assert result == {
        "document": {"title": "No title.", "contents": '"No title."\ncontent'}
    }


def test_format_document_none_content_snapshot():
    result = format_document("title", None)
    assert result == {
        "document": {"title": "title", "contents": '"title"\nNo snippet available.'}
    }


def test_format_document_both_none_snapshot():
    result = format_document(None, None)
    assert result == {
        "document": {
            "title": "No title.",
            "contents": '"No title."\nNo snippet available.',
        }
    }
