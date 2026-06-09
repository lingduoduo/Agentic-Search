"""Unit tests for src.backend.servers.rerank."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.backend.servers.rerank import (
    RerankerConfig,
    SentenceTransformerReranker,
    passage_to_string,
    string_to_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(contents: str) -> dict:
    return {"document": {"contents": contents}}


def _make_mock_torch() -> MagicMock:
    """Return a minimal torch mock that satisfies isinstance checks in _predict."""
    mock_torch = MagicMock()
    # Make torch.Tensor a distinct dummy type so numpy arrays don't match it
    mock_torch.Tensor = type("_MockTensor", (), {})
    return mock_torch


def _make_reranker(scores: list[float]) -> SentenceTransformerReranker:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(scores, dtype=np.float32)
    return SentenceTransformerReranker(model=mock_model, batch_size=32, device="cpu")


@pytest.fixture(autouse=True)
def _patch_rerank_torch(monkeypatch):
    """Patch _require_torch inside rerank so tests don't need a real torch install."""
    mock_torch = _make_mock_torch()
    monkeypatch.setattr("src.backend.servers.rerank._require_torch", lambda: mock_torch)


# ---------------------------------------------------------------------------
# passage_to_string
# ---------------------------------------------------------------------------


class TestPassageToString:
    def test_nested_document_title_and_body(self):
        doc = _doc('"My Title"\nBody text here')
        result = passage_to_string(doc)
        assert result == "(Title: My Title) Body text here"

    def test_nested_document_unquoted_title_and_body(self):
        doc = _doc("My Title\nBody text here")
        result = passage_to_string(doc)
        assert result == "(Title: My Title) Body text here"

    def test_nested_document_no_newline_title_only(self):
        doc = _doc("Just a single line")
        result = passage_to_string(doc)
        # title="Just a single line", body="" → clean_title truthy → "(Title: ...) "
        assert "Just a single line" in result

    def test_nested_document_no_contents_falls_back_to_json_dump(self):
        doc = {"document": {"other_field": "value"}}
        result = passage_to_string(doc)
        assert "other_field" in result

    def test_flat_document_with_contents_key(self):
        doc = {"contents": "Title line\nBody"}
        result = passage_to_string(doc)
        assert "Title line" in result

    def test_flat_document_empty_contents(self):
        doc = {"contents": ""}
        result = passage_to_string(doc)
        assert result == ""

    def test_flat_document_missing_contents_key(self):
        doc = {}
        result = passage_to_string(doc)
        assert result == ""


# ---------------------------------------------------------------------------
# string_to_document
# ---------------------------------------------------------------------------


class TestStringToDocument:
    def test_with_title_format_returns_nested_document(self):
        result = string_to_document("(Title: My Title) Body content here")
        assert "document" in result
        assert "My Title" in result["document"]["contents"]
        assert "Body content here" in result["document"]["contents"]

    def test_without_title_format_returns_text_as_contents(self):
        result = string_to_document("plain text without title")
        assert result == {"document": {"contents": "plain text without title"}}

    def test_round_trip_through_passage_to_string(self):
        original = _doc('"My Title"\nSome body text')
        as_string = passage_to_string(original)
        recovered = string_to_document(as_string)
        assert "My Title" in recovered["document"]["contents"]
        assert "Some body text" in recovered["document"]["contents"]

    def test_multiline_body_preserved(self):
        text = "(Title: Header) Line1\nLine2\nLine3"
        result = string_to_document(text)
        assert "Line1" in result["document"]["contents"]


# ---------------------------------------------------------------------------
# RerankerConfig.validate
# ---------------------------------------------------------------------------


class TestRerankerConfigValidate:
    def test_default_config_is_valid(self):
        RerankerConfig().validate()

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name_or_path"):
            RerankerConfig(model_name_or_path="").validate()

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            RerankerConfig(batch_size=0).validate()

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            RerankerConfig(batch_size=-1).validate()

    def test_zero_rerank_topk_raises(self):
        with pytest.raises(ValueError, match="rerank_topk"):
            RerankerConfig(rerank_topk=0).validate()

    def test_invalid_reranker_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker type"):
            RerankerConfig(reranker_type="cross_encoder").validate()


# ---------------------------------------------------------------------------
# SentenceTransformerReranker.rerank
# ---------------------------------------------------------------------------


class TestSentenceTransformerRerankerRerank:
    def test_basic_rerank_sorts_by_score_descending(self):
        reranker = _make_reranker([0.1, 0.9])
        result = reranker.rerank(["query"], [[_doc("low"), _doc("high")]])
        assert len(result) == 1
        assert result[0][0]["score"] > result[0][1]["score"]

    def test_topk_truncates_results(self):
        reranker = _make_reranker([0.9, 0.5, 0.1])
        result = reranker.rerank(["q"], [[_doc("a"), _doc("b"), _doc("c")]], topk=2)
        assert len(result[0]) == 2

    def test_topk_none_returns_all(self):
        reranker = _make_reranker([0.3, 0.7, 0.5])
        result = reranker.rerank(["q"], [[_doc("a"), _doc("b"), _doc("c")]])
        assert len(result[0]) == 3

    def test_empty_document_list_returns_empty(self):
        reranker = _make_reranker([])
        result = reranker.rerank(["q"], [[]])
        assert result == [[]]

    def test_mismatched_queries_and_documents_raises(self):
        reranker = _make_reranker([])
        with pytest.raises(ValueError, match="same length"):
            reranker.rerank(["q1", "q2"], [[_doc("x")]])

    def test_multiple_queries_results_grouped_correctly(self):
        reranker = _make_reranker([0.8, 0.2, 0.6, 0.4])
        queries = ["q1", "q2"]
        docs = [[_doc("a1"), _doc("a2")], [_doc("b1"), _doc("b2")]]
        result = reranker.rerank(queries, docs)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_scores_are_floats(self):
        reranker = _make_reranker([0.75])
        result = reranker.rerank(["q"], [[_doc("x")]])
        assert isinstance(result[0][0]["score"], float)

    def test_model_predict_called_once(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5])
        reranker = SentenceTransformerReranker(
            model=mock_model, batch_size=8, device="cpu"
        )
        reranker.rerank(["q"], [[_doc("a"), _doc("b")]])
        mock_model.predict.assert_called_once()

    def test_model_predict_wrong_score_count_raises(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        reranker = SentenceTransformerReranker(
            model=mock_model, batch_size=8, device="cpu"
        )

        with pytest.raises(ValueError, match="wrong number of scores"):
            reranker.rerank(["q"], [[_doc("a"), _doc("b")]])
