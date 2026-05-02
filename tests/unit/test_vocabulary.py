"""Unit tests for src/search/vocabulary.py."""

import pytest

from src.search.vocabulary import (
    EOS_token,
    SOS_token,
    Vocabulary,
    build_vocabulary_from_sequences,
    extract_keywords,
    normalize_text,
    tokenize_text,
)


class TestVocabulary:
    def test_initial_state(self):
        vocab = Vocabulary()
        assert vocab.num_token == 2
        assert vocab.idx2token[SOS_token] == "SOS"
        assert vocab.idx2token[EOS_token] == "EOS"
        assert vocab.token2idx == {}
        assert vocab.token2cnt == {}

    def test_create_vocab_new_token(self):
        vocab = Vocabulary()
        vocab.create_vocab("hello")
        assert vocab.token2idx["hello"] == 2
        assert vocab.token2cnt["hello"] == 1
        assert vocab.idx2token[2] == "hello"
        assert vocab.num_token == 3

    def test_create_vocab_existing_token_increments_count(self):
        vocab = Vocabulary()
        vocab.create_vocab("hello")
        vocab.create_vocab("hello")
        assert vocab.token2cnt["hello"] == 2
        assert vocab.num_token == 3  # no new index assigned

    def test_create_vocab_multiple_distinct_tokens(self):
        vocab = Vocabulary()
        vocab.create_vocab("a")
        vocab.create_vocab("b")
        assert vocab.num_token == 4
        assert vocab.token2idx["a"] == 2
        assert vocab.token2idx["b"] == 3

    def test_add_tokens_splits_on_whitespace(self):
        vocab = Vocabulary()
        vocab.add_tokens("hello world foo")
        assert "hello" in vocab.token2idx
        assert "world" in vocab.token2idx
        assert "foo" in vocab.token2idx
        assert vocab.num_token == 5

    def test_add_tokens_duplicate_words(self):
        vocab = Vocabulary()
        vocab.add_tokens("cat cat dog")
        assert vocab.token2cnt["cat"] == 2
        assert vocab.token2cnt["dog"] == 1
        assert vocab.num_token == 4  # SOS, EOS, cat, dog

    def test_add_tokens_empty_string(self):
        vocab = Vocabulary()
        vocab.add_tokens("")
        assert vocab.num_token == 2

    def test_idx2token_and_token2idx_are_inverse(self):
        vocab = Vocabulary()
        vocab.create_vocab("x")
        idx = vocab.token2idx["x"]
        assert vocab.idx2token[idx] == "x"


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        result = normalize_text("hello, world!")
        assert "," not in result
        assert "!" not in result
        assert "hello" in result
        assert "world" in result

    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_removes_non_ascii_by_default(self):
        result = normalize_text("café")
        assert "é" not in result
        assert "cafe" in result

    def test_keep_non_ascii_flag(self):
        result = normalize_text("café", keep_non_ascii=True)
        assert "é" in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("!@#$%") == ""

    def test_numbers_preserved(self):
        assert normalize_text("123 456") == "123 456"

    def test_apostrophe_removed(self):
        result = normalize_text("it's a test")
        assert "'" not in result


class TestTokenizeText:
    def test_basic_tokenization(self):
        assert tokenize_text("Hello World") == ["hello", "world"]

    def test_max_length_truncation(self):
        assert tokenize_text("a b c d e", max_length=3) == ["a", "b", "c"]

    def test_max_length_none_returns_all(self):
        tokens = tokenize_text("a b c d e", max_length=None)
        assert tokens == ["a", "b", "c", "d", "e"]

    def test_no_max_length_kwarg(self):
        assert tokenize_text("a b c") == ["a", "b", "c"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize_text("") == []

    def test_max_length_larger_than_token_count(self):
        assert tokenize_text("a b", max_length=10) == ["a", "b"]

    def test_punctuation_stripped_before_tokenize(self):
        tokens = tokenize_text("hello, world!")
        assert tokens == ["hello", "world"]


class TestBuildVocabularyFromSequences:
    def test_basic(self):
        vocab = build_vocabulary_from_sequences(["hello world", "hello foo"])
        assert "hello" in vocab.token2idx
        assert "world" in vocab.token2idx
        assert "foo" in vocab.token2idx

    def test_count_accumulation_across_sequences(self):
        vocab = build_vocabulary_from_sequences(["cat dog", "cat bird"])
        assert vocab.token2cnt["cat"] == 2
        assert vocab.token2cnt["dog"] == 1

    def test_empty_sequences_are_skipped(self):
        vocab = build_vocabulary_from_sequences(["", "hello"])
        assert "hello" in vocab.token2idx

    def test_empty_iterable(self):
        vocab = build_vocabulary_from_sequences([])
        assert vocab.num_token == 2

    def test_max_length_limits_tokens_per_sequence(self):
        # Only the first 2 tokens of each sequence are indexed
        vocab = build_vocabulary_from_sequences(["a b c d"], max_length=2)
        assert "a" in vocab.token2idx
        assert "b" in vocab.token2idx
        assert "c" not in vocab.token2idx


class TestExtractKeywords:
    def test_returns_most_frequent_first(self):
        keywords = extract_keywords("cat cat dog cat bird dog")
        assert keywords[0] == "cat"
        assert keywords[1] == "dog"

    def test_limit_respected(self):
        keywords = extract_keywords("a b c d e f g h", limit=3)
        assert len(keywords) == 3

    def test_limit_zero_returns_empty(self):
        assert extract_keywords("hello world", limit=0) == []

    def test_limit_negative_returns_empty(self):
        assert extract_keywords("hello world", limit=-1) == []

    def test_max_length_limits_tokens_considered(self):
        # With max_length=2 only "a" and "b" enter the counter; "c" never seen
        keywords = extract_keywords("a b c c c", max_length=2, limit=5)
        assert "c" not in keywords

    def test_single_token_text(self):
        keywords = extract_keywords("word", limit=5)
        assert keywords == ["word"]
