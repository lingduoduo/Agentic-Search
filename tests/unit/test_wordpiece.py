from pathlib import Path

import pytest

from src.model.wordpiece import (
    MAX_CHARS_PER_WORD,
    PAD_ID,
    UNK_ID,
    WordPieceVocabulary,
)

# A miniature vocabulary that exercises every branch. Ids are positional.
_TOKENS = (
    ["[PAD]"]
    + [f"[unused{index}]" for index in range(99)]
    + ["[UNK]"]
    + [
        "the",
        "post",
        "##mo",
        "##rte",
        "##m",
        "stand",
        "##up",
        "dashboard",
    ]
)
_ID = {token: index for index, token in enumerate(_TOKENS)}


def _vocab() -> WordPieceVocabulary:
    return WordPieceVocabulary.from_tokens(_TOKENS)


def test_special_ids_follow_the_bert_layout():
    assert PAD_ID == 0
    assert UNK_ID == 100
    assert _vocab().size == len(_TOKENS)


def test_whole_word_encodes_to_its_own_id():
    assert _vocab().encode("dashboard") == [_ID["dashboard"]]


def test_unseen_word_decomposes_into_continuations():
    assert _vocab().encode("postmortem") == [
        _ID["post"],
        _ID["##mo"],
        _ID["##rte"],
        _ID["##m"],
    ]


def test_greedy_match_prefers_the_longest_prefix():
    assert _vocab().encode("standup") == [_ID["stand"], _ID["##up"]]


def test_undecomposable_word_falls_back_to_unknown_once():
    # 'zzz' shares no prefix with any vocabulary entry.
    assert _vocab().encode("zzz") == [UNK_ID]


def test_unknown_fallback_is_per_word_not_per_query():
    assert _vocab().encode("zzz dashboard") == [UNK_ID, _ID["dashboard"]]


def test_word_longer_than_the_guard_is_unknown_without_scanning():
    assert _vocab().encode("a" * (MAX_CHARS_PER_WORD + 1)) == [UNK_ID]


def test_empty_and_punctuation_only_text_encode_to_nothing():
    assert _vocab().encode("") == []
    assert _vocab().encode("!!! ???") == []


def test_normalization_matches_the_shared_text_pipeline():
    # normalize_text lowercases, strips accents, and drops punctuation.
    assert _vocab().encode("Dashboard!") == _vocab().encode("dashboard")


def test_from_file_reads_one_token_per_line_in_id_order(tmp_path: Path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(_TOKENS) + "\n", encoding="utf-8")

    vocabulary = WordPieceVocabulary.from_file(path)

    assert vocabulary.size == len(_TOKENS)
    assert vocabulary.encode("standup") == [_ID["stand"], _ID["##up"]]


def test_from_file_rejects_a_vocabulary_without_the_required_specials(
    tmp_path: Path,
):
    path = tmp_path / "vocab.txt"
    path.write_text("the\npost\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[PAD\]"):
        WordPieceVocabulary.from_file(path)
