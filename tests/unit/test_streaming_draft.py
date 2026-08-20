"""Tests for the incremental answer-draft reader."""

from __future__ import annotations

from src.context.streaming_draft import IncrementalDraftReader

IDS = {"D1", "D2"}


def _feed_all(reader: IncrementalDraftReader, text: str, size: int = 7) -> list:
    """Feed in small chunks, to prove claim boundaries survive arbitrary splits."""
    out = []
    for index in range(0, len(text), size):
        out.extend(reader.feed(text[index : index + size]))
    return out


DRAFT = (
    '{"abstain": false, "missing_information": [], "claims": ['
    '{"text": "First claim.", "evidence_ids": ["D1"]}, '
    '{"text": "Second claim.", "evidence_ids": ["D1", "D2"]}]}'
)


def test_yields_each_claim_as_it_completes():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(reader, DRAFT)
    assert [claim.text for claim in claims] == ["First claim.", "Second claim."]
    assert claims[1].evidence_ids == ["D1", "D2"]
    assert reader.abstain is False
    assert reader.gave_up is False


def test_a_claim_is_not_yielded_until_its_closing_brace_arrives():
    reader = IncrementalDraftReader(IDS)
    partial = '{"abstain": false, "missing_information": [], "claims": [{"text": "Half'
    assert reader.feed(partial) == []
    assert reader.abstain is False


def test_abstaining_drafts_yield_nothing():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": true, "missing_information": [], "claims": ['
        '{"text": "Discarded.", "evidence_ids": ["D1"]}]}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_claims_before_abstain_yields_nothing():
    """Old key order: abstain is not decidable in time, so refuse to stream."""
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"claims": [{"text": "Too early.", "evidence_ids": ["D1"]}], '
        '"missing_information": [], "abstain": false}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_old_key_order_with_abstain_true_gives_up_before_emitting():
    """Dangerous case: claims appear before abstain in the buffer, but abstain
    is true. The refusal is gated on abstain's presence, not its position, so
    this must give up before emitting -- and it must hold in a single feed()
    call, with no chunk boundary to hide behind."""
    reader = IncrementalDraftReader(IDS)
    claims = reader.feed(
        '{"claims": [{"text": "Discarded.", "evidence_ids": ["D1"]}], '
        '"missing_information": [], "abstain": true}'
    )
    assert claims == []
    assert reader.gave_up is True


def test_unknown_evidence_ids_give_up():
    """The whole-draft parse rejects these, so streaming must not race ahead."""
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": false, "missing_information": [], "claims": ['
        '{"text": "Bad cite.", "evidence_ids": ["NOPE"]}]}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_malformed_claim_gives_up_without_raising():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": false, "missing_information": [], "claims": ['
        '{"text": "Ok.", "evidence_ids": ["D1"], "extra": 1}]}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_non_object_claims_element_gives_up_instead_of_stalling():
    """A bare string (or any non-object) in the claims array must not leave the
    cursor stuck forever -- that would silently drop every claim after it while
    reporting gave_up as False."""
    reader = IncrementalDraftReader(IDS)
    claims = reader.feed(
        '{"abstain": false, "missing_information": [], "claims": '
        '["oops", {"text": "ok", "evidence_ids": ["D1"]}]}'
    )
    assert claims == []
    assert reader.gave_up is True


def test_incomplete_object_tail_does_not_give_up():
    """A well-formed claim object that has not finished arriving yet must be
    treated as merely incomplete, not as unrecognised input."""
    reader = IncrementalDraftReader(IDS)
    claims = reader.feed(
        '{"abstain": false, "missing_information": [], "claims": '
        '[{"text": "Not done yet'
    )
    assert claims == []
    assert reader.gave_up is False


def test_braces_inside_claim_text_do_not_split_the_object():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": false, "missing_information": [], "claims": ['
        '{"text": "Use {\\"k\\": 1} in config.", "evidence_ids": ["D1"]}]}',
    )
    assert [claim.text for claim in claims] == ['Use {"k": 1} in config.']


def test_prose_prefix_gives_up():
    """Some providers wrap JSON in a fence or a sentence."""
    reader = IncrementalDraftReader(IDS)
    assert _feed_all(reader, 'Sure! ```json\n{"abstain": false') == []


def test_feed_after_giving_up_stays_silent():
    reader = IncrementalDraftReader(IDS)
    _feed_all(reader, '{"abstain": true, "missing_information": [], "claims": [')
    assert reader.gave_up is True
    assert reader.feed('{"text": "x", "evidence_ids": ["D1"]}]}') == []
