from __future__ import annotations

import pytest

from examples.verify_claim_streaming import (
    StreamSummary,
    main,
    summarize,
)


def timed(*events: dict) -> list[tuple[float, dict]]:
    """Attach a monotonically increasing arrival time to each event."""
    return [(float(index), event) for index, event in enumerate(events)]


def claim(text: str) -> dict:
    return {"type": "claim", "text": text}


def answer(text: str) -> dict:
    return {"type": "answer", "text": text}


def test_streamed_claims_that_join_to_the_answer_pass() -> None:
    summary = summarize(
        timed(
            {"type": "progress", "turn": 1, "text": "retrieve"},
            claim("FAISS indexes vectors."),
            claim("It supports IVF and HNSW."),
            answer("FAISS indexes vectors. It supports IVF and HNSW."),
            {"type": "done", "session_id": "s1"},
        )
    )

    assert summary.diagnostics == ()
    assert summary.ok
    assert summary.claims == ("FAISS indexes vectors.", "It supports IVF and HNSW.")
    assert summary.time_to_first_claim == 1.0
    assert summary.time_to_answer == 3.0
    assert summary.lead == 2.0


def test_answer_that_is_not_the_join_of_the_claims_is_reported() -> None:
    summary = summarize(
        timed(
            claim("Dense retrieval uses embeddings."),
            answer("Dense retrieval uses embeddings. Sparse retrieval uses BM25."),
        )
    )

    assert not summary.ok
    assert any("join" in diagnostic for diagnostic in summary.diagnostics)


def test_no_claims_reports_the_abstain_ordering_assumption() -> None:
    summary = summarize(timed(answer("FAISS indexes vectors.")))

    assert not summary.ok
    assert any("abstain" in diagnostic for diagnostic in summary.diagnostics)
    assert summary.time_to_first_claim is None


def test_claims_arriving_after_the_answer_are_not_streaming() -> None:
    summary = summarize(timed(answer("A. B."), claim("A."), claim("B.")))

    assert not summary.ok
    assert any("before" in diagnostic for diagnostic in summary.diagnostics)


def test_claims_that_share_the_answers_arrival_time_are_not_streaming() -> None:
    summary = summarize([(4.0, claim("A.")), (4.0, answer("A."))])

    assert not summary.ok
    assert any("lead" in diagnostic for diagnostic in summary.diagnostics)


def test_a_missing_answer_event_is_reported() -> None:
    summary = summarize(timed(claim("A.")))

    assert not summary.ok
    assert any("answer" in diagnostic for diagnostic in summary.diagnostics)


def test_a_server_error_event_is_reported() -> None:
    summary = summarize(timed({"type": "error", "detail": "llm unavailable"}))

    assert not summary.ok
    assert any("llm unavailable" in diagnostic for diagnostic in summary.diagnostics)


def test_main_returns_zero_when_the_stream_satisfies_the_invariant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events = timed(claim("A."), claim("B."), answer("A. B."))

    exit_code = main(["--query", "q"], open_stream=lambda **_: iter(events))

    assert exit_code == 0
    assert "PASS" in capsys.readouterr().out


def test_main_returns_one_when_the_invariant_is_violated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events = timed(claim("A."), answer("A. B."))

    exit_code = main(["--query", "q"], open_stream=lambda **_: iter(events))

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_passes_the_request_arguments_through_to_the_stream() -> None:
    seen: dict = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        return iter(timed(claim("A."), answer("A.")))

    main(
        ["--query", "what is FAISS", "--url", "http://host/api", "--top-k", "9"],
        open_stream=fake_stream,
    )

    assert seen["query"] == "what is FAISS"
    assert seen["url"] == "http://host/api"
    assert seen["top_k"] == 9


def test_summary_reports_a_transport_failure_rather_than_raising() -> None:
    def failing_stream(**_):
        raise ConnectionError("connection refused")

    exit_code = main(["--query", "q"], open_stream=failing_stream)

    assert exit_code == 2


def test_stream_summary_is_immutable() -> None:
    summary = StreamSummary(
        claims=("A.",),
        answer="A.",
        time_to_first_claim=1.0,
        time_to_answer=2.0,
        diagnostics=(),
    )

    with pytest.raises(AttributeError):
        summary.answer = "B."  # type: ignore[misc]
