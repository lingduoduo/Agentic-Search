from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from src.agents.control_flow_trace import ControlFlowEvent, ControlFlowRecorder


def test_recorder_sequences_and_sanitizes_details() -> None:
    recorder = ControlFlowRecorder(request_id="req-1")

    first = recorder.record(
        turn=1,
        component="search_tool",
        action="vector_db_search",
        status="completed",
        details={
            "document_count": 5,
            "query": "secret query",
            "safe_message": "x" * 300,
        },
    )
    second = recorder.record(
        turn=1,
        component="evidence_judge",
        action="evidence_evaluated",
        status="completed",
        details={"evidence_score": 0.72, "sufficient": True},
    )

    assert [first.sequence, second.sequence] == [1, 2]
    assert "query" not in first.details
    assert len(first.details["safe_message"]) == 256
    assert recorder.snapshot() == [first, second]
    assert first.timestamp.endswith("Z")


def test_recorder_copies_details_and_snapshot() -> None:
    details: dict[str, object] = {"document_count": 2}
    recorder = ControlFlowRecorder("req")
    event = recorder.record(
        turn=1,
        component="search_tool",
        action="vector_db_search",
        status="completed",
        details=details,
    )
    details["document_count"] = 99
    snapshot = recorder.snapshot()
    snapshot.clear()

    assert event.details["document_count"] == 2
    assert len(recorder.snapshot()) == 1


def test_event_is_frozen() -> None:
    event = ControlFlowRecorder("req").record(
        turn=1,
        component="planner",
        action="turn_parsed",
        status="completed",
    )
    with pytest.raises(FrozenInstanceError):
        event.sequence = 7  # type: ignore[misc]


def test_recorder_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unsupported control-flow status"):
        ControlFlowRecorder("req").record(
            turn=1,
            component="planner",
            action="turn_parsed",
            status="mystery",
        )


@pytest.mark.asyncio
async def test_sink_failure_does_not_break_recording() -> None:
    calls: list[ControlFlowEvent] = []

    async def broken_sink(event: ControlFlowEvent) -> None:
        calls.append(event)
        raise RuntimeError("sink unavailable")

    recorder = ControlFlowRecorder("req", sink=broken_sink)
    recorder.record(
        turn=1,
        component="planner",
        action="turn_parsed",
        status="completed",
    )
    await recorder.drain()
    event = recorder.record(
        turn=1,
        component="planner",
        action="search_planned",
        status="decided",
    )
    await asyncio.sleep(0)

    assert event.sequence == 2
    assert len(recorder.snapshot()) == 2
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_sink_receives_events_in_recorded_order() -> None:
    received: list[int] = []

    async def sink(event: ControlFlowEvent) -> None:
        received.append(event.sequence)

    recorder = ControlFlowRecorder("req", sink=sink)
    recorder.record(
        turn=1, component="planner", action="turn_parsed", status="completed"
    )
    recorder.record(
        turn=1, component="planner", action="search_planned", status="decided"
    )
    await recorder.drain()

    assert received == [1, 2]
