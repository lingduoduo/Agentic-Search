from __future__ import annotations

from src.internal.servers.web import request_capture as rc


def test_record_stage_noops_when_inactive():
    # No active capture → record_stage is a silent no-op, active() is None.
    assert rc.active() is None
    rc.record_stage("intent", "classify_route", {"label": "chat"})
    assert rc.active() is None


def test_start_capture_records_and_snapshots():
    token = rc.start_capture("req-1", "what is faiss")
    try:
        cap = rc.active()
        assert cap is not None and cap.request_id == "req-1"
        rc.record_stage("intent", "classify_route", {"raw": "chat"}, duration_ms=12.0)
        rc.record_stage("final", "answer", {"answer": "hi"})
        cap.route = "chat"
        cap.finish()
        snap = cap.snapshot()
    finally:
        rc.reset_capture(token)
    assert rc.active() is None
    assert snap["request_id"] == "req-1"
    assert snap["query"] == "what is faiss"
    assert snap["route"] == "chat"
    assert [s["stage"] for s in snap["stages"]] == ["intent", "final"]
    assert snap["stages"][0]["payload"] == {"raw": "chat"}
    assert snap["stages"][0]["duration_ms"] == 12.0
    assert snap["total_ms"] is not None


def test_capture_stage_times_and_records():
    token = rc.start_capture("req-2", "q")
    try:
        with rc.capture_stage("llm", "complete") as payload:
            payload["model"] = "gpt-4o-mini"
            payload["completion"] = "ok"
        cap = rc.active()
        assert cap.stages[0].stage == "llm"
        assert cap.stages[0].payload == {"model": "gpt-4o-mini", "completion": "ok"}
        assert cap.stages[0].duration_ms is not None
    finally:
        rc.reset_capture(token)


def test_capture_stage_noops_when_inactive():
    # Must not raise and must record nothing when no capture is active.
    with rc.capture_stage("llm", "complete") as payload:
        payload["model"] = "x"
    assert rc.active() is None
