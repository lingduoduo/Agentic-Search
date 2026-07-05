"""Ambient per-request capture of full raw stage payloads for the Dev Console.

A ``RequestCapture`` is stashed in a ContextVar for the duration of one
``/api/agent`` request when debug panels are enabled. Instrumentation points
call ``record_stage`` / ``capture_stage``, which no-op instantly when no capture
is active — so the hot path pays only one ContextVar ``.get()`` when the flag is
off. This is a SEPARATE channel from the sanitized ControlFlowRecorder: payloads
here are raw (full prompts, document bodies, completions) and never persisted.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from time import monotonic

_current: contextvars.ContextVar["RequestCapture | None"] = contextvars.ContextVar(
    "request_capture", default=None
)


@dataclass
class StageRecord:
    stage: str
    label: str
    timestamp: float
    duration_ms: float | None
    payload: dict


@dataclass
class RequestCapture:
    request_id: str
    query: str
    created_at: float
    route: str | None = None
    route_degraded: str | None = None
    total_ms: float | None = None
    stages: list[StageRecord] = field(default_factory=list)
    _start: float = field(default_factory=monotonic, repr=False)

    def add(
        self, stage: str, label: str, payload: dict, duration_ms: float | None = None
    ) -> None:
        self.stages.append(
            StageRecord(
                stage=stage,
                label=label,
                timestamp=monotonic() - self._start,
                duration_ms=duration_ms,
                payload=payload,
            )
        )

    def finish(self) -> None:
        self.total_ms = (monotonic() - self._start) * 1000.0

    def snapshot(self) -> dict:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "created_at": self.created_at,
            "route": self.route,
            "route_degraded": self.route_degraded,
            "total_ms": self.total_ms,
            "stages": [
                {
                    "stage": s.stage,
                    "label": s.label,
                    "timestamp": s.timestamp,
                    "duration_ms": s.duration_ms,
                    "payload": s.payload,
                }
                for s in self.stages
            ],
        }


def start_capture(request_id: str, query: str) -> contextvars.Token:
    """Begin a capture for this request; returns a token to reset with."""
    return _current.set(
        RequestCapture(request_id=request_id, query=query, created_at=monotonic())
    )


def reset_capture(token: contextvars.Token) -> None:
    _current.reset(token)


def active() -> "RequestCapture | None":
    return _current.get()


def record_stage(
    stage: str, label: str, payload: dict, duration_ms: float | None = None
) -> None:
    cap = _current.get()
    if cap is not None:
        cap.add(stage, label, payload, duration_ms)


@contextlib.contextmanager
def capture_stage(stage: str, label: str):
    """Time a block and record a stage from the payload dict the caller mutates.

    No-op (still yields a throwaway dict) when no capture is active.
    """
    cap = _current.get()
    if cap is None:
        yield {}
        return
    payload: dict = {}
    started = monotonic()
    try:
        yield payload
    finally:
        cap.add(stage, label, payload, (monotonic() - started) * 1000.0)
