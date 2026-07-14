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


def pipeline_stage_summary(
    query: str,
    *,
    citations: list,
    documents: list,
    metadata: dict,
) -> dict:
    """Normalize pipeline metadata for persistence and debug capture."""
    ranking = dict(metadata.get("ranking") or {})
    inference = dict(metadata.get("inference") or {})
    operations = list(ranking.get("operations") or [])
    reranker = ranking.get("reranker")
    if reranker is None and "external_rerank" in operations:
        reranker = ranking.get("rerank_url") or "external"
    rerank_status = ranking.get("rerank_status")
    if reranker is None and rerank_status not in (None, "disabled"):
        reranker = ranking.get("rerank_url") or "external"
    degradation_reason = ranking.get("degradation_reason") or metadata.get(
        "rerank_degraded"
    )
    if degradation_reason is None and ranking.get("degraded"):
        degradation_reason = rerank_status or "reranking_degraded"
    inference_mode = inference.get("mode")
    if metadata.get("inference_fallback") == "synthesis_failed":
        inference_mode = "deterministic_fallback"
    return {
        "retrieval": {
            "query": metadata.get("retrieval_query", query),
            "provider": metadata.get("source_provider"),
            "candidate_count": metadata.get(
                "candidate_count", ranking.get("candidate_count")
            ),
        },
        "ranking": {
            "operations": operations,
            "evidence_count": len(documents),
            "reranker": reranker,
            "degradation_reason": degradation_reason,
        },
        "inference": {
            "mode": inference_mode or "unknown",
            "model": inference.get("model"),
        },
        "answer": {
            "citations": list(citations),
            "document_ids": [document.id for document in documents],
        },
    }


def record_pipeline_stages(summary: dict) -> None:
    """Mirror a normalized persisted summary into the ambient debug capture."""
    for stage in ("retrieval", "ranking", "inference"):
        record_stage(stage, stage, summary[stage])


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
