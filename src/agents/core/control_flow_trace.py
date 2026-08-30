"""Safe structured tracing for one agent control-flow run."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

JsonValue = str | int | float | bool | None
EventSink = Callable[["ControlFlowEvent"], Awaitable[None]]

ALLOWED_STATUSES = frozenset({"started", "completed", "decided", "skipped", "failed"})
ALLOWED_DETAIL_KEYS = frozenset(
    {
        "retriever",
        "query_count",
        "document_count",
        "citation_count",
        "evidence_score",
        "sufficient",
        "search_round",
        "effective_budget",
        "cache_hit_count",
        "fallback",
        "duplicate_count",
        "overflow_count",
        "error_category",
        "safe_message",
        "exit_reason",
        "decision",
        "llm_first_token_ms",
        "time_to_first_claim_ms",
    }
)
_MAX_DETAIL_STRING = 256


@dataclass(frozen=True, slots=True)
class ControlFlowEvent:
    sequence: int
    timestamp: str
    turn: int
    component: str
    action: str
    status: str
    duration_ms: int | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sanitize_details(details: Mapping[str, object]) -> dict[str, JsonValue]:
    sanitized: dict[str, JsonValue] = {}
    for key, value in details.items():
        if key not in ALLOWED_DETAIL_KEYS:
            continue
        if isinstance(value, str):
            sanitized[key] = value[:_MAX_DETAIL_STRING]
        elif value is None or isinstance(value, (bool, int, float)):
            sanitized[key] = value
    return sanitized


class ControlFlowRecorder:
    """Assign order, sanitize, retain, log, and optionally stream trace events."""

    def __init__(
        self,
        request_id: str,
        *,
        session_id: str | None = None,
        sink: EventSink | None = None,
    ) -> None:
        self._request_id = request_id
        self._session_id = session_id
        self._sink = sink
        self._sink_disabled = False
        self._sink_lock = asyncio.Lock()
        self._pending: list[asyncio.Task[None]] = []
        self._events: list[ControlFlowEvent] = []

    def record(
        self,
        *,
        turn: int,
        component: str,
        action: str,
        status: str,
        duration_ms: int | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ControlFlowEvent:
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported control-flow status: {status}")
        event = ControlFlowEvent(
            sequence=len(self._events) + 1,
            timestamp=datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            turn=turn,
            component=component,
            action=action,
            status=status,
            duration_ms=duration_ms,
            details=_sanitize_details(details or {}),
        )
        self._events.append(event)
        logger.info(
            "agent_control_flow",
            extra={
                "control_flow_request_id": self._request_id,
                "control_flow_session_id": self._session_id,
                "control_flow_event": event.to_dict(),
            },
        )
        self._schedule_sink(event)
        return event

    def snapshot(self) -> list[ControlFlowEvent]:
        return list(self._events)

    async def drain(self) -> None:
        pending, self._pending = self._pending, []
        if pending:
            await asyncio.gather(*pending)

    def _schedule_sink(self, event: ControlFlowEvent) -> None:
        if self._sink is None or self._sink_disabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending.append(loop.create_task(self._deliver(event)))

    async def _deliver(self, event: ControlFlowEvent) -> None:
        async with self._sink_lock:
            if self._sink is None or self._sink_disabled:
                return
            try:
                await self._sink(event)
            except Exception:  # noqa: BLE001 - tracing must never fail the agent
                self._sink_disabled = True
                logger.warning(
                    "control-flow trace sink disabled after failure",
                    exc_info=True,
                )
