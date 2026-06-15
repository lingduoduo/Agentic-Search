"""Lightweight OpenTelemetry-based tracing abstraction for the RAG pipeline.

No-op by default — call setup_phoenix() or setup_langfuse() in setup.py to activate.
"""

from __future__ import annotations

import contextlib
from typing import Generator


# ---------------------------------------------------------------------------
# Tracer implementations
# ---------------------------------------------------------------------------


class NoOpTracer:
    """Default tracer — all spans are no-ops, zero overhead."""

    @contextlib.contextmanager
    def span(self, name: str, **attrs: object) -> Generator[None, None, None]:
        yield


class OtelTracer:
    """OpenTelemetry tracer. Works with any OTEL-compatible backend."""

    def __init__(self, tracer_name: str = "agentic_search") -> None:
        from opentelemetry import trace  # type: ignore[import]

        self._tracer = trace.get_tracer(tracer_name)

    @contextlib.contextmanager
    def span(self, name: str, **attrs: object) -> Generator[None, None, None]:
        with self._tracer.start_as_current_span(name) as otel_span:
            for k, v in attrs.items():
                if v is None:
                    continue
                otel_span.set_attribute(
                    k, v if isinstance(v, (bool, int, float, str)) else str(v)
                )
            yield


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracer: NoOpTracer | OtelTracer = NoOpTracer()


def get_tracer() -> NoOpTracer | OtelTracer:
    return _tracer


def set_tracer(tracer: NoOpTracer | OtelTracer) -> None:
    global _tracer
    _tracer = tracer
