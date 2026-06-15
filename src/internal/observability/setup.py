"""One-call setup for Phoenix (Arize) and Langfuse tracing via OpenTelemetry.

Both require: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

Phoenix also works with: pip install arize-phoenix-otel
Langfuse also works with: pip install langfuse
"""

from __future__ import annotations

import base64

from .tracer import OtelTracer, set_tracer


def setup_phoenix(
    endpoint: str = "http://localhost:6006/v1/traces",
    *,
    project_name: str = "agentic-search",
    service_name: str = "agentic-search",
) -> None:
    """Configure OTEL tracing to send spans to an Arize Phoenix instance.

    Starts Phoenix locally with:  python -m phoenix.server.main serve
    Then call: setup_phoenix()
    """
    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Phoenix tracing requires: "
            "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        ) from exc

    resource = Resource.create(
        {"service.name": service_name, "project.name": project_name}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    set_tracer(OtelTracer())


def setup_langfuse(
    *,
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com",
    service_name: str = "agentic-search",
) -> None:
    """Configure OTEL tracing to send spans to Langfuse.

    Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from the Langfuse dashboard.
    """
    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Langfuse tracing requires: "
            "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        ) from exc

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    otel_endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=otel_endpoint,
        headers={"Authorization": f"Basic {auth}"},
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    set_tracer(OtelTracer())
