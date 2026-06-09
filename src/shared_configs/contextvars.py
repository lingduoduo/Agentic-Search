"""Shared context variables for request-scoped state."""

import contextlib
from contextvars import ContextVar

CURRENT_TENANT_ID_CONTEXTVAR: ContextVar[str | None] = ContextVar(
    "current_tenant_id", default=None
)
CURRENT_ENDPOINT_CONTEXTVAR: ContextVar[str | None] = ContextVar(
    "current_endpoint", default=None
)


def get_current_tenant_id() -> str | None:
    return CURRENT_TENANT_ID_CONTEXTVAR.get()


@contextlib.contextmanager
def trace(name, **kwargs):
    """No-op tracing context manager."""
    yield
