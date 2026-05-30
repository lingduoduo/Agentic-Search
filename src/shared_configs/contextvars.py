"""Shared context variable stubs."""

import contextlib


def get_current_tenant_id() -> str | None:
    return None


@contextlib.contextmanager
def trace(name, **kwargs):
    """No-op tracing context manager."""
    yield
