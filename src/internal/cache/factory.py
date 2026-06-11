from __future__ import annotations

import os
from collections.abc import Callable

from src.internal.cache.interface import CacheBackend
from src.internal.cache.interface import CacheBackendType

CACHE_BACKEND: CacheBackendType = CacheBackendType(
    os.environ.get("CACHE_BACKEND", CacheBackendType.REDIS.value)
)


def _build_redis_backend(tenant_id: str) -> CacheBackend:
    import redis

    from src.internal.cache.redis_backend import RedisCacheBackend

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    password = os.environ.get("REDIS_PASSWORD") or None
    client = redis.Redis(host=host, port=port, password=password)
    return RedisCacheBackend(client)


def _build_postgres_backend(tenant_id: str) -> CacheBackend:
    from src.internal.cache.postgres_backend import PostgresCacheBackend

    return PostgresCacheBackend(tenant_id)


_BACKEND_BUILDERS: dict[CacheBackendType, Callable[[str], CacheBackend]] = {
    CacheBackendType.REDIS: _build_redis_backend,
    CacheBackendType.POSTGRES: _build_postgres_backend,
}


def get_tenant_cache_backend(*, tenant_id: str | None = None) -> CacheBackend:
    """Return a tenant-aware ``CacheBackend``.

    If *tenant_id* is ``None``, the current tenant is read from the
    thread-local context variable.
    """
    if tenant_id is None:
        from shared_configs.contextvars import get_current_tenant_id

        tenant_id = get_current_tenant_id()

    builder = _BACKEND_BUILDERS.get(CACHE_BACKEND)
    if builder is None:
        raise ValueError(
            f"Unsupported CACHE_BACKEND={CACHE_BACKEND!r}. "
            f"Supported values: {[t.value for t in CacheBackendType]}"
        )
    return builder(tenant_id)


def get_shared_cache_backend() -> CacheBackend:
    """Return a ``CacheBackend`` in the shared (cross-tenant) namespace."""
    from shared_configs.configs import DEFAULT_REDIS_PREFIX

    return get_tenant_cache_backend(tenant_id=DEFAULT_REDIS_PREFIX)
