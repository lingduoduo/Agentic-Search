"""Abstract cache backend interface and in-memory implementation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...
    @abstractmethod
    def set(
        self, key: str, value: Any, ex: int | None = None, ttl: int | None = None
    ) -> None: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def expire(self, key: str, ttl_seconds: int) -> None: ...


class InMemoryCache(CacheBackend):
    """Thread-unsafe in-memory cache for local / test mode."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(
        self, key: str, value: Any, ex: int | None = None, ttl: int | None = None
    ) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._store

    def expire(self, key: str, ttl_seconds: int) -> None:
        pass  # TTL not tracked in-memory


# Exception types that are safe to swallow on cache failures
CACHE_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (Exception,)

_default_cache: CacheBackend | None = None


def get_cache_backend() -> CacheBackend:
    """Return the active cache backend, defaulting to InMemoryCache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = InMemoryCache()
    return _default_cache


def set_cache_backend(backend: CacheBackend) -> None:
    """Override the global cache backend (e.g. for tests or Redis integration)."""
    global _default_cache
    _default_cache = backend
