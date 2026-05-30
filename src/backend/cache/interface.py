"""Abstract cache backend interface and in-memory implementation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...


class InMemoryCache(CacheBackend):
    """Thread-unsafe in-memory cache for local / test mode."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._store
