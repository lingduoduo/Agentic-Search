"""Abstract cache backend interface."""

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
