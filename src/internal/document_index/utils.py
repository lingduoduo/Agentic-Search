"""Utility stubs for the document index layer."""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import time
from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeVar

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up a standard logger for the given name."""
    return logging.getLogger(name)


def batch_generator(
    items: Iterable[_T], batch_size: int
) -> Generator[list[_T], None, None]:
    """Yield successive non-overlapping batches from items."""
    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# C0 controls (except \t, \n, \r) and DEL — intentionally excludes C1 range (\x80-\x9f)
_INVALID_UNICODE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def remove_invalid_unicode_chars(text: str) -> str:
    """Remove control characters that are invalid in most storage backends."""
    return _INVALID_UNICODE_RE.sub("", text)


def convert_metadata_list_of_strings_to_dict(
    metadata: list[str] | dict[str, Any],
) -> dict[str, Any]:
    """Convert a list of 'key:value' strings to a dict, or pass through a dict."""
    if isinstance(metadata, dict):
        return metadata
    result: dict[str, Any] = {}
    for item in metadata:
        if ":" in item:
            key, _, value = item.partition(":")
            result[key] = value
    return result


def get_experts_stores_representations(
    owners: list[str] | None,
) -> list[str]:
    """Return expert representation list. Stub — returns input unchanged."""
    return owners or []


def split_relationship_id(relationship_id: str) -> tuple[str, str, str]:
    """Split a 'source:RELATION:target' relationship ID string.

    Returns (source, relation, target). Raises ValueError if format is wrong.
    """
    parts = relationship_id.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"relationship_id must be 'source:RELATION:target', got: {relationship_id!r}"
        )
    return parts[0], parts[1], parts[2]


@contextlib.contextmanager
def redis_shared_lock(lock_name: str, **kwargs: Any):
    """Stub lock context manager. Does not acquire a real Redis lock."""
    yield


class _NullKVStore:
    """Stub key-value store that always returns None."""

    def get(self, key: str) -> None:
        return None

    def set(self, key: str, value: object) -> None:
        pass

    def delete(self, key: str) -> None:
        pass


def get_shared_kv_store() -> _NullKVStore:
    """Stub — returns a no-op KV store. Replace with real Redis KV when needed."""
    return _NullKVStore()


def log_function_time(
    *,
    print_only: bool = False,
    debug_only: bool = False,
    include_args: bool = False,
    include_args_subset: dict[str, Any] | None = None,
) -> Callable[[_F], _F]:
    """Decorator that logs the wall-clock time of the wrapped function.

    Logs at DEBUG when debug_only=True, otherwise at INFO.
    print_only, include_args, and include_args_subset are accepted for
    call-site compatibility but have no effect.
    """

    def decorator(func: _F) -> _F:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                msg = f"{func.__qualname__} took {elapsed:.3f}s"
                if debug_only:
                    logger.debug(msg)
                else:
                    logger.info(msg)

        return wrapper  # type: ignore[return-value]

    return decorator
