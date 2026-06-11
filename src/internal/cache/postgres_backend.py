"""PostgreSQL-backed ``CacheBackend`` for NO_VECTOR_DB deployments.

Uses the ``cache_store`` table for key-value storage, PostgreSQL advisory locks
for distributed locking, and a polling loop for the BLPOP pattern.

DB session helpers (``get_session_with_tenant``, ``get_session_with_current_tenant``)
and the ``CacheStore`` model are resolved lazily so the module can be imported
without a live database connection.  Operations will raise ``NotImplementedError``
if the DB layer is not wired up.
"""

from __future__ import annotations

import hashlib
import struct
import time
import uuid
from contextlib import AbstractContextManager
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.internal.cache.interface import CacheBackend
from src.internal.cache.interface import CacheLock
from src.internal.cache.interface import TTL_KEY_NOT_FOUND
from src.internal.cache.interface import TTL_NO_EXPIRY

_LIST_KEY_PREFIX = "_q:"
_LIST_KEY_RANGE_TERMINATOR = ";"
_LIST_ITEM_TTL_SECONDS = 3600
_LOCK_POLL_INTERVAL = 0.1
_BLPOP_POLL_INTERVAL = 0.25


def _get_session_with_tenant(tenant_id: str):  # type: ignore[return]
    try:
        from src.internal.db.engine.sql_engine import get_session_with_tenant  # type: ignore[import]

        return get_session_with_tenant(tenant_id=tenant_id)
    except ImportError:
        raise NotImplementedError(
            "PostgresCacheBackend requires src.internal.db.engine.sql_engine"
        )


def _get_session_with_current_tenant():  # type: ignore[return]
    try:
        from src.internal.db.engine.sql_engine import get_session_with_current_tenant  # type: ignore[import]

        return get_session_with_current_tenant()
    except ImportError:
        raise NotImplementedError(
            "PostgresCacheBackend requires src.internal.db.engine.sql_engine"
        )


def _get_cache_store_model():  # type: ignore[return]
    try:
        from src.internal.db.models import CacheStore  # type: ignore[import]

        return CacheStore
    except ImportError:
        raise NotImplementedError(
            "PostgresCacheBackend requires src.internal.db.models.CacheStore"
        )


def _list_item_key(key: str) -> str:
    return f"{_LIST_KEY_PREFIX}{key}:{time.time_ns()}:{uuid.uuid4().hex}"


def _to_bytes(value: str | bytes | int | float) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode()


class PostgresCacheLock(CacheLock):
    """Advisory-lock-based distributed lock backed by PostgreSQL."""

    def __init__(self, lock_id: int, timeout: float | None, tenant_id: str) -> None:
        self._lock_id = lock_id
        self._timeout = timeout
        self._tenant_id = tenant_id
        self._session_cm: AbstractContextManager[Session] | None = None
        self._session: Session | None = None
        self._acquired = False

    def acquire(
        self,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ) -> bool:
        self._session_cm = _get_session_with_tenant(self._tenant_id)
        self._session = self._session_cm.__enter__()
        try:
            if not blocking:
                return self._try_lock()

            effective_timeout = blocking_timeout or self._timeout
            deadline = (
                (time.monotonic() + effective_timeout) if effective_timeout else None
            )
            while True:
                if self._try_lock():
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(_LOCK_POLL_INTERVAL)
        finally:
            if not self._acquired:
                self._close_session()

    def release(self) -> None:
        if not self._acquired or self._session is None:
            return
        try:
            self._session.execute(select(func.pg_advisory_unlock(self._lock_id)))
        finally:
            self._acquired = False
            self._close_session()

    def owned(self) -> bool:
        return self._acquired

    def _close_session(self) -> None:
        if self._session_cm is not None:
            try:
                self._session_cm.__exit__(None, None, None)
            finally:
                self._session_cm = None
                self._session = None

    def _try_lock(self) -> bool:
        assert self._session is not None
        result = self._session.execute(
            select(func.pg_try_advisory_lock(self._lock_id))
        ).scalar()
        if result:
            self._acquired = True
            return True
        return False


class PostgresCacheBackend(CacheBackend):
    """``CacheBackend`` backed by the ``cache_store`` table in PostgreSQL."""

    def __init__(self, tenant_id: str) -> None:
        self._tenant_id = tenant_id

    def get(self, key: str) -> bytes | None:
        CacheStore = _get_cache_store_model()
        stmt = select(CacheStore.value).where(
            CacheStore.key == key,
            or_(CacheStore.expires_at.is_(None), CacheStore.expires_at > func.now()),
        )
        with _get_session_with_tenant(self._tenant_id) as session:
            value = session.execute(stmt).scalar_one_or_none()
        if value is None:
            return None
        return bytes(value)

    def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,
    ) -> None:
        CacheStore = _get_cache_store_model()
        value_bytes = _to_bytes(value)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=ex)
            if ex is not None
            else None
        )
        stmt = (
            pg_insert(CacheStore)
            .values(key=key, value=value_bytes, expires_at=expires_at)
            .on_conflict_do_update(
                index_elements=[CacheStore.key],
                set_={"value": value_bytes, "expires_at": expires_at},
            )
        )
        with _get_session_with_tenant(self._tenant_id) as session:
            session.execute(stmt)
            session.commit()

    def delete(self, key: str) -> None:
        CacheStore = _get_cache_store_model()
        with _get_session_with_tenant(self._tenant_id) as session:
            session.execute(delete(CacheStore).where(CacheStore.key == key))
            session.commit()

    def exists(self, key: str) -> bool:
        CacheStore = _get_cache_store_model()
        stmt = (
            select(CacheStore.key)
            .where(
                CacheStore.key == key,
                or_(
                    CacheStore.expires_at.is_(None),
                    CacheStore.expires_at > func.now(),
                ),
            )
            .limit(1)
        )
        with _get_session_with_tenant(self._tenant_id) as session:
            return session.execute(stmt).first() is not None

    def expire(self, key: str, seconds: int) -> None:
        CacheStore = _get_cache_store_model()
        new_exp = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        stmt = (
            update(CacheStore).where(CacheStore.key == key).values(expires_at=new_exp)
        )
        with _get_session_with_tenant(self._tenant_id) as session:
            session.execute(stmt)
            session.commit()

    def ttl(self, key: str) -> int:
        CacheStore = _get_cache_store_model()
        stmt = select(CacheStore.expires_at).where(CacheStore.key == key)
        with _get_session_with_tenant(self._tenant_id) as session:
            result = session.execute(stmt).first()
        if result is None:
            return TTL_KEY_NOT_FOUND
        expires_at: datetime | None = result[0]
        if expires_at is None:
            return TTL_NO_EXPIRY
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return TTL_KEY_NOT_FOUND
        return int(remaining)

    def lock(self, name: str, timeout: float | None = None) -> CacheLock:
        return PostgresCacheLock(
            self._lock_id_for(name), timeout, tenant_id=self._tenant_id
        )

    def rpush(self, key: str, value: str | bytes) -> None:
        self.set(_list_item_key(key), value, ex=_LIST_ITEM_TTL_SECONDS)

    def blpop(self, keys: list[str], timeout: int = 0) -> tuple[bytes, bytes] | None:
        if timeout <= 0:
            raise ValueError(
                "PostgresCacheBackend.blpop requires timeout > 0. "
                "timeout=0 would block the calling thread indefinitely."
            )
        CacheStore = _get_cache_store_model()
        deadline = time.monotonic() + timeout
        while True:
            for key in keys:
                lower = f"{_LIST_KEY_PREFIX}{key}:"
                upper = f"{_LIST_KEY_PREFIX}{key}{_LIST_KEY_RANGE_TERMINATOR}"
                stmt = (
                    select(CacheStore)
                    .where(
                        CacheStore.key >= lower,
                        CacheStore.key < upper,
                        or_(
                            CacheStore.expires_at.is_(None),
                            CacheStore.expires_at > func.now(),
                        ),
                    )
                    .order_by(CacheStore.key)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                with _get_session_with_tenant(self._tenant_id) as session:
                    row = session.execute(stmt).scalars().first()
                    if row is not None:
                        value = bytes(row.value) if row.value else b""
                        session.delete(row)
                        session.commit()
                        return (key.encode(), value)
            if time.monotonic() >= deadline:
                return None
            time.sleep(_BLPOP_POLL_INTERVAL)

    def _lock_id_for(self, name: str) -> int:
        h = hashlib.md5(
            f"{self._tenant_id}:{name}".encode(), usedforsecurity=False
        ).digest()
        return struct.unpack("q", h[:8])[0]


def cleanup_expired_cache_entries() -> None:
    """Delete rows whose ``expires_at`` is in the past."""
    CacheStore = _get_cache_store_model()
    with _get_session_with_current_tenant() as session:
        session.execute(
            delete(CacheStore).where(
                CacheStore.expires_at.is_not(None),
                CacheStore.expires_at < func.now(),
            )
        )
        session.commit()
