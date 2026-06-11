from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy import pool
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from shared_configs.contextvars import get_current_tenant_id

from src.internal.db.engine.iam_auth import provide_iam_token

logger = logging.getLogger(__name__)

# Postgres config from environment
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "password")
POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
DB_READONLY_USER: str = os.environ.get("DB_READONLY_USER", "")
DB_READONLY_PASSWORD: str = os.environ.get("DB_READONLY_PASSWORD", "")
POSTGRES_POOL_PRE_PING: bool = (
    os.environ.get("POSTGRES_POOL_PRE_PING", "true").lower() == "true"
)
POSTGRES_POOL_RECYCLE: int = int(os.environ.get("POSTGRES_POOL_RECYCLE", "300"))
POSTGRES_USE_NULL_POOL: bool = (
    os.environ.get("POSTGRES_USE_NULL_POOL", "false").lower() == "true"
)
POSTGRES_TCP_KEEPALIVES: bool = (
    os.environ.get("POSTGRES_TCP_KEEPALIVES", "true").lower() == "true"
)
POSTGRES_TCP_KEEPALIVES_IDLE: int = int(
    os.environ.get("POSTGRES_TCP_KEEPALIVES_IDLE", "30")
)
POSTGRES_TCP_KEEPALIVES_INTERVAL: int = int(
    os.environ.get("POSTGRES_TCP_KEEPALIVES_INTERVAL", "10")
)
POSTGRES_TCP_KEEPALIVES_COUNT: int = int(
    os.environ.get("POSTGRES_TCP_KEEPALIVES_COUNT", "6")
)
LOG_POSTGRES_LATENCY: bool = (
    os.environ.get("LOG_POSTGRES_LATENCY", "false").lower() == "true"
)
LOG_POSTGRES_CONN_COUNTS: bool = (
    os.environ.get("LOG_POSTGRES_CONN_COUNTS", "false").lower() == "true"
)
MULTI_TENANT: bool = os.environ.get("MULTI_TENANT", "false").lower() in (
    "1",
    "true",
    "yes",
)
USE_IAM_AUTH: bool = os.environ.get("USE_IAM_AUTH", "false").lower() == "true"
# The schema name used in single-tenant mode — equals POSTGRES_DEFAULT_SCHEMA ("public")
POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE: str = POSTGRES_DEFAULT_SCHEMA
POSTGRES_UNKNOWN_APP_NAME: str = "unknown"

SYNC_DB_API = "psycopg2"
ASYNC_DB_API = "asyncpg"

SCHEMA_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")


def is_valid_schema_name(name: str) -> bool:
    return SCHEMA_NAME_REGEX.match(name) is not None


def psycopg2_keepalive_connect_args() -> dict[str, int]:
    if not POSTGRES_TCP_KEEPALIVES:
        return {}
    return {
        "keepalives": 1,
        "keepalives_idle": POSTGRES_TCP_KEEPALIVES_IDLE,
        "keepalives_interval": POSTGRES_TCP_KEEPALIVES_INTERVAL,
        "keepalives_count": POSTGRES_TCP_KEEPALIVES_COUNT,
    }


def _merge_psycopg2_connect_args(
    extra_engine_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    caller_connect_args = extra_engine_kwargs.pop("connect_args", None) or {}
    merged: dict[str, Any] = {
        **psycopg2_keepalive_connect_args(),
        **caller_connect_args,
    }
    return merged or None


def build_connection_string(
    *,
    db_api: str = ASYNC_DB_API,
    user: str = POSTGRES_USER,
    password: str = POSTGRES_PASSWORD,
    host: str = POSTGRES_HOST,
    port: str = POSTGRES_PORT,
    db: str = POSTGRES_DB,
    app_name: str | None = None,
    use_iam_auth: bool = USE_IAM_AUTH,
    region: str = "us-west-2",  # noqa: ARG001
) -> str:
    if use_iam_auth:
        base_conn_str = f"postgresql+{db_api}://{user}@{host}:{port}/{db}"
    else:
        base_conn_str = f"postgresql+{db_api}://{user}:{password}@{host}:{port}/{db}"

    if app_name and db_api != "asyncpg":
        separator = "&" if "?" in base_conn_str else "?"
        return f"{base_conn_str}{separator}application_name={app_name}"
    return base_conn_str


if LOG_POSTGRES_LATENCY:

    @event.listens_for(Engine, "before_cursor_execute")
    def before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):  # noqa: ARG001
        conn.info["query_start_time"] = time.time()

    @event.listens_for(Engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        total_time = time.time() - conn.info["query_start_time"]
        if total_time > 0.1:
            logger.debug(
                "Query Complete: %s\n\nTotal Time: %.4f seconds", statement, total_time
            )


if LOG_POSTGRES_CONN_COUNTS:
    _checkout_count = 0
    _checkin_count = 0

    @event.listens_for(Engine, "checkout")
    def log_checkout(dbapi_connection, connection_record, connection_proxy):  # noqa: ARG001
        global _checkout_count
        _checkout_count += 1
        active = connection_proxy._pool.checkedout()
        idle = connection_proxy._pool.checkedin()
        size = connection_proxy._pool.size()
        logger.debug(
            "Connection Checkout — active=%s idle=%s pool_size=%s total=%s",
            active,
            idle,
            size,
            _checkout_count,
        )

    @event.listens_for(Engine, "checkin")
    def log_checkin(dbapi_connection, connection_record):  # noqa: ARG001
        global _checkin_count
        _checkin_count += 1
        logger.debug("Total connection checkins: %s", _checkin_count)


class SqlEngine:
    _engine: Engine | None = None
    _readonly_engine: Engine | None = None
    _lock: threading.Lock = threading.Lock()
    _readonly_lock: threading.Lock = threading.Lock()
    _app_name: str = POSTGRES_UNKNOWN_APP_NAME

    @classmethod
    def init_engine(
        cls,
        pool_size: int,
        max_overflow: int,
        app_name: str | None = None,  # noqa: ARG003
        db_api: str = SYNC_DB_API,
        use_iam: bool = USE_IAM_AUTH,
        connection_string: str | None = None,
        **extra_engine_kwargs: Any,
    ) -> None:
        with cls._lock:
            if cls._engine:
                return

            if not connection_string:
                connection_string = build_connection_string(
                    db_api=db_api,
                    app_name=cls._app_name + "_sync",
                    use_iam_auth=use_iam,
                )

            final_engine_kwargs: dict[str, Any] = {}

            if db_api == SYNC_DB_API:
                merged_connect_args = _merge_psycopg2_connect_args(extra_engine_kwargs)
                if merged_connect_args is not None:
                    final_engine_kwargs["connect_args"] = merged_connect_args

            if POSTGRES_USE_NULL_POOL:
                final_engine_kwargs.update(extra_engine_kwargs)
                final_engine_kwargs["poolclass"] = pool.NullPool
                final_engine_kwargs.pop("pool_size", None)
                final_engine_kwargs.pop("max_overflow", None)
            else:
                final_engine_kwargs["pool_size"] = pool_size
                final_engine_kwargs["max_overflow"] = max_overflow
                final_engine_kwargs["pool_pre_ping"] = POSTGRES_POOL_PRE_PING
                final_engine_kwargs["pool_recycle"] = POSTGRES_POOL_RECYCLE
                final_engine_kwargs.update(extra_engine_kwargs)

            logger.info("Creating engine with kwargs: %s", final_engine_kwargs)
            engine = create_engine(connection_string, **final_engine_kwargs)

            if use_iam:
                event.listen(engine, "do_connect", provide_iam_token)

            cls._engine = engine

    @classmethod
    def init_readonly_engine(
        cls,
        pool_size: int,
        max_overflow: int,
        **extra_engine_kwargs: Any,
    ) -> None:
        with cls._readonly_lock:
            if cls._readonly_engine:
                return

            if not DB_READONLY_USER or not DB_READONLY_PASSWORD:
                raise ValueError(
                    "DB_READONLY_USER and DB_READONLY_PASSWORD must be set"
                )

            connection_string = build_connection_string(
                user=DB_READONLY_USER,
                password=DB_READONLY_PASSWORD,
                use_iam_auth=False,
                db_api=SYNC_DB_API,
            )

            final_engine_kwargs: dict[str, Any] = {}
            merged_connect_args = _merge_psycopg2_connect_args(extra_engine_kwargs)
            if merged_connect_args is not None:
                final_engine_kwargs["connect_args"] = merged_connect_args

            if POSTGRES_USE_NULL_POOL:
                final_engine_kwargs.update(extra_engine_kwargs)
                final_engine_kwargs["poolclass"] = pool.NullPool
                final_engine_kwargs.pop("pool_size", None)
                final_engine_kwargs.pop("max_overflow", None)
            else:
                final_engine_kwargs["pool_size"] = pool_size
                final_engine_kwargs["max_overflow"] = max_overflow
                final_engine_kwargs["pool_pre_ping"] = POSTGRES_POOL_PRE_PING
                final_engine_kwargs["pool_recycle"] = POSTGRES_POOL_RECYCLE
                final_engine_kwargs.update(extra_engine_kwargs)

            logger.info("Creating readonly engine with kwargs: %s", final_engine_kwargs)
            engine = create_engine(connection_string, **final_engine_kwargs)

            if USE_IAM_AUTH:
                event.listen(engine, "do_connect", provide_iam_token)

            cls._readonly_engine = engine

    @classmethod
    def get_engine(cls) -> Engine:
        if not cls._engine:
            raise RuntimeError("Engine not initialized. Call init_engine first.")
        return cls._engine

    @classmethod
    def get_readonly_engine(cls) -> Engine:
        if not cls._readonly_engine:
            raise RuntimeError(
                "Readonly engine not initialized. Call init_readonly_engine first."
            )
        return cls._readonly_engine

    @classmethod
    def set_app_name(cls, app_name: str) -> None:
        cls._app_name = app_name

    @classmethod
    def get_app_name(cls) -> str:
        return cls._app_name or ""

    @classmethod
    def reset_engine(cls) -> None:
        with cls._lock:
            if cls._engine:
                cls._engine.dispose()
                cls._engine = None

    @classmethod
    def reset_readonly_engine(cls) -> None:
        with cls._readonly_lock:
            if cls._readonly_engine:
                cls._readonly_engine.dispose()
                cls._readonly_engine = None

    @classmethod
    @contextmanager
    def scoped_engine(cls, **init_kwargs: Any) -> Generator[None, None, None]:
        cls.init_engine(**init_kwargs)
        try:
            yield
        finally:
            cls.reset_engine()


def get_sqlalchemy_engine() -> Engine:
    return SqlEngine.get_engine()


def get_readonly_sqlalchemy_engine() -> Engine:
    return SqlEngine.get_readonly_engine()


def _safe_close_session(session: Session) -> None:
    try:
        session.close()
    except DBAPIError:
        logger.warning(
            "DB connection lost during session cleanup — will be recycled by the pool."
        )


@contextmanager
def get_session_with_tenant(*, tenant_id: str) -> Generator[Session, None, None]:
    engine = get_sqlalchemy_engine()

    if not is_valid_schema_name(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not MULTI_TENANT and tenant_id == POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE:
        session = Session(bind=engine, expire_on_commit=False)
        try:
            yield session
        finally:
            _safe_close_session(session)
        return

    schema_translate_map = {None: tenant_id}
    with engine.connect().execution_options(
        schema_translate_map=schema_translate_map
    ) as connection:
        session = Session(bind=connection, expire_on_commit=False)
        try:
            yield session
        finally:
            _safe_close_session(session)


@contextmanager
def get_session_with_current_tenant() -> Generator[Session, None, None]:
    tenant_id = get_current_tenant_id() or POSTGRES_DEFAULT_SCHEMA
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        yield session


@contextmanager
def get_session_with_current_tenant_if_none(
    session: Session | None,
) -> Generator[Session, None, None]:
    if session is None:
        with get_session_with_current_tenant() as s:
            yield s
    else:
        yield session


@contextmanager
def get_session_with_shared_schema() -> Generator[Session, None, None]:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA)
    try:
        with get_session_with_tenant(tenant_id=POSTGRES_DEFAULT_SCHEMA) as session:
            yield session
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def get_session() -> Generator[Session, None, None]:
    """For use with FastAPI Depends."""
    tenant_id = get_current_tenant_id() or POSTGRES_DEFAULT_SCHEMA

    if tenant_id == POSTGRES_DEFAULT_SCHEMA and MULTI_TENANT:
        raise HTTPException(status_code=401, detail="User must authenticate")

    if not is_valid_schema_name(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    with get_session_with_current_tenant() as db_session:
        yield db_session


@contextmanager
def get_db_readonly_user_session_with_current_tenant() -> Generator[
    Session, None, None
]:
    tenant_id = get_current_tenant_id() or POSTGRES_DEFAULT_SCHEMA
    readonly_engine = get_readonly_sqlalchemy_engine()

    if not is_valid_schema_name(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not MULTI_TENANT and tenant_id == POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE:
        with Session(readonly_engine, expire_on_commit=False) as session:
            yield session
        return

    schema_translate_map = {None: tenant_id}
    with readonly_engine.connect().execution_options(
        schema_translate_map=schema_translate_map
    ) as connection:
        with Session(bind=connection, expire_on_commit=False) as session:
            yield session
