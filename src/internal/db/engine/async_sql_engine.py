from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from typing import AsyncContextManager

from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

from src.shared_configs.contextvars import get_current_tenant_id

from src.internal.db.engine.iam_auth import create_ssl_context_if_iam
from src.internal.db.engine.iam_auth import get_iam_auth_token
from src.internal.db.engine.sql_engine import ASYNC_DB_API
from src.internal.db.engine.sql_engine import build_connection_string
from src.internal.db.engine.sql_engine import is_valid_schema_name
from src.internal.db.engine.sql_engine import MULTI_TENANT
from src.internal.db.engine.sql_engine import POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE
from src.internal.db.engine.sql_engine import POSTGRES_HOST
from src.internal.db.engine.sql_engine import POSTGRES_POOL_PRE_PING
from src.internal.db.engine.sql_engine import POSTGRES_POOL_RECYCLE
from src.internal.db.engine.sql_engine import POSTGRES_PORT
from src.internal.db.engine.sql_engine import POSTGRES_USE_NULL_POOL
from src.internal.db.engine.sql_engine import POSTGRES_USER
from src.internal.db.engine.sql_engine import SqlEngine
from src.internal.db.engine.sql_engine import USE_IAM_AUTH

POSTGRES_API_SERVER_POOL_SIZE: int = int(
    os.environ.get("POSTGRES_API_SERVER_POOL_SIZE", "5")
)
POSTGRES_API_SERVER_POOL_OVERFLOW: int = int(
    os.environ.get("POSTGRES_API_SERVER_POOL_OVERFLOW", "10")
)
AWS_REGION_NAME: str = os.environ.get("AWS_REGION_NAME", "us-east-2")

_ASYNC_ENGINE: AsyncEngine | None = None


def get_sqlalchemy_async_engine() -> AsyncEngine:
    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is None:
        app_name = SqlEngine.get_app_name() + "_async"
        connection_string = build_connection_string(
            db_api=ASYNC_DB_API, use_iam_auth=USE_IAM_AUTH
        )

        connect_args: dict[str, Any] = {}
        if app_name:
            connect_args["server_settings"] = {"application_name": app_name}
        connect_args["ssl"] = create_ssl_context_if_iam()
        connect_args["statement_cache_size"] = 0

        engine_kwargs: dict[str, Any] = {
            "connect_args": connect_args,
            "pool_pre_ping": POSTGRES_POOL_PRE_PING,
            "pool_recycle": POSTGRES_POOL_RECYCLE,
        }

        if POSTGRES_USE_NULL_POOL:
            engine_kwargs["poolclass"] = pool.NullPool  # ty: ignore[invalid-assignment]
        else:
            engine_kwargs["pool_size"] = POSTGRES_API_SERVER_POOL_SIZE
            engine_kwargs["max_overflow"] = POSTGRES_API_SERVER_POOL_OVERFLOW

        _ASYNC_ENGINE = create_async_engine(connection_string, **engine_kwargs)

        if USE_IAM_AUTH:

            @event.listens_for(_ASYNC_ENGINE.sync_engine, "do_connect")
            def provide_iam_token_async(
                dialect: Any,  # noqa: ARG001
                conn_rec: Any,  # noqa: ARG001
                cargs: Any,  # noqa: ARG001
                cparams: Any,
            ) -> None:
                token = get_iam_auth_token(
                    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, AWS_REGION_NAME
                )
                cparams["password"] = token
                cparams["ssl"] = create_ssl_context_if_iam()

    return _ASYNC_ENGINE


async def reset_sqlalchemy_async_engine() -> None:
    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is not None:
        await _ASYNC_ENGINE.dispose()
        _ASYNC_ENGINE = None


async def get_async_session(
    tenant_id: str | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    """For use with FastAPI Depends on async endpoints."""
    if tenant_id is None:
        tenant_id = get_current_tenant_id() or POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE

    if not is_valid_schema_name(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    engine = get_sqlalchemy_async_engine()

    if not MULTI_TENANT and tenant_id == POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE:
        async with AsyncSession(bind=engine, expire_on_commit=False) as session:
            yield session
        return

    schema_translate_map = {None: tenant_id}
    async with engine.connect() as connection:
        connection = await connection.execution_options(
            schema_translate_map=schema_translate_map
        )
        async with AsyncSession(
            bind=connection, expire_on_commit=False
        ) as async_session:
            yield async_session


def get_async_session_context_manager(
    tenant_id: str | None = None,
) -> AsyncContextManager[AsyncSession]:
    return asynccontextmanager(get_async_session)(tenant_id)
