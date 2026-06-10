from sqlalchemy import text

from src.backend.db.engine.async_sql_engine import get_sqlalchemy_async_engine
from src.backend.db.engine.sql_engine import get_sqlalchemy_engine


async def warm_up_connections(
    sync_connections_to_warm_up: int = 20, async_connections_to_warm_up: int = 20
) -> None:
    sync_engine = get_sqlalchemy_engine()
    sync_conns = [sync_engine.connect() for _ in range(sync_connections_to_warm_up)]
    for conn in sync_conns:
        conn.execute(text("SELECT 1"))
    for conn in sync_conns:
        conn.close()

    async_engine = get_sqlalchemy_async_engine()
    async_conns = [
        await async_engine.connect() for _ in range(async_connections_to_warm_up)
    ]
    for async_conn in async_conns:
        await async_conn.execute(text("SELECT 1"))
    for async_conn in async_conns:
        await async_conn.close()
