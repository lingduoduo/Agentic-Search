from __future__ import annotations

import re

from sqlalchemy import text

from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA
from shared_configs.configs import TENANT_ID_PREFIX

from src.backend.db.engine.sql_engine import get_session_with_shared_schema
from src.backend.db.engine.sql_engine import MULTI_TENANT
from src.backend.db.engine.sql_engine import SqlEngine

TENANT_ID_PATTERN = re.compile(
    rf"^{re.escape(TENANT_ID_PREFIX)}("
    r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    r"|i-[a-f0-9]+"
    r"|dev"
    r")$"
)


def validate_tenant_id(tenant_id: str) -> bool:
    """Validate tenant_id format. Critical for SQL injection prevention — schema names cannot be parameterized."""
    return bool(TENANT_ID_PATTERN.match(tenant_id))


def get_schemas_needing_migration(
    tenant_schemas: list[str], head_rev: str
) -> list[str]:
    """Return only schemas whose alembic version is not at head."""
    if not tenant_schemas:
        return []

    engine = SqlEngine.get_engine()

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _alembic_version_snapshot"))
        conn.execute(text("DROP TABLE IF EXISTS _tenant_schemas_input"))
        conn.execute(text("CREATE TEMP TABLE _tenant_schemas_input (schema_name text)"))
        conn.execute(
            text(
                "INSERT INTO _tenant_schemas_input (schema_name) SELECT unnest(CAST(:schemas AS text[]))"
            ),
            {"schemas": tenant_schemas},
        )
        conn.execute(
            text(
                "CREATE TEMP TABLE _alembic_version_snapshot (schema_name text, version_num text)"
            )
        )
        conn.execute(
            text("""
            DO $$
            DECLARE
                s        text;
                schemas  text[];
            BEGIN
                SELECT array_agg(schema_name) INTO schemas FROM _tenant_schemas_input;
                IF schemas IS NULL THEN RETURN; END IF;
                FOREACH s IN ARRAY schemas LOOP
                    BEGIN
                        EXECUTE format(
                            'INSERT INTO _alembic_version_snapshot
                             SELECT %L, version_num FROM %I.alembic_version',
                            s, s
                        );
                    EXCEPTION
                        WHEN undefined_table THEN NULL;
                        WHEN invalid_schema_name THEN NULL;
                    END;
                END LOOP;
            END;
            $$
        """)
        )

        rows = conn.execute(
            text("SELECT schema_name, version_num FROM _alembic_version_snapshot")
        )
        version_by_schema = {row[0]: row[1] for row in rows}

        conn.execute(text("DROP TABLE IF EXISTS _alembic_version_snapshot"))
        conn.execute(text("DROP TABLE IF EXISTS _tenant_schemas_input"))

    return [s for s in tenant_schemas if version_by_schema.get(s) != head_rev]


def get_all_tenant_ids() -> list[str]:
    """Return all tenant IDs. Returns [POSTGRES_DEFAULT_SCHEMA] in single-tenant mode."""
    if not MULTI_TENANT:
        return [POSTGRES_DEFAULT_SCHEMA]

    with get_session_with_shared_schema() as session:
        result = session.execute(
            text(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('pg_catalog', 'information_schema', :default_schema)"
            ),
            {"default_schema": POSTGRES_DEFAULT_SCHEMA},
        )
        tenant_ids = [row[0] for row in result]

    return [t for t in tenant_ids if t is None or validate_tenant_id(t)]
