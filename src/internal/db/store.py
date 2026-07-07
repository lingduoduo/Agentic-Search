"""SQLite-backed metadata store for local Agentic Search workflows."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    ChatMessageRecord,
    ChatSessionRecord,
    ConnectorConfig,
    DocumentPermission,
    GroupRecord,
    HookRecord,
    IndexAttemptRecord,
    IndexAttemptStatus,
    StoredDocument,
    UserMemoryRecord,
    UserRecord,
)
from src.internal.feedback.runtime import deterministic_capture


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        msg = "metadata JSON fields must decode to an object"
        raise ValueError(msg)
    return loaded


class _NullSession:
    """Null-object DB session for local (no-SQLAlchemy) mode.

    All mutating operations are no-ops; reads return None so callers can
    guard with ``if obj is not None:`` rather than crashing.
    """

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def expunge_all(self) -> None:
        pass

    def get(self, model, pk):
        return None

    def add(self, obj) -> None:
        pass

    def flush(self) -> None:
        pass

    def delete(self, obj) -> None:
        pass


@contextlib.contextmanager
def get_session_with_current_tenant():
    """Yield a null-object session (no real DB in local mode)."""
    yield _NullSession()


class AgenticSearchStore:
    """Small SQLite repository for connector, document, ACL, chat, and indexing state."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AgenticSearchStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS connector_configs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                config_json TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                contents TEXT NOT NULL,
                url TEXT,
                connector_id TEXT REFERENCES connector_configs(id) ON DELETE SET NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                name TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS group_members (
                group_id TEXT NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY (group_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS document_permissions (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                principal_type TEXT NOT NULL,
                principal_id TEXT NOT NULL DEFAULT '',
                access TEXT NOT NULL DEFAULT 'read',
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, principal_type, principal_id, access)
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                title TEXT,
                metadata_json TEXT NOT NULL,
                llm_name TEXT,
                persona_id TEXT,
                flow_type TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hooks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                hook_point TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                api_key TEXT,
                fail_strategy TEXT NOT NULL DEFAULT 'soft',
                timeout_seconds REAL NOT NULL DEFAULT 5.0,
                is_active INTEGER NOT NULL DEFAULT 1,
                is_reachable INTEGER,
                creator_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_attempts (
                id TEXT PRIMARY KEY,
                connector_id TEXT REFERENCES connector_configs(id) ON DELETE SET NULL,
                status TEXT NOT NULL,
                total_documents INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                error TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_reports (
                report_name TEXT PRIMARY KEY,
                requestor_id TEXT,
                time_created TEXT NOT NULL,
                period_from TEXT,
                period_to TEXT,
                data BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS token_rate_limits (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                scope_id TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                token_budget INTEGER NOT NULL,
                period_hours INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS standard_answer_categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS standard_answers (
                id TEXT PRIMARY KEY,
                keyword TEXT NOT NULL,
                answer TEXT NOT NULL,
                match_regex INTEGER NOT NULL DEFAULT 0,
                match_any_keywords INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS standard_answer_category_links (
                answer_id TEXT NOT NULL,
                category_id TEXT NOT NULL,
                PRIMARY KEY (answer_id, category_id)
            );

            CREATE TABLE IF NOT EXISTS scim_tokens (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                token_display TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used_at TEXT
            );

            CREATE TABLE IF NOT EXISTS scim_user_mappings (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE,
                external_id TEXT,
                scim_username TEXT,
                department TEXT,
                manager TEXT,
                given_name TEXT,
                family_name TEXT,
                scim_emails_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scim_group_mappings (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL UNIQUE,
                external_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_is_active (
                user_id TEXT PRIMARY KEY,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_documents_connector_updated
                ON documents(connector_id, updated_at DESC, id);
            CREATE INDEX IF NOT EXISTS idx_documents_updated
                ON documents(updated_at DESC, id);
            CREATE INDEX IF NOT EXISTS idx_group_members_user
                ON group_members(user_id, group_id);
            CREATE INDEX IF NOT EXISTS idx_document_permissions_principal
                ON document_permissions(principal_type, principal_id, document_id);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at DESC, id);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_llm
                ON chat_sessions(llm_name, created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_persona
                ON chat_sessions(persona_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_flow
                ON chat_sessions(flow_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                ON chat_messages(session_id, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_index_attempts_connector_updated
                ON index_attempts(connector_id, updated_at DESC, id);
            CREATE TABLE IF NOT EXISTS user_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_memories_user_active
                ON user_memories(user_id, is_active, created_at, id);

            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                signal TEXT NOT NULL CHECK (signal IN ('thumbs_up', 'thumbs_down')),
                metadata_json TEXT NOT NULL DEFAULT '{}',
                parent_feedback_id TEXT,
                correlation_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_created
                ON retrieval_feedback(created_at);
            """
        )
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Idempotent migrations for databases created before column additions."""
        for col, ddl in [
            ("llm_name", "ALTER TABLE chat_sessions ADD COLUMN llm_name TEXT"),
            ("persona_id", "ALTER TABLE chat_sessions ADD COLUMN persona_id TEXT"),
            ("flow_type", "ALTER TABLE chat_sessions ADD COLUMN flow_type TEXT"),
        ]:
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        for idx_ddl in [
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_llm ON chat_sessions(llm_name, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_persona ON chat_sessions(persona_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_flow ON chat_sessions(flow_type, created_at)",
        ]:
            self._conn.execute(idx_ddl)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_memories_user_active
                ON user_memories(user_id, is_active, created_at, id);
            """
        )
        for col, ddl in [
            (
                "metadata_json",
                "ALTER TABLE retrieval_feedback ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
            ),
            (
                "parent_feedback_id",
                "ALTER TABLE retrieval_feedback ADD COLUMN parent_feedback_id TEXT",
            ),
            (
                "correlation_id",
                "ALTER TABLE retrieval_feedback ADD COLUMN correlation_id TEXT",
            ),
        ]:
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def upsert_connector(self, connector: ConnectorConfig) -> ConnectorConfig:
        now = _now()
        created_at = connector.created_at or self._created_at(
            "connector_configs", connector.id, now
        )
        record = ConnectorConfig(
            id=connector.id,
            name=connector.name,
            source=connector.source,
            config=dict(connector.config),
            enabled=connector.enabled,
            metadata=dict(connector.metadata),
            created_at=created_at,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO connector_configs (
                id, name, source, config_json, enabled, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                source = excluded.source,
                config_json = excluded.config_json,
                enabled = excluded.enabled,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                record.id,
                record.name,
                record.source,
                _json_dumps(record.config),
                int(record.enabled),
                _json_dumps(record.metadata),
                record.created_at,
                record.updated_at,
            ),
        )
        self._conn.commit()
        return record

    def get_connector(self, connector_id: str) -> ConnectorConfig | None:
        row = self._conn.execute(
            "SELECT * FROM connector_configs WHERE id = ?", (connector_id,)
        ).fetchone()
        return self._row_to_connector(row) if row else None

    def list_connectors(self, *, enabled: bool | None = None) -> list[ConnectorConfig]:
        query = "SELECT * FROM connector_configs"
        params: tuple[object, ...] = ()
        if enabled is not None:
            query += " WHERE enabled = ?"
            params = (int(enabled),)
        query += " ORDER BY name"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_connector(row) for row in rows]

    def upsert_document(self, document: StoredDocument) -> StoredDocument:
        now = _now()
        created_at = document.created_at or self._created_at(
            "documents", document.id, now
        )
        record = StoredDocument(
            id=document.id,
            title=document.title,
            contents=document.contents,
            url=document.url,
            connector_id=document.connector_id,
            metadata=dict(document.metadata),
            created_at=created_at,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO documents (
                id, title, contents, url, connector_id, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                contents = excluded.contents,
                url = excluded.url,
                connector_id = excluded.connector_id,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                record.id,
                record.title,
                record.contents,
                record.url,
                record.connector_id,
                _json_dumps(record.metadata),
                record.created_at,
                record.updated_at,
            ),
        )
        self._conn.commit()
        return record

    def upsert_documents_bulk(
        self, documents: list[StoredDocument]
    ) -> list[StoredDocument]:
        """Upsert multiple documents in a single transaction."""
        if not documents:
            return []
        now = _now()
        doc_ids = [d.id for d in documents]
        placeholders = ", ".join("?" * len(doc_ids))
        existing_created_at: dict[str, str] = {
            row["id"]: str(row["created_at"])
            for row in self._conn.execute(
                f"SELECT id, created_at FROM documents WHERE id IN ({placeholders})",
                tuple(doc_ids),
            ).fetchall()
        }
        records: list[StoredDocument] = []
        params: list[tuple] = []
        for document in documents:
            fetched = existing_created_at.get(document.id)
            created_at = (
                fetched if fetched is not None else (document.created_at or now)
            )
            record = StoredDocument(
                id=document.id,
                title=document.title,
                contents=document.contents,
                url=document.url,
                connector_id=document.connector_id,
                metadata=dict(document.metadata),
                created_at=created_at,
                updated_at=now,
            )
            records.append(record)
            params.append(
                (
                    record.id,
                    record.title,
                    record.contents,
                    record.url,
                    record.connector_id,
                    _json_dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
            )
        try:
            self._conn.executemany(
                """
                INSERT INTO documents (
                    id, title, contents, url, connector_id, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    contents = excluded.contents,
                    url = excluded.url,
                    connector_id = excluded.connector_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                params,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return records

    def get_document(self, document_id: str) -> StoredDocument | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(
        self, *, connector_id: str | None = None, limit: int | None = None
    ) -> list[StoredDocument]:
        query = "SELECT * FROM documents"
        params: list[object] = []
        if connector_id is not None:
            query += " WHERE connector_id = ?"
            params.append(connector_id)
        query += " ORDER BY updated_at DESC, id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_document(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE id = ?", (document_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_documents_bulk(self, doc_ids: list[str]) -> int:
        """Delete multiple documents by ID in a single transaction. Returns count deleted."""
        if not doc_ids:
            return 0
        placeholders = ", ".join("?" * len(doc_ids))
        try:
            cursor = self._conn.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})",
                tuple(doc_ids),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return cursor.rowcount

    def upsert_user(self, user: UserRecord) -> UserRecord:
        now = _now()
        created_at = user.created_at or self._created_at("users", user.id, now)
        record = UserRecord(
            id=user.id,
            email=user.email,
            name=user.name,
            metadata=dict(user.metadata),
            created_at=created_at,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO users (id, email, name, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                record.id,
                record.email,
                record.name,
                _json_dumps(record.metadata),
                record.created_at,
                record.updated_at,
            ),
        )
        self._conn.commit()
        return record

    def get_user(self, user_id: str) -> UserRecord | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self) -> list[UserRecord]:
        rows = self._conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [self._row_to_user(row) for row in rows]

    def delete_user(self, user_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def upsert_group(self, group: GroupRecord) -> GroupRecord:
        now = _now()
        created_at = group.created_at or self._created_at("groups", group.id, now)
        self._conn.execute(
            """
            INSERT INTO groups (id, name, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                group.id,
                group.name,
                _json_dumps(group.metadata),
                created_at,
                now,
            ),
        )
        self._conn.execute("DELETE FROM group_members WHERE group_id = ?", (group.id,))
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO group_members (group_id, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            [(group.id, user_id, now) for user_id in group.user_ids],
        )
        self._conn.commit()
        return self.get_group(group.id) or GroupRecord(id=group.id, name=group.name)

    def list_groups(self) -> list[GroupRecord]:
        """Return all groups ordered by name."""
        rows = self._conn.execute("SELECT * FROM groups ORDER BY name").fetchall()
        return self._rows_to_groups(rows)

    def delete_group(self, group_id: str) -> bool:
        """Delete *group_id* and its memberships. Returns True if it existed."""
        cursor = self._conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def remove_user_from_group(self, user_id: str, group_id: str) -> None:
        """Remove *user_id* from *group_id*. No-op if the membership doesn't exist."""
        self._conn.execute(
            "DELETE FROM group_members WHERE group_id = ? AND user_id = ?",
            (group_id, user_id),
        )
        self._conn.commit()

    def add_user_to_group(self, user_id: str, group_id: str) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO group_members (group_id, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            (group_id, user_id, _now()),
        )
        self._conn.commit()

    def get_group(self, group_id: str) -> GroupRecord | None:
        row = self._conn.execute(
            "SELECT * FROM groups WHERE id = ?", (group_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_group(row)

    def list_groups_for_user(self, user_id: str) -> list[GroupRecord]:
        rows = self._conn.execute(
            """
            SELECT g.* FROM groups g
            JOIN group_members gm ON gm.group_id = g.id
            WHERE gm.user_id = ?
            ORDER BY g.name
            """,
            (user_id,),
        ).fetchall()
        return self._rows_to_groups(rows)

    def list_group_ids_for_user(self, user_id: str) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT group_id FROM group_members
            WHERE user_id = ?
            ORDER BY group_id
            """,
            (user_id,),
        ).fetchall()
        return [row["group_id"] for row in rows]

    def grant_document_access(
        self, permission: DocumentPermission
    ) -> DocumentPermission:
        principal_id = permission.principal_id or ""
        now = permission.created_at or _now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO document_permissions (
                document_id, principal_type, principal_id, access, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                permission.document_id,
                permission.principal_type,
                principal_id,
                permission.access,
                now,
            ),
        )
        self._conn.commit()
        return DocumentPermission(
            document_id=permission.document_id,
            principal_type=permission.principal_type,
            principal_id=principal_id or None,
            access=permission.access,
            created_at=now,
        )

    def grant_document_access_bulk(self, permissions: list[DocumentPermission]) -> int:
        """Write multiple permission grants in a single transaction. Returns count written."""
        if not permissions:
            return 0
        now = _now()
        params = [
            (
                p.document_id,
                p.principal_type,
                p.principal_id or "",
                p.access,
                p.created_at or now,
            )
            for p in permissions
        ]
        try:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO document_permissions (
                    document_id, principal_type, principal_id, access, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                params,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        # INSERT OR REPLACE always writes one row per entry; len(params) == rows written.
        return len(params)

    def get_document_permissions(self, document_id: str) -> list[DocumentPermission]:
        rows = self._conn.execute(
            """
            SELECT * FROM document_permissions
            WHERE document_id = ?
            ORDER BY principal_type, principal_id, access
            """,
            (document_id,),
        ).fetchall()
        return [self._row_to_permission(row) for row in rows]

    def documents_visible_to_user(
        self,
        user_id: str | None,
        *,
        group_ids: Iterable[str] | None = None,
    ) -> list[StoredDocument]:
        groups = set(group_ids or [])
        if user_id is not None:
            groups.update(self.list_group_ids_for_user(user_id))
        clauses = ["p.principal_type = 'public'"]
        params: list[object] = []
        if user_id is not None:
            clauses.append("(p.principal_type = 'user' AND p.principal_id = ?)")
            params.append(user_id)
        if groups:
            placeholders = ", ".join("?" for _ in groups)
            clauses.append(
                f"(p.principal_type = 'group' AND p.principal_id IN ({placeholders}))"
            )
            params.extend(sorted(groups))
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT d.* FROM documents d
            JOIN document_permissions p ON p.document_id = d.id
            WHERE {" OR ".join(clauses)}
            ORDER BY d.updated_at DESC, d.id
            """,
            tuple(params),
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def create_chat_session(
        self,
        *,
        user_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ChatSessionRecord:
        now = _now()
        record = ChatSessionRecord(
            id=session_id or _new_id("session"),
            user_id=user_id,
            title=title,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO chat_sessions (
                id, user_id, title, metadata_json,
                llm_name, persona_id, flow_type,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.user_id,
                record.title,
                _json_dumps(record.metadata),
                record.metadata.get("llm_name"),
                str(record.metadata["persona_id"])
                if record.metadata.get("persona_id") is not None
                else None,
                record.metadata.get("flow_type"),
                record.created_at,
                record.updated_at,
            ),
        )
        self._conn.commit()
        return record

    def get_chat_session(self, session_id: str) -> ChatSessionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._row_to_chat_session(row) if row else None

    def add_chat_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> ChatMessageRecord:
        now = _now()
        record = ChatMessageRecord(
            id=message_id or _new_id("message"),
            session_id=session_id,
            role=role,
            content=content,
            metadata=dict(metadata or {}),
            created_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, role, content, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.role,
                record.content,
                _json_dumps(record.metadata),
                record.created_at,
            ),
        )
        self._conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        self._conn.commit()
        return record

    def list_chat_messages(self, session_id: str) -> list[ChatMessageRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at, id
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_chat_message(row) for row in rows]

    def get_user_query_texts(self, limit: int | None = None) -> list[str]:
        """Distinct non-empty user query texts, most recently asked first.

        Ordered by each distinct query's most recent occurrence (for building a
        distillation corpus). Ties broken by content for deterministic output.
        """
        sql = (
            "SELECT content FROM chat_messages "
            "WHERE role = 'user' AND TRIM(content) != '' "
            "GROUP BY content ORDER BY MAX(created_at) DESC, content ASC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [str(row["content"]) for row in rows]

    def create_index_attempt(
        self,
        *,
        connector_id: str | None = None,
        status: IndexAttemptStatus = "not_started",
        total_documents: int = 0,
        total_chunks: int = 0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        attempt_id: str | None = None,
    ) -> IndexAttemptRecord:
        now = _now()
        record = IndexAttemptRecord(
            id=attempt_id or _new_id("index"),
            connector_id=connector_id,
            status=status,
            total_documents=total_documents,
            total_chunks=total_chunks,
            error=error,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO index_attempts (
                id, connector_id, status, total_documents, total_chunks,
                error, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.connector_id,
                record.status,
                record.total_documents,
                record.total_chunks,
                record.error,
                _json_dumps(record.metadata),
                record.created_at,
                record.updated_at,
            ),
        )
        self._conn.commit()
        return record

    def update_index_attempt(
        self,
        attempt_id: str,
        *,
        status: IndexAttemptStatus | None = None,
        total_documents: int | None = None,
        total_chunks: int | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexAttemptRecord:
        existing = self.get_index_attempt(attempt_id)
        if existing is None:
            msg = f"Index attempt {attempt_id!r} does not exist"
            raise KeyError(msg)
        updated = IndexAttemptRecord(
            id=existing.id,
            connector_id=existing.connector_id,
            status=status or existing.status,
            total_documents=(
                existing.total_documents if total_documents is None else total_documents
            ),
            total_chunks=existing.total_chunks
            if total_chunks is None
            else total_chunks,
            error=existing.error if error is None else error,
            metadata=existing.metadata if metadata is None else dict(metadata),
            created_at=existing.created_at,
            updated_at=_now(),
        )
        self._conn.execute(
            """
            UPDATE index_attempts
            SET status = ?, total_documents = ?, total_chunks = ?, error = ?,
                metadata_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                updated.status,
                updated.total_documents,
                updated.total_chunks,
                updated.error,
                _json_dumps(updated.metadata),
                updated.updated_at,
                updated.id,
            ),
        )
        self._conn.commit()
        return updated

    def get_index_attempt(self, attempt_id: str) -> IndexAttemptRecord | None:
        row = self._conn.execute(
            "SELECT * FROM index_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        return self._row_to_index_attempt(row) if row else None

    def list_index_attempts(
        self, *, connector_id: str | None = None
    ) -> list[IndexAttemptRecord]:
        query = "SELECT * FROM index_attempts"
        params: tuple[object, ...] = ()
        if connector_id is not None:
            query += " WHERE connector_id = ?"
            params = (connector_id,)
        query += " ORDER BY created_at DESC, id"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_index_attempt(row) for row in rows]

    def delete_connector(self, connector_id: str) -> bool:
        """Delete a connector and all its associated documents and index attempts.

        Returns True if the connector existed and was deleted.
        """
        if (
            self._conn.execute(
                "SELECT 1 FROM connector_configs WHERE id = ?", (connector_id,)
            ).fetchone()
            is None
        ):
            return False
        self._conn.execute(
            "DELETE FROM index_attempts WHERE connector_id = ?", (connector_id,)
        )
        self._conn.execute(
            "DELETE FROM documents WHERE connector_id = ?", (connector_id,)
        )
        self._conn.execute(
            "DELETE FROM connector_configs WHERE id = ?", (connector_id,)
        )
        self._conn.commit()
        return True

    def delete_old_index_attempts(self, connector_id: str, keep_last_n: int) -> int:
        """Delete all but the most recent *keep_last_n* attempts for *connector_id*.

        Returns the number of rows deleted.
        """
        rows = self._conn.execute(
            "SELECT id FROM index_attempts WHERE connector_id = ? ORDER BY created_at DESC, id",
            (connector_id,),
        ).fetchall()
        to_delete = [row["id"] for row in rows[keep_last_n:]]
        if not to_delete:
            return 0
        placeholders = ", ".join("?" * len(to_delete))
        cursor = self._conn.execute(
            f"DELETE FROM index_attempts WHERE id IN ({placeholders})",
            tuple(to_delete),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_stale_index_attempts(
        self,
        *,
        older_than_days: int,
        statuses: tuple[str, ...] = ("success", "failed"),
    ) -> int:
        """Delete finished attempts whose *updated_at* is older than *older_than_days*.

        Only rows matching *statuses* are considered; defaults to completed states.
        Returns the number of rows deleted.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=older_than_days)
        ).isoformat()
        placeholders = ", ".join("?" * len(statuses))
        cursor = self._conn.execute(
            f"DELETE FROM index_attempts WHERE status IN ({placeholders}) AND updated_at < ?",
            (*statuses, cutoff),
        )
        self._conn.commit()
        return cursor.rowcount

    def _created_at(self, table: str, record_id: str, default: str) -> str:
        row = self._conn.execute(
            f"SELECT created_at FROM {table} WHERE id = ?", (record_id,)
        ).fetchone()
        return str(row["created_at"]) if row else default

    def update_chat_session_title(self, session_id: str, title: str) -> bool:
        now = _now()
        cursor = self._conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_chat_session(self, session_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: int = 100,
        filter_days: int | None = None,
    ) -> list[ChatSessionRecord]:
        """Return up to *limit* sessions for *user_id*, newest first.

        If *filter_days* is given only sessions created in the last N days
        are returned.
        """
        if filter_days is not None:
            from datetime import datetime, timedelta, timezone

            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=filter_days)
            ).isoformat()
            rows = self._conn.execute(
                """
                SELECT * FROM chat_sessions
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, cutoff, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM chat_sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_chat_session(row) for row in rows]

    def query_session_analytics(
        self,
        start: str,
        end: str,
    ) -> list[tuple[str, int, int]]:
        """Return ``(day, session_count, message_count)`` rows between start and end.

        ``start`` and ``end`` must be UTC ISO-format strings (as produced by
        ``_now()``). The interval is ``[start, end)``.
        """
        rows = self._conn.execute(
            """
            SELECT
                date(s.created_at) AS day,
                COUNT(DISTINCT s.id) AS sessions,
                COUNT(m.id) AS messages
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.created_at >= ? AND s.created_at < ?
            GROUP BY day
            ORDER BY day
            """,
            (start, end),
        ).fetchall()
        return [(row["day"], row["sessions"], row["messages"]) for row in rows]

    def user_session_analytics(
        self,
        start: str,
        end: str,
    ) -> list[tuple[str, int]]:
        """Return ``(day, unique_active_users)`` rows between start and end.

        Only sessions with a non-NULL ``user_id`` are counted.
        """
        rows = self._conn.execute(
            """
            SELECT
                date(created_at) AS day,
                COUNT(DISTINCT user_id) AS active_users
            FROM chat_sessions
            WHERE created_at >= ? AND created_at < ?
              AND user_id IS NOT NULL
            GROUP BY day
            ORDER BY day
            """,
            (start, end),
        ).fetchall()
        return [(row["day"], row["active_users"]) for row in rows]

    def query_analytics_by_llm(
        self,
        start: str,
        end: str,
    ) -> list[tuple[str, int]]:
        """Return ``(llm_name, session_count)`` totals between start and end."""
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(llm_name, 'unknown') AS llm,
                COUNT(DISTINCT id) AS cnt
            FROM chat_sessions
            WHERE created_at >= ? AND created_at < ?
            GROUP BY llm
            ORDER BY cnt DESC
            """,
            (start, end),
        ).fetchall()
        return [(row["llm"], row["cnt"]) for row in rows]

    def query_analytics_by_persona(
        self,
        start: str,
        end: str,
    ) -> list[tuple[str, int]]:
        """Return ``(persona_id, session_count)`` totals between start and end."""
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(persona_id, 'default') AS persona,
                COUNT(DISTINCT id) AS cnt
            FROM chat_sessions
            WHERE created_at >= ? AND created_at < ?
            GROUP BY persona
            ORDER BY cnt DESC
            """,
            (start, end),
        ).fetchall()
        return [(row["persona"], row["cnt"]) for row in rows]

    def query_analytics_by_flow(
        self,
        start: str,
        end: str,
    ) -> list[tuple[str, int]]:
        """Return ``(flow_type, session_count)`` totals between start and end."""
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(flow_type, 'chat') AS flow,
                COUNT(DISTINCT id) AS cnt
            FROM chat_sessions
            WHERE created_at >= ? AND created_at < ?
            GROUP BY flow
            ORDER BY cnt DESC
            """,
            (start, end),
        ).fetchall()
        return [(row["flow"], row["cnt"]) for row in rows]

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def upsert_hook(self, hook: HookRecord) -> HookRecord:
        now = _now()
        created_at = hook.created_at or self._created_at("hooks", hook.id, now)
        self._conn.execute(
            """
            INSERT INTO hooks (
                id, name, hook_point, endpoint_url, api_key,
                fail_strategy, timeout_seconds, is_active, is_reachable,
                creator_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                hook_point = excluded.hook_point,
                endpoint_url = excluded.endpoint_url,
                api_key = excluded.api_key,
                fail_strategy = excluded.fail_strategy,
                timeout_seconds = excluded.timeout_seconds,
                is_active = excluded.is_active,
                is_reachable = excluded.is_reachable,
                creator_id = excluded.creator_id,
                updated_at = excluded.updated_at
            """,
            (
                hook.id,
                hook.name,
                hook.hook_point,
                hook.endpoint_url,
                hook.api_key,
                hook.fail_strategy,
                hook.timeout_seconds,
                int(hook.is_active),
                None if hook.is_reachable is None else int(hook.is_reachable),
                hook.creator_id,
                created_at,
                now,
            ),
        )
        self._conn.commit()
        return self.get_hook(hook.id) or hook

    def get_hook(self, hook_id: str) -> HookRecord | None:
        row = self._conn.execute(
            "SELECT * FROM hooks WHERE id = ?", (hook_id,)
        ).fetchone()
        return self._row_to_hook(row) if row else None

    def list_hooks(self) -> list[HookRecord]:
        rows = self._conn.execute(
            "SELECT * FROM hooks ORDER BY created_at DESC, id"
        ).fetchall()
        return [self._row_to_hook(row) for row in rows]

    def delete_hook(self, hook_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM hooks WHERE id = ?", (hook_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_hook(self, row: sqlite3.Row) -> HookRecord:
        is_reachable_raw = row["is_reachable"]
        return HookRecord(
            id=row["id"],
            name=row["name"],
            hook_point=row["hook_point"],
            endpoint_url=row["endpoint_url"],
            api_key=row["api_key"],
            fail_strategy=row["fail_strategy"],
            timeout_seconds=float(row["timeout_seconds"]),
            is_active=bool(row["is_active"]),
            is_reachable=(None if is_reachable_raw is None else bool(is_reachable_raw)),
            creator_id=row["creator_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_connector(self, row: sqlite3.Row) -> ConnectorConfig:
        return ConnectorConfig(
            id=row["id"],
            name=row["name"],
            source=row["source"],
            config=_json_loads(row["config_json"]),
            enabled=bool(row["enabled"]),
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_document(self, row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            id=row["id"],
            title=row["title"],
            contents=row["contents"],
            url=row["url"],
            connector_id=row["connector_id"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_user(self, row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_group(self, row: sqlite3.Row) -> GroupRecord:
        members_by_group = self._members_by_group_id([row["id"]])
        return self._row_to_group_with_members(row, members_by_group[row["id"]])

    def _rows_to_groups(self, rows: list[sqlite3.Row]) -> list[GroupRecord]:
        if not rows:
            return []
        members_by_group = self._members_by_group_id([row["id"] for row in rows])
        return [
            self._row_to_group_with_members(row, members_by_group[row["id"]])
            for row in rows
        ]

    def _members_by_group_id(self, group_ids: Iterable[str]) -> dict[str, list[str]]:
        ids = list(dict.fromkeys(group_ids))
        members_by_group: dict[str, list[str]] = defaultdict(list)
        if not ids:
            return members_by_group
        placeholders = ", ".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT group_id, user_id FROM group_members
            WHERE group_id IN ({placeholders})
            ORDER BY group_id, user_id
            """,
            tuple(ids),
        ).fetchall()
        for member in rows:
            members_by_group[member["group_id"]].append(member["user_id"])
        return members_by_group

    def _row_to_group_with_members(
        self, row: sqlite3.Row, members: list[str]
    ) -> GroupRecord:
        return GroupRecord(
            id=row["id"],
            name=row["name"],
            user_ids=members,
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_permission(self, row: sqlite3.Row) -> DocumentPermission:
        principal_id = row["principal_id"] or None
        return DocumentPermission(
            document_id=row["document_id"],
            principal_type=row["principal_type"],
            principal_id=principal_id,
            access=row["access"],
            created_at=row["created_at"],
        )

    def _row_to_chat_session(self, row: sqlite3.Row) -> ChatSessionRecord:
        return ChatSessionRecord(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_chat_message(self, row: sqlite3.Row) -> ChatMessageRecord:
        return ChatMessageRecord(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def _row_to_index_attempt(self, row: sqlite3.Row) -> IndexAttemptRecord:
        return IndexAttemptRecord(
            id=row["id"],
            connector_id=row["connector_id"],
            status=row["status"],
            total_documents=row["total_documents"],
            total_chunks=row["total_chunks"],
            error=row["error"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_user_memory(self, row: sqlite3.Row) -> UserMemoryRecord:
        return UserMemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            memory_text=row["memory_text"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Query history helpers
    # ------------------------------------------------------------------

    def upsert_message_feedback(
        self,
        message_id: str,
        is_positive: bool | None,
        feedback_text: str | None = None,
    ) -> bool:
        """Persist feedback for a chat message. Returns False if message not found."""
        row = self._conn.execute(
            "SELECT metadata_json FROM chat_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return False
        meta = _json_loads(row["metadata_json"])
        if is_positive is True:
            meta["feedback"] = "like"
        elif is_positive is False:
            meta["feedback"] = "dislike"
        if feedback_text is not None:
            meta["feedback_text"] = feedback_text
        self._conn.execute(
            "UPDATE chat_messages SET metadata_json = ? WHERE id = ?",
            (_json_dumps(meta), message_id),
        )
        self._conn.commit()
        return True

    def get_paginated_chat_sessions(
        self,
        page_num: int,
        page_size: int,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        feedback_type: str | None = None,
    ) -> list[ChatSessionRecord]:
        """Return a single page of chat sessions ordered by creation time descending.

        *feedback_type* filters to sessions that contain at least one message
        with the given feedback value (``"like"`` or ``"dislike"``).
        """
        if feedback_type:
            clauses = ["s.id = s.id"]
            params: list[object] = []
            if start_time:
                clauses.append("s.created_at >= ?")
                params.append(start_time)
            if end_time:
                clauses.append("s.created_at <= ?")
                params.append(end_time)
            if user_id:
                clauses.append("s.user_id = ?")
                params.append(user_id)
            clauses.append(
                "EXISTS ("
                "  SELECT 1 FROM chat_messages m"
                "  WHERE m.session_id = s.id"
                "  AND json_extract(m.metadata_json, '$.feedback') = ?"
                ")"
            )
            params.append(feedback_type)
            where = "WHERE " + " AND ".join(clauses)
            params.extend([page_size, page_num * page_size])
            rows = self._conn.execute(
                f"SELECT s.* FROM chat_sessions s {where}"
                " ORDER BY s.created_at DESC LIMIT ? OFFSET ?",
                tuple(params),
            ).fetchall()
        else:
            clauses_simple: list[str] = []
            params_simple: list[object] = []
            if start_time:
                clauses_simple.append("created_at >= ?")
                params_simple.append(start_time)
            if end_time:
                clauses_simple.append("created_at <= ?")
                params_simple.append(end_time)
            if user_id:
                clauses_simple.append("user_id = ?")
                params_simple.append(user_id)
            where = ("WHERE " + " AND ".join(clauses_simple)) if clauses_simple else ""
            params_simple.extend([page_size, page_num * page_size])
            rows = self._conn.execute(
                f"SELECT * FROM chat_sessions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                tuple(params_simple),
            ).fetchall()
        return [self._row_to_chat_session(row) for row in rows]

    def get_chat_sessions_count(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        feedback_type: str | None = None,
    ) -> int:
        """Return total session count matching the given filters."""
        if feedback_type:
            clauses = ["s.id = s.id"]
            params: list[object] = []
            if start_time:
                clauses.append("s.created_at >= ?")
                params.append(start_time)
            if end_time:
                clauses.append("s.created_at <= ?")
                params.append(end_time)
            if user_id:
                clauses.append("s.user_id = ?")
                params.append(user_id)
            clauses.append(
                "EXISTS ("
                "  SELECT 1 FROM chat_messages m"
                "  WHERE m.session_id = s.id"
                "  AND json_extract(m.metadata_json, '$.feedback') = ?"
                ")"
            )
            params.append(feedback_type)
            where = "WHERE " + " AND ".join(clauses)
            row = self._conn.execute(
                f"SELECT COUNT(*) AS cnt FROM chat_sessions s {where}", tuple(params)
            ).fetchone()
        else:
            clauses_simple: list[str] = []
            params_simple: list[object] = []
            if start_time:
                clauses_simple.append("created_at >= ?")
                params_simple.append(start_time)
            if end_time:
                clauses_simple.append("created_at <= ?")
                params_simple.append(end_time)
            if user_id:
                clauses_simple.append("user_id = ?")
                params_simple.append(user_id)
            where = ("WHERE " + " AND ".join(clauses_simple)) if clauses_simple else ""
            row = self._conn.execute(
                f"SELECT COUNT(*) AS cnt FROM chat_sessions {where}",
                tuple(params_simple),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_audit_summary(
        self,
        start: str,
        end: str,
    ) -> dict[str, Any]:
        """Return audit metrics for the given UTC ISO window.

        Returns a dict with:
          total_sessions, total_messages, sessions_with_feedback,
          sessions_with_dislike, dislike_rate, top_disliked_queries.
        """

        def _count(sql: str, params: tuple) -> int:
            row = self._conn.execute(sql, params).fetchone()
            return int(row[0]) if row else 0

        total_sessions = _count(
            "SELECT COUNT(*) FROM chat_sessions WHERE created_at >= ? AND created_at < ?",
            (start, end),
        )
        total_messages = _count(
            """
            SELECT COUNT(*) FROM chat_messages m
            JOIN chat_sessions s ON s.id = m.session_id
            WHERE s.created_at >= ? AND s.created_at < ?
            """,
            (start, end),
        )
        sessions_with_feedback = _count(
            """
            SELECT COUNT(DISTINCT m.session_id)
            FROM chat_messages m
            JOIN chat_sessions s ON s.id = m.session_id
            WHERE s.created_at >= ? AND s.created_at < ?
              AND json_extract(m.metadata_json, '$.feedback') IS NOT NULL
            """,
            (start, end),
        )
        sessions_with_dislike = _count(
            """
            SELECT COUNT(DISTINCT m.session_id)
            FROM chat_messages m
            JOIN chat_sessions s ON s.id = m.session_id
            WHERE s.created_at >= ? AND s.created_at < ?
              AND json_extract(m.metadata_json, '$.feedback') = 'dislike'
            """,
            (start, end),
        )
        dislike_rate = (
            round(sessions_with_dislike / sessions_with_feedback, 4)
            if sessions_with_feedback > 0
            else 0.0
        )

        top_rows = self._conn.execute(
            """
            SELECT first_msg.content AS query
            FROM chat_sessions s
            JOIN chat_messages first_msg
              ON first_msg.id = (
                SELECT id FROM chat_messages
                WHERE session_id = s.id AND role = 'user'
                ORDER BY created_at LIMIT 1
              )
            WHERE s.created_at >= ? AND s.created_at < ?
              AND EXISTS (
                SELECT 1 FROM chat_messages m
                WHERE m.session_id = s.id
                  AND json_extract(m.metadata_json, '$.feedback') = 'dislike'
              )
            ORDER BY s.created_at DESC
            LIMIT 10
            """,
            (start, end),
        ).fetchall()
        top_disliked_queries = [row[0] for row in top_rows]

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "sessions_with_feedback": sessions_with_feedback,
            "sessions_with_dislike": sessions_with_dislike,
            "dislike_rate": dislike_rate,
            "top_disliked_queries": top_disliked_queries,
        }

    def list_all_users(self) -> list[UserRecord]:
        """Return every user record ordered by created_at."""
        rows = self._conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [self._row_to_user(row) for row in rows]

    def get_user_by_email(self, email: str) -> UserRecord | None:
        """Look up a user by email (case-insensitive)."""
        row = self._conn.execute(
            "SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def get_users_by_ids(self, user_ids: list[str]) -> dict[str, UserRecord]:
        """Return a {user_id: UserRecord} map for the given IDs in one query."""
        if not user_ids:
            return {}
        placeholders = ", ".join("?" * len(user_ids))
        rows = self._conn.execute(
            f"SELECT * FROM users WHERE id IN ({placeholders})", tuple(user_ids)
        ).fetchall()
        return {row["id"]: self._row_to_user(row) for row in rows}

    def get_users_emails_batch(self, user_ids: list[str]) -> dict[str, str | None]:
        """Return {user_id: email} for a batch of user IDs in one query."""
        if not user_ids:
            return {}
        placeholders = ", ".join("?" * len(user_ids))
        rows = self._conn.execute(
            f"SELECT id, email FROM users WHERE id IN ({placeholders})",
            tuple(user_ids),
        ).fetchall()
        return {row["id"]: row["email"] for row in rows}

    def get_groups_for_users_batch(
        self, user_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Return {user_id: [(group_id, group_name)]} for all users in one JOIN query."""
        if not user_ids:
            return {uid: [] for uid in user_ids}
        placeholders = ", ".join("?" * len(user_ids))
        rows = self._conn.execute(
            f"""
            SELECT gm.user_id, g.id AS group_id, g.name AS group_name
            FROM groups g
            JOIN group_members gm ON gm.group_id = g.id
            WHERE gm.user_id IN ({placeholders})
            ORDER BY gm.user_id, g.name
            """,
            tuple(user_ids),
        ).fetchall()
        result: dict[str, list[tuple[str, str]]] = {uid: [] for uid in user_ids}
        for row in rows:
            result[row["user_id"]].append((row["group_id"], row["group_name"]))
        return result

    # ------------------------------------------------------------------
    # Usage report storage
    # ------------------------------------------------------------------

    def save_usage_report(
        self,
        report_name: str,
        data: bytes,
        requestor_id: str | None = None,
        period_from: str | None = None,
        period_to: str | None = None,
    ) -> None:
        """Persist a usage report ZIP as a BLOB alongside its metadata."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO usage_reports
                (report_name, requestor_id, time_created, period_from, period_to, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (report_name, requestor_id, _now(), period_from, period_to, data),
        )
        self._conn.commit()

    def get_all_usage_reports(self) -> list[dict[str, str | None]]:
        """Return metadata rows for all stored usage reports."""
        rows = self._conn.execute(
            """
            SELECT report_name, requestor_id, time_created, period_from, period_to
            FROM usage_reports ORDER BY time_created DESC
            """
        ).fetchall()
        return [
            {
                "report_name": row["report_name"],
                "requestor_id": row["requestor_id"],
                "time_created": row["time_created"],
                "period_from": row["period_from"],
                "period_to": row["period_to"],
            }
            for row in rows
        ]

    def get_usage_report_data(self, report_name: str) -> bytes | None:
        """Return the raw ZIP bytes for *report_name*, or None if not found."""
        row = self._conn.execute(
            "SELECT data FROM usage_reports WHERE report_name = ?", (report_name,)
        ).fetchone()
        return bytes(row["data"]) if row else None

    # ------------------------------------------------------------------
    # Token rate limits
    # ------------------------------------------------------------------

    def insert_token_rate_limit(
        self,
        scope: str,
        token_budget: int,
        period_hours: int,
        enabled: bool = True,
        scope_id: str | None = None,
    ) -> dict[str, object]:
        """Insert a new token rate limit record and return it as a dict."""
        now = _now()
        record_id = _new_id("trl")
        self._conn.execute(
            """
            INSERT INTO token_rate_limits
                (id, scope, scope_id, enabled, token_budget, period_hours, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                scope,
                scope_id,
                int(enabled),
                token_budget,
                period_hours,
                now,
                now,
            ),
        )
        self._conn.commit()
        return {
            "id": record_id,
            "scope": scope,
            "scope_id": scope_id,
            "enabled": enabled,
            "token_budget": token_budget,
            "period_hours": period_hours,
            "created_at": now,
            "updated_at": now,
        }

    def get_token_rate_limits(
        self,
        scope: str,
        scope_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return token rate limit records matching *scope* and optional *scope_id*."""
        if scope_id is None:
            rows = self._conn.execute(
                "SELECT * FROM token_rate_limits WHERE scope = ? AND scope_id IS NULL ORDER BY created_at",
                (scope,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM token_rate_limits WHERE scope = ? AND scope_id = ? ORDER BY created_at",
                (scope, scope_id),
            ).fetchall()
        return [self._row_to_rate_limit(r) for r in rows]

    def get_all_group_token_rate_limits(self) -> list[dict[str, object]]:
        """Return all group-scoped rate limits joined with their group name."""
        rows = self._conn.execute(
            """
            SELECT trl.*, g.name AS group_name
            FROM token_rate_limits trl
            LEFT JOIN groups g ON g.id = trl.scope_id
            WHERE trl.scope = 'group'
            ORDER BY g.name, trl.created_at
            """
        ).fetchall()
        result = []
        for r in rows:
            record = self._row_to_rate_limit(r)
            record["group_name"] = r["group_name"]
            result.append(record)
        return result

    @staticmethod
    def _row_to_rate_limit(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "scope": row["scope"],
            "scope_id": row["scope_id"],
            "enabled": bool(row["enabled"]),
            "token_budget": row["token_budget"],
            "period_hours": row["period_hours"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # Standard answers (manage module)
    # ------------------------------------------------------------------

    def insert_standard_answer_category(self, name: str) -> dict[str, str]:
        now = _now()
        rec_id = _new_id("sac")
        self._conn.execute(
            "INSERT INTO standard_answer_categories (id, name, created_at) VALUES (?, ?, ?)",
            (rec_id, name, now),
        )
        self._conn.commit()
        return {"id": rec_id, "name": name, "created_at": now}

    def fetch_standard_answer_categories(self) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT * FROM standard_answer_categories ORDER BY name"
        ).fetchall()
        return [
            {"id": r["id"], "name": r["name"], "created_at": r["created_at"]}
            for r in rows
        ]

    def fetch_standard_answer_category(self, category_id: str) -> dict[str, str] | None:
        row = self._conn.execute(
            "SELECT * FROM standard_answer_categories WHERE id = ?", (category_id,)
        ).fetchone()
        return (
            {"id": row["id"], "name": row["name"], "created_at": row["created_at"]}
            if row
            else None
        )

    def update_standard_answer_category(
        self, category_id: str, name: str
    ) -> dict[str, str]:
        row = self._conn.execute(
            "SELECT * FROM standard_answer_categories WHERE id = ?", (category_id,)
        ).fetchone()
        if not row:
            msg = f"Category {category_id!r} not found"
            raise KeyError(msg)
        self._conn.execute(
            "UPDATE standard_answer_categories SET name = ? WHERE id = ?",
            (name, category_id),
        )
        self._conn.commit()
        return {"id": category_id, "name": name, "created_at": row["created_at"]}

    def insert_standard_answer(
        self,
        keyword: str,
        answer: str,
        category_ids: list[str],
        match_regex: bool = False,
        match_any_keywords: bool = False,
    ) -> dict[str, object]:
        now = _now()
        rec_id = _new_id("sa")
        self._conn.execute(
            """
            INSERT INTO standard_answers
                (id, keyword, answer, match_regex, match_any_keywords, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec_id,
                keyword,
                answer,
                int(match_regex),
                int(match_any_keywords),
                now,
                now,
            ),
        )
        for cid in category_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO standard_answer_category_links (answer_id, category_id) VALUES (?, ?)",
                (rec_id, cid),
            )
        self._conn.commit()
        return self._fetch_standard_answer(rec_id)  # type: ignore[return-value]

    def fetch_standard_answers(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT * FROM standard_answers ORDER BY created_at DESC"
        ).fetchall()
        if not rows:
            return []
        answer_ids = [r["id"] for r in rows]
        placeholders = ", ".join("?" * len(answer_ids))
        link_rows = self._conn.execute(
            f"SELECT answer_id, category_id FROM standard_answer_category_links WHERE answer_id IN ({placeholders})",
            tuple(answer_ids),
        ).fetchall()
        links: dict[str, list[str]] = {}
        for lr in link_rows:
            links.setdefault(lr["answer_id"], []).append(lr["category_id"])
        return [
            {
                "id": r["id"],
                "keyword": r["keyword"],
                "answer": r["answer"],
                "match_regex": bool(r["match_regex"]),
                "match_any_keywords": bool(r["match_any_keywords"]),
                "categories": links.get(r["id"], []),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def fetch_standard_answer(self, answer_id: str) -> dict[str, object] | None:
        return self._fetch_standard_answer(answer_id)

    def _fetch_standard_answer(self, answer_id: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT * FROM standard_answers WHERE id = ?", (answer_id,)
        ).fetchone()
        if not row:
            return None
        cat_rows = self._conn.execute(
            "SELECT category_id FROM standard_answer_category_links WHERE answer_id = ?",
            (answer_id,),
        ).fetchall()
        return {
            "id": row["id"],
            "keyword": row["keyword"],
            "answer": row["answer"],
            "match_regex": bool(row["match_regex"]),
            "match_any_keywords": bool(row["match_any_keywords"]),
            "categories": [r["category_id"] for r in cat_rows],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_standard_answer(
        self,
        answer_id: str,
        keyword: str,
        answer: str,
        category_ids: list[str],
        match_regex: bool = False,
        match_any_keywords: bool = False,
    ) -> dict[str, object]:
        now = _now()
        self._conn.execute(
            """
            UPDATE standard_answers
            SET keyword=?, answer=?, match_regex=?, match_any_keywords=?, updated_at=?
            WHERE id=?
            """,
            (
                keyword,
                answer,
                int(match_regex),
                int(match_any_keywords),
                now,
                answer_id,
            ),
        )
        self._conn.execute(
            "DELETE FROM standard_answer_category_links WHERE answer_id = ?",
            (answer_id,),
        )
        for cid in category_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO standard_answer_category_links (answer_id, category_id) VALUES (?, ?)",
                (answer_id, cid),
            )
        self._conn.commit()
        return self._fetch_standard_answer(answer_id)  # type: ignore[return-value]

    def remove_standard_answer(self, answer_id: str) -> None:
        self._conn.execute(
            "DELETE FROM standard_answer_category_links WHERE answer_id = ?",
            (answer_id,),
        )
        self._conn.execute("DELETE FROM standard_answers WHERE id = ?", (answer_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # SCIM token storage
    # ------------------------------------------------------------------

    def get_scim_token_by_hash(self, token_hash: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT * FROM scim_tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        return self._row_to_scim_token(row) if row else None

    def create_scim_token(
        self, name: str, token_hash: str, token_display: str
    ) -> dict[str, object]:
        now = _now()
        rec_id = _new_id("sctok")
        self._conn.execute(
            """
            INSERT INTO scim_tokens (id, name, token_hash, token_display, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (rec_id, name, token_hash, token_display, now),
        )
        self._conn.commit()
        return {
            "id": rec_id,
            "name": name,
            "token_hash": token_hash,
            "token_display": token_display,
            "is_active": True,
            "created_at": now,
            "last_used_at": None,
        }

    def update_scim_token_last_used(self, token_id: str) -> None:
        self._conn.execute(
            "UPDATE scim_tokens SET last_used_at = ? WHERE id = ?", (_now(), token_id)
        )
        self._conn.commit()

    def list_scim_tokens(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT * FROM scim_tokens ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_scim_token(r) for r in rows]

    def revoke_scim_token(self, token_id: str) -> bool:
        cur = self._conn.execute(
            "UPDATE scim_tokens SET is_active = 0 WHERE id = ?", (token_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_scim_token(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "name": row["name"],
            "token_hash": row["token_hash"],
            "token_display": row["token_display"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
        }

    def get_scim_user_mapping(self, user_id: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT * FROM scim_user_mappings WHERE user_id = ?", (user_id,)
        ).fetchone()
        return self._row_to_scim_user_mapping(row) if row else None

    @staticmethod
    def _row_to_scim_user_mapping(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "external_id": row["external_id"],
            "scim_username": row["scim_username"],
            "department": row["department"],
            "manager": row["manager"],
            "given_name": row["given_name"],
            "family_name": row["family_name"],
            "scim_emails_json": row["scim_emails_json"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_scim_user_mapping(
        self,
        user_id: str,
        external_id: str | None = None,
        scim_username: str | None = None,
        department: str | None = None,
        manager: str | None = None,
        given_name: str | None = None,
        family_name: str | None = None,
        scim_emails_json: str | None = None,
    ) -> None:
        now = _now()
        rec_id = _new_id("scum")
        self._conn.execute(
            """
            INSERT INTO scim_user_mappings
                (id, user_id, external_id, scim_username, department, manager,
                 given_name, family_name, scim_emails_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec_id,
                user_id,
                external_id,
                scim_username,
                department,
                manager,
                given_name,
                family_name,
                scim_emails_json,
                now,
                now,
            ),
        )
        self._conn.commit()

    def upsert_scim_user_mapping(
        self,
        user_id: str,
        external_id: str | None = None,
        scim_username: str | None = None,
        department: str | None = None,
        manager: str | None = None,
        given_name: str | None = None,
        family_name: str | None = None,
        scim_emails_json: str | None = None,
    ) -> None:
        now = _now()
        existing = self._conn.execute(
            "SELECT id FROM scim_user_mappings WHERE user_id = ?", (user_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                """
                UPDATE scim_user_mappings
                SET external_id=?, scim_username=?, department=?, manager=?,
                    given_name=?, family_name=?, scim_emails_json=?, updated_at=?
                WHERE user_id=?
                """,
                (
                    external_id,
                    scim_username,
                    department,
                    manager,
                    given_name,
                    family_name,
                    scim_emails_json,
                    now,
                    user_id,
                ),
            )
        else:
            rec_id = _new_id("scum")
            self._conn.execute(
                """
                INSERT INTO scim_user_mappings
                    (id, user_id, external_id, scim_username, department, manager,
                     given_name, family_name, scim_emails_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id,
                    user_id,
                    external_id,
                    scim_username,
                    department,
                    manager,
                    given_name,
                    family_name,
                    scim_emails_json,
                    now,
                    now,
                ),
            )
        self._conn.commit()

    def delete_scim_user_mapping(self, user_id: str) -> None:
        self._conn.execute(
            "DELETE FROM scim_user_mappings WHERE user_id = ?", (user_id,)
        )
        self._conn.commit()

    def get_scim_group_mapping(self, group_id: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT * FROM scim_group_mappings WHERE group_id = ?", (group_id,)
        ).fetchone()
        return self._row_to_scim_group_mapping(row) if row else None

    @staticmethod
    def _row_to_scim_group_mapping(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "group_id": row["group_id"],
            "external_id": row["external_id"],
            "created_at": row["created_at"],
        }

    def create_scim_group_mapping(
        self, group_id: str, external_id: str | None = None
    ) -> None:
        rec_id = _new_id("scgm")
        self._conn.execute(
            "INSERT OR IGNORE INTO scim_group_mappings (id, group_id, external_id, created_at) VALUES (?, ?, ?, ?)",
            (rec_id, group_id, external_id, _now()),
        )
        self._conn.commit()

    def upsert_scim_group_mapping(self, group_id: str, external_id: str | None) -> None:
        existing = self._conn.execute(
            "SELECT id FROM scim_group_mappings WHERE group_id = ?", (group_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE scim_group_mappings SET external_id = ? WHERE group_id = ?",
                (external_id, group_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO scim_group_mappings (id, group_id, external_id, created_at) VALUES (?, ?, ?, ?)",
                (_new_id("scgm"), group_id, external_id, _now()),
            )
        self._conn.commit()

    def delete_scim_group_mapping(self, group_id: str) -> None:
        self._conn.execute(
            "DELETE FROM scim_group_mappings WHERE group_id = ?", (group_id,)
        )
        self._conn.commit()

    def set_user_active(self, user_id: str, is_active: bool) -> None:
        """Set the is_active flag for a user (used by SCIM provisioning)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO user_is_active (user_id, is_active) VALUES (?, ?)",
            (user_id, int(is_active)),
        )
        self._conn.commit()

    def get_user_active(self, user_id: str) -> bool:
        """Return the is_active flag for a user (defaults to True)."""
        row = self._conn.execute(
            "SELECT is_active FROM user_is_active WHERE user_id = ?", (user_id,)
        ).fetchone()
        return bool(row["is_active"]) if row else True

    # ------------------------------------------------------------------
    # User memories
    # ------------------------------------------------------------------

    def add_user_memory(
        self,
        user_id: str,
        memory_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> UserMemoryRecord | None:
        """Persist a sanitized memory for a user."""
        if not memory_text.strip():
            return None

        captured, capture_meta = deterministic_capture(memory_text.strip())
        now = _now()
        record_id = _new_id("mem")
        merged_meta = dict(metadata or {})
        merged_meta.update(capture_meta)
        self._conn.execute(
            """
            INSERT INTO user_memories
                (id, user_id, memory_text, metadata_json, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (record_id, user_id, captured, _json_dumps(merged_meta), now, now),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM user_memories WHERE id = ?", (record_id,)
        ).fetchone()
        return self._row_to_user_memory(row)

    def update_user_memory_at_index(
        self,
        user_id: str,
        index: int,
        new_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> UserMemoryRecord | None:
        """Replace one active memory by zero-based display index."""
        if index < 0 or not new_text.strip():
            return None

        row = self._conn.execute(
            """
            SELECT * FROM user_memories
            WHERE user_id = ? AND is_active = 1
            ORDER BY created_at, id
            LIMIT 1 OFFSET ?
            """,
            (user_id, index),
        ).fetchone()
        if row is None:
            return None

        captured, capture_meta = deterministic_capture(new_text.strip())
        merged_meta = _json_loads(row["metadata_json"])
        merged_meta.update(metadata or {})
        merged_meta.update(capture_meta)
        self._conn.execute(
            """
            UPDATE user_memories
            SET memory_text = ?, metadata_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (captured, _json_dumps(merged_meta), _now(), row["id"]),
        )
        self._conn.commit()
        updated = self._conn.execute(
            "SELECT * FROM user_memories WHERE id = ?", (row["id"],)
        ).fetchone()
        return self._row_to_user_memory(updated)

    def get_user_memories(self, user_id: str) -> list[str]:
        """Return active memory text for a user in display order."""
        rows = self._conn.execute(
            """
            SELECT memory_text FROM user_memories
            WHERE user_id = ? AND is_active = 1
            ORDER BY created_at, id
            """,
            (user_id,),
        ).fetchall()
        return [str(row["memory_text"]) for row in rows]

    # ------------------------------------------------------------------
    # Retrieval feedback
    # ------------------------------------------------------------------

    def save_retrieval_feedback(
        self,
        session_id: str | None,
        signal: str,
        *,
        note: str | None = None,
        source: str | None = None,
        parent_feedback_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a thumbs_up or thumbs_down signal for a search session."""
        feedback_id = _new_id("fb")
        metadata_json = dict(metadata or {})
        if note is not None:
            sanitized_note, note_meta = deterministic_capture(note)
            metadata_json["note"] = sanitized_note
            metadata_json["note_capture"] = note_meta
        if source is not None:
            metadata_json["source"] = source
        self._conn.execute(
            """
            INSERT INTO retrieval_feedback
                (id, session_id, signal, metadata_json, parent_feedback_id, correlation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                session_id,
                signal,
                _json_dumps(metadata_json),
                parent_feedback_id,
                correlation_id,
                _now(),
            ),
        )
        self._conn.commit()
        return feedback_id

    def list_retrieval_feedback(self) -> list[dict[str, object]]:
        """Return retrieval feedback rows for tests and local tooling."""
        rows = self._conn.execute(
            """
            SELECT id, session_id, signal, metadata_json, parent_feedback_id,
                   correlation_id, created_at
            FROM retrieval_feedback
            ORDER BY created_at, id
            """
        ).fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "signal": row["signal"],
                "metadata": _json_loads(row["metadata_json"]),
                "parent_feedback_id": row["parent_feedback_id"],
                "correlation_id": row["correlation_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_feedback_summary(self) -> dict[str, object]:
        """Return aggregate feedback metrics.

        thumbs_up_rate — fraction of rated queries that got thumbs_up.
        ctr            — fraction of distinct sessions that received any feedback.
        rated_queries  — total feedback rows persisted.
        """
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                                                    AS total,
                SUM(CASE WHEN signal = 'thumbs_up' THEN 1 ELSE 0 END)     AS ups,
                COUNT(DISTINCT session_id)                                  AS rated_sessions
            FROM retrieval_feedback
            """
        ).fetchone()
        total = int(row["total"]) if row else 0
        ups = int(row["ups"] or 0) if row else 0
        rated_sessions = int(row["rated_sessions"] or 0) if row else 0

        total_sessions_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chat_sessions"
        ).fetchone()
        total_sessions = int(total_sessions_row["n"]) if total_sessions_row else 0

        thumbs_up_rate = round(ups / total, 4) if total else 0.0
        ctr = round(rated_sessions / total_sessions, 4) if total_sessions else 0.0
        return {
            "thumbs_up_rate": thumbs_up_rate,
            "ctr": ctr,
            "rated_queries": total,
        }
