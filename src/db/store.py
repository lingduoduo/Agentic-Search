"""SQLite-backed metadata store for local Agentic Search workflows."""

from __future__ import annotations

import json
import sqlite3
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
    UserRecord,
)


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


class AgenticSearchStore:
    """Small SQLite repository for connector, document, ACL, chat, and indexing state."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AgenticSearchStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

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
            """
        )
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
        for user_id in group.user_ids:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO group_members (group_id, user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (group.id, user_id, now),
            )
        self._conn.commit()
        return self.get_group(group.id) or GroupRecord(id=group.id, name=group.name)

    def list_groups(self) -> list[GroupRecord]:
        """Return all groups ordered by name."""
        rows = self._conn.execute("SELECT * FROM groups ORDER BY name").fetchall()
        return [self._row_to_group(row) for row in rows]

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
        return [self._row_to_group(row) for row in rows]

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
            groups.update(group.id for group in self.list_groups_for_user(user_id))
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
                id, user_id, title, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.user_id,
                record.title,
                _json_dumps(record.metadata),
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

    def _created_at(self, table: str, record_id: str, default: str) -> str:
        row = self._conn.execute(
            f"SELECT created_at FROM {table} WHERE id = ?", (record_id,)
        ).fetchone()
        return str(row["created_at"]) if row else default

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
        members = self._conn.execute(
            """
            SELECT user_id FROM group_members
            WHERE group_id = ?
            ORDER BY user_id
            """,
            (row["id"],),
        ).fetchall()
        return GroupRecord(
            id=row["id"],
            name=row["name"],
            user_ids=[member["user_id"] for member in members],
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

    # ------------------------------------------------------------------
    # Query history helpers
    # ------------------------------------------------------------------

    def get_paginated_chat_sessions(
        self,
        page_num: int,
        page_size: int,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[ChatSessionRecord]:
        """Return a single page of chat sessions ordered by creation time descending."""
        clauses: list[str] = []
        params: list[object] = []
        if start_time:
            clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("created_at <= ?")
            params.append(end_time)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([page_size, page_num * page_size])
        rows = self._conn.execute(
            f"SELECT * FROM chat_sessions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            tuple(params),
        ).fetchall()
        return [self._row_to_chat_session(row) for row in rows]

    def get_chat_sessions_count(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> int:
        """Return total number of chat sessions matching the given time filter."""
        clauses: list[str] = []
        params: list[object] = []
        if start_time:
            clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("created_at <= ?")
            params.append(end_time)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) AS cnt FROM chat_sessions {where}", tuple(params)
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def list_all_users(self) -> list[UserRecord]:
        """Return every user record ordered by created_at."""
        rows = self._conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [self._row_to_user(row) for row in rows]

    # ------------------------------------------------------------------
    # Usage report storage
    # ------------------------------------------------------------------

    def _ensure_usage_reports_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_reports (
                report_name TEXT PRIMARY KEY,
                requestor_id TEXT,
                time_created TEXT NOT NULL,
                period_from TEXT,
                period_to TEXT,
                data BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_usage_report(
        self,
        report_name: str,
        data: bytes,
        requestor_id: str | None = None,
        period_from: str | None = None,
        period_to: str | None = None,
    ) -> None:
        """Persist a usage report ZIP as a BLOB alongside its metadata."""
        self._ensure_usage_reports_table()
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
        self._ensure_usage_reports_table()
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
        self._ensure_usage_reports_table()
        row = self._conn.execute(
            "SELECT data FROM usage_reports WHERE report_name = ?", (report_name,)
        ).fetchone()
        return bytes(row["data"]) if row else None
