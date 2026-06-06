"""Local index fixture backed by AgenticSearchStore (SQLite).

Reads directly from the same SQLite database that the running API server uses.

Set AGENTIC_SEARCH_WEB_DB_PATH to the file path used when launching the server.
With the default in-memory DB (":memory:") the fixture will always return empty
results, so integration tests that verify indexing require a file-based path.
"""

from __future__ import annotations

import os
from typing import Any


SQLITE_DB_PATH = os.getenv("AGENTIC_SEARCH_WEB_DB_PATH", ":memory:")


class IndexFixture:
    """Test fixture for inspecting the document index via the local SQLite store."""

    def __init__(self, index_name: str | None = None) -> None:
        # index_name is accepted for backwards compat but ignored.
        del index_name
        from src.backend.db.store import AgenticSearchStore

        self._store = AgenticSearchStore(SQLITE_DB_PATH)

    def get_documents_by_id(
        self, document_ids: list[str], wanted_doc_count: int = 1_000
    ) -> dict[str, Any]:
        documents: list[dict[str, Any]] = []
        for doc_id in document_ids[:wanted_doc_count]:
            doc = self._store.get_document(doc_id)
            if doc is None:
                continue
            perms = self._store.get_document_permissions(doc_id)

            acl_entries: set[str] = set()
            is_public = False
            for perm in perms:
                if perm.principal_type == "public":
                    acl_entries.add("PUBLIC")
                    is_public = True
                elif perm.principal_type == "user" and perm.principal_id:
                    user = self._store.get_user(perm.principal_id)
                    if user and user.email:
                        acl_entries.add(f"user_email:{user.email}")
                    else:
                        acl_entries.add(f"user:{perm.principal_id}")
                elif perm.principal_type == "group" and perm.principal_id:
                    acl_entries.add(f"group:{perm.principal_id}")

            documents.append(
                {
                    "fields": {
                        "document_id": doc.id,
                        "access_control_list": {entry: 1 for entry in acl_entries},
                        "document_sets": {},
                        "public": is_public,
                    }
                }
            )
        return {"documents": documents}
