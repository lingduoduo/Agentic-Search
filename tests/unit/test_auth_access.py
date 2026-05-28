from __future__ import annotations

import pytest

from src.access import acl_for_store_user
from src.access import acl_for_user
from src.access import can_access_document
from src.access import get_access_for_document
from src.access import metadata_with_acl
from src.auth import AuthenticatedUser
from src.auth import generate_user_jwt_token
from src.auth import user_from_jwt_token
from src.db import AgenticSearchStore
from src.db import DocumentPermission
from src.db import GroupRecord
from src.db import StoredDocument
from src.db import UserRecord
from src.retrieval.models import DocumentAccess


def test_user_jwt_round_trips_identity_and_groups():
    token = generate_user_jwt_token(
        user_id="alice",
        email="alice@example.test",
        group_ids=["eng", "ops"],
        secret="test-secret",
    )

    user = user_from_jwt_token(token, secret="test-secret")

    assert user.id == "alice"
    assert user.email == "alice@example.test"
    assert user.group_ids == frozenset({"eng", "ops"})


def test_user_jwt_rejects_bad_signature():
    token = generate_user_jwt_token(user_id="alice", secret="test-secret")

    with pytest.raises(ValueError, match="signature"):
        user_from_jwt_token(token, secret="other-secret")


def test_access_acl_allows_public_user_and_group_entries():
    public = DocumentAccess(is_public=True)
    private_user = DocumentAccess(is_public=False, user_ids={"alice"})
    private_group = DocumentAccess(is_public=False, group_ids={"eng"})
    reader = AuthenticatedUser(id="alice", group_ids=frozenset({"eng"}))

    assert acl_for_user(reader) >= {"public", "user:alice", "group:eng"}
    assert can_access_document(public, None)
    assert can_access_document(private_user, reader)
    assert can_access_document(private_group, reader)
    assert not can_access_document(private_group, AuthenticatedUser(id="bob"))


def test_store_permissions_build_document_and_user_acl(tmp_path):
    with AgenticSearchStore(tmp_path / "state.sqlite3") as store:
        store.upsert_user(UserRecord(id="alice", email="alice@example.test"))
        store.upsert_group(
            GroupRecord(id="eng", name="Engineering", user_ids=["alice"])
        )
        store.upsert_document(StoredDocument(id="doc", title="Doc", contents="Body"))
        store.grant_document_access(
            DocumentPermission(
                document_id="doc", principal_type="group", principal_id="eng"
            )
        )

        access = get_access_for_document(store, "doc")
        user_acl = acl_for_store_user(store, "alice")

    assert access == DocumentAccess(is_public=False, group_ids={"eng"})
    assert user_acl >= {"public", "user:alice", "email:alice@example.test", "group:eng"}
    assert metadata_with_acl(access=access)["acl"] == ["group:eng"]
