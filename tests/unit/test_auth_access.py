from __future__ import annotations

import pytest

from src.backend.access import acl_for_store_user
from src.backend.access import acl_for_user
from src.backend.access import can_access_document
from src.backend.access import get_access_for_document
from src.backend.access import metadata_with_acl
from src.backend.auth import AuthenticatedUser
from src.backend.auth import generate_user_jwt_token
from src.backend.auth import user_from_jwt_token
from src.backend.db import AgenticSearchStore
from src.backend.db import DocumentPermission
from src.backend.db import GroupRecord
from src.backend.db import StoredDocument
from src.backend.db import UserRecord
from src.backend.document_index.models import DocumentAccess


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
    external_group = {"acl": ["external_group:okta-eng"]}
    reader = AuthenticatedUser(
        id="alice",
        group_ids=frozenset({"eng"}),
        metadata={"external_group_ids": ["okta-eng"]},
    )

    assert acl_for_user(reader) >= {
        "public",
        "user:alice",
        "group:eng",
        "external_group:okta-eng",
    }
    assert can_access_document(public, None)
    assert can_access_document(private_user, reader)
    assert can_access_document(private_group, reader)
    assert not acl_for_user(reader).isdisjoint(external_group["acl"])
    assert not can_access_document(private_group, AuthenticatedUser(id="bob"))


def test_store_permissions_build_document_and_user_acl(tmp_path):
    with AgenticSearchStore(tmp_path / "state.sqlite3") as store:
        store.upsert_user(UserRecord(id="alice", email="alice@example.test"))
        store.upsert_group(
            GroupRecord(id="eng", name="Engineering", user_ids=["alice"])
        )
        store.create_scim_group_mapping("eng", external_id="okta-eng")
        store.upsert_document(StoredDocument(id="doc", title="Doc", contents="Body"))
        store.grant_document_access(
            DocumentPermission(
                document_id="doc", principal_type="group", principal_id="eng"
            )
        )

        access = get_access_for_document(store, "doc")
        user_acl = acl_for_store_user(store, "alice")

    assert access == DocumentAccess(is_public=False, group_ids={"eng"})
    assert user_acl >= {
        "public",
        "user:alice",
        "email:alice@example.test",
        "group:eng",
        "external_group:okta-eng",
    }
    assert metadata_with_acl(access=access)["acl"] == ["group:eng"]
