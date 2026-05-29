"""
This file takes the happy path to adding a curator to a user group and then tests
the permissions of the curator manipulating connectors.
"""

import os

import pytest
from requests.exceptions import HTTPError

from tests.integration.common_utils.types import AccessType
from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.managers.connector import ConnectorManager
from tests.integration.common_utils.managers.user_group import UserGroupManager


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Curator and user group tests are enterprise only",
)
def test_connector_permissions(reset: None) -> None:  # noqa: ARG001
    # Creating an admin user (first user created is automatically an admin)

    # Creating a curator

    # Creating a user group
    user_group_1 = UserGroupManager.create(
        name="user_group_1",
        user_ids=[curator.id],  # noqa: F821,F841
        cc_pair_ids=[],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    UserGroupManager.wait_for_sync(
        user_groups_to_check=[user_group_1],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    # setting the user as a curator for the user group
    UserGroupManager.set_curator_status(
        test_user_group=user_group_1,
        user_to_set_as_curator=curator,  # noqa: F821,F841
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Creating another user group that the user is not a curator of
    user_group_2 = UserGroupManager.create(
        name="user_group_2",
        user_ids=[curator.id],  # noqa: F821,F841
        cc_pair_ids=[],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    UserGroupManager.wait_for_sync(
        user_groups_to_check=[user_group_1],
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # END OF HAPPY PATH

    """Tests for things Curators should not be able to do"""

    # Curators should not be able to create a connector for a
    # user group they are not a curator of
    with pytest.raises(HTTPError):
        ConnectorManager.create(
            name="invalid_connector_2",
            source=DocumentSource.CONFLUENCE,
            groups=[user_group_1.id, user_group_2.id],
            access_type=AccessType.PRIVATE,
            user_performing_action=curator,  # noqa: F821,F841
        )

    """Tests for things Curators should be able to do"""

    # Curators should be able to create a private
    # connector for a user group they are a curator of
    valid_connector = ConnectorManager.create(
        name="valid_connector",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        access_type=AccessType.PRIVATE,
        user_performing_action=curator,  # noqa: F821,F841
    )
    assert valid_connector.id is not None

    # Verify the created connector
    created_connector = ConnectorManager.get(
        valid_connector.id,
        user_performing_action=curator,  # noqa: F821,F841
    )
    assert created_connector.name == valid_connector.name
    assert created_connector.source == valid_connector.source

    # Verify that the connector can be found in the list of all connectors
    assert any(conn.id == valid_connector.id for conn in all_connectors)  # noqa: F821,F841

    # Test editing the connector
    valid_connector.name = "updated_valid_connector"
    ConnectorManager.edit(valid_connector, user_performing_action=curator)  # noqa: F821,F841

    # Verify the edit
    updated_connector = ConnectorManager.get(
        valid_connector.id,
        user_performing_action=curator,  # noqa: F821,F841
    )
    assert updated_connector.name == "updated_valid_connector"

    # Test deleting the connector

    # Verify the deletion
    all_connectors_after_delete = ConnectorManager.get_all(
        user_performing_action=curator  # noqa: F821,F841
    )
    assert all(conn.id != valid_connector.id for conn in all_connectors_after_delete)

    # Test that curator cannot create a connector for a group they are not a curator of
    with pytest.raises(HTTPError):
        ConnectorManager.create(
            name="invalid_connector_3",
            source=DocumentSource.CONFLUENCE,
            groups=[user_group_2.id],
            access_type=AccessType.PRIVATE,
            user_performing_action=curator,  # noqa: F821,F841
        )

    # Curators should be able to create a public connector
    public_connector = ConnectorManager.create(
        name="curator_public_connector",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        access_type=AccessType.PUBLIC,
        user_performing_action=curator,  # noqa: F821,F841
    )
    assert public_connector.id is not None
