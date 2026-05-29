"""
This test tests the happy path for curator permissions
"""

import os

import pytest

from tests.integration.common_utils.types import AccessType
from tests.integration.common_utils.types import UserRole
from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.managers.cc_pair import CCPairManager
from tests.integration.common_utils.managers.connector import ConnectorManager
from tests.integration.common_utils.managers.credential import CredentialManager
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.managers.user_group import UserGroupManager


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Curator tests are enterprise only",
)
def test_whole_curator_flow(reset: None) -> None:  # noqa: ARG001
    # Creating an admin user (first user created is automatically an admin)
    assert UserManager.is_role(admin_user, UserRole.ADMIN)  # noqa: F821,F841

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
    # Making curator a curator of user_group_1
    UserGroupManager.set_curator_status(
        test_user_group=user_group_1,
        user_to_set_as_curator=curator,  # noqa: F821,F841
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    assert UserManager.is_role(curator, UserRole.CURATOR)  # noqa: F821,F841

    # Creating a credential as curator
    test_credential = CredentialManager.create(
        name="curator_test_credential",
        source=DocumentSource.FILE,
        curator_public=False,
        groups=[user_group_1.id],
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Creating a connector as curator
    test_connector = ConnectorManager.create(
        name="curator_test_connector",
        source=DocumentSource.FILE,
        access_type=AccessType.PRIVATE,
        groups=[user_group_1.id],
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Test editing the connector
    test_connector.name = "updated_test_connector"
    ConnectorManager.edit(connector=test_connector, user_performing_action=curator)  # noqa: F821,F841

    # Creating a CC pair as curator
    test_cc_pair = CCPairManager.create(
        connector_id=test_connector.id,
        credential_id=test_credential.id,
        name="curator_test_cc_pair",
        access_type=AccessType.PRIVATE,
        groups=[user_group_1.id],
        user_performing_action=curator,  # noqa: F821,F841
    )

    CCPairManager.verify(cc_pair=test_cc_pair, user_performing_action=admin_user)  # noqa: F821,F841

    # Verify that the curator can pause and unpause the CC pair

    # Verify that the curator can delete the CC pair
    CCPairManager.wait_for_deletion_completion(
        cc_pair_id=test_cc_pair.id,
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Verify that the CC pair has been deleted
    CCPairManager.verify(
        cc_pair=test_cc_pair,
        verify_deleted=True,
        user_performing_action=admin_user,  # noqa: F821,F841
    )


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Curator tests are enterprise only",
)
def test_global_curator_flow(reset: None) -> None:  # noqa: ARG001
    # Creating an admin user (first user created is automatically an admin)
    assert UserManager.is_role(admin_user, UserRole.ADMIN)  # noqa: F821,F841

    # Creating a user
    assert UserManager.is_role(global_curator, UserRole.BASIC)  # noqa: F821,F841

    # Set the user to a global curator
    UserManager.set_role(
        user_to_set=global_curator,  # noqa: F821,F841
        target_role=UserRole.GLOBAL_CURATOR,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    assert UserManager.is_role(global_curator, UserRole.GLOBAL_CURATOR)  # noqa: F821,F841

    # Creating a user group containing the global curator
    user_group_1 = UserGroupManager.create(
        name="user_group_1",
        user_ids=[global_curator.id],  # noqa: F821,F841
        cc_pair_ids=[],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    UserGroupManager.wait_for_sync(
        user_groups_to_check=[user_group_1],
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Creating a credential as global curator
    test_credential = CredentialManager.create(
        name="curator_test_credential",
        source=DocumentSource.FILE,
        curator_public=False,
        groups=[user_group_1.id],
        user_performing_action=global_curator,  # noqa: F821,F841
    )

    # Creating a connector as global curator
    test_connector = ConnectorManager.create(
        name="curator_test_connector",
        source=DocumentSource.FILE,
        access_type=AccessType.PRIVATE,
        groups=[user_group_1.id],
        user_performing_action=global_curator,  # noqa: F821,F841
    )

    # Test editing the connector
    test_connector.name = "updated_test_connector"
    ConnectorManager.edit(
        connector=test_connector,
        user_performing_action=global_curator,  # noqa: F821,F841
    )

    # Creating a CC pair as global curator
    test_cc_pair = CCPairManager.create(
        connector_id=test_connector.id,
        credential_id=test_credential.id,
        name="curator_test_cc_pair",
        access_type=AccessType.PRIVATE,
        groups=[user_group_1.id],
        user_performing_action=global_curator,  # noqa: F821,F841
    )

    CCPairManager.verify(cc_pair=test_cc_pair, user_performing_action=admin_user)  # noqa: F821,F841

    # Verify that the curator can pause and unpause the CC pair
    CCPairManager.pause_cc_pair(
        cc_pair=test_cc_pair,
        user_performing_action=global_curator,  # noqa: F821,F841
    )

    # Verify that the curator can delete the CC pair
    CCPairManager.wait_for_deletion_completion(
        cc_pair_id=test_cc_pair.id,
        user_performing_action=global_curator,  # noqa: F821,F841
    )

    # Verify that the CC pair has been deleted
    CCPairManager.verify(
        cc_pair=test_cc_pair,
        verify_deleted=True,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
