"""
This file takes the happy path to adding a curator to a user group and then tests
the permissions of the curator manipulating connector-credential pairs.
"""

import os

import pytest
from onyx_openapi_client.exceptions import ApiException  # ty: ignore[unresolved-import]

from tests.integration.common_utils.types import AccessType
from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.managers.cc_pair import CCPairManager
from tests.integration.common_utils.managers.connector import ConnectorManager
from tests.integration.common_utils.managers.credential import CredentialManager
from tests.integration.common_utils.managers.user_group import UserGroupManager


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Curator and User Group tests are enterprise only",
)
def test_cc_pair_permissions(reset: None) -> None:  # noqa: ARG001
    # Creating an admin user (first user created is automatically an admin)

    # Creating a curator

    # Creating a user group
    user_group_1 = UserGroupManager.create(
        name="curated_user_group",
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
        name="uncurated_user_group",
        user_ids=[curator.id],  # noqa: F821,F841
        cc_pair_ids=[],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    UserGroupManager.wait_for_sync(
        user_groups_to_check=[user_group_1],
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    connector_1 = ConnectorManager.create(
        name="admin_owned_connector",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        access_type=AccessType.PRIVATE,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    # currently we dont enforce permissions at the connector level
    # pending cc_pair -> connector rework
    # connector_2 = ConnectorManager.create(
    #     name="curator_visible_connector",
    #     source=DocumentSource.CONFLUENCE,
    #     groups=[user_group_2.id],
    #     is_public=False,
    #     user_performing_action=admin_user,
    # )
    # Create a credentials that the curator is and is not curator of
    credential_1 = CredentialManager.create(
        name="curator_owned_credential",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        curator_public=False,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    credential_2 = CredentialManager.create(
        name="curator_visible_credential",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_2.id],
        curator_public=False,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # END OF HAPPY PATH

    """Tests for things Curators should not be able to do"""

    # Curators should not be able to create a cc
    # pair for a user group they are not a curator of
    with pytest.raises(ApiException):
        CCPairManager.create(
            connector_id=connector_1.id,
            credential_id=credential_1.id,
            name="invalid_cc_pair_2",
            access_type=AccessType.PRIVATE,
            groups=[user_group_1.id, user_group_2.id],
            user_performing_action=curator,  # noqa: F821,F841
        )

    # Curators should not be able to create a cc
    # pair without an attached user group
    with pytest.raises(ApiException):
        CCPairManager.create(
            connector_id=connector_1.id,
            credential_id=credential_1.id,
            name="invalid_cc_pair_2",
            access_type=AccessType.PRIVATE,
            groups=[],
            user_performing_action=curator,  # noqa: F821,F841
        )

    # # This test is currently disabled because permissions are
    # # not enforced at the connector level
    # # Curators should not be able to create a cc pair
    # # for a user group that the connector does not belong to (NOT WORKING)
    # with pytest.raises(HTTPError):
    #     CCPairManager.create(
    #         connector_id=connector_2.id,
    #         credential_id=credential_1.id,
    #         name="invalid_cc_pair_3",
    #         access_type=AccessType.PRIVATE,
    #         groups=[user_group_1.id],
    #         user_performing_action=curator,
    #     )

    # Curators should not be able to create a cc
    # pair for a user group that the credential does not belong to
    with pytest.raises(ApiException):
        CCPairManager.create(
            connector_id=connector_1.id,
            credential_id=credential_2.id,
            name="invalid_cc_pair_4",
            access_type=AccessType.PRIVATE,
            groups=[user_group_1.id],
            user_performing_action=curator,  # noqa: F821,F841
        )

    """Tests for things Curators should be able to do"""

    # Re-create connector since the credential_2 validation error above
    # triggers connector deletion in the exception handler
    connector_1 = ConnectorManager.create(
        name="admin_owned_connector_2",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        access_type=AccessType.PRIVATE,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Curators should be able to create a private
    # cc pair for a user group they are a curator of
    valid_cc_pair = CCPairManager.create(
        name="valid_cc_pair",
        connector_id=connector_1.id,
        credential_id=credential_1.id,
        access_type=AccessType.PRIVATE,
        groups=[user_group_1.id],
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Verify the created cc pair
    CCPairManager.verify(
        cc_pair=valid_cc_pair,
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Test pausing the cc pair

    # Test deleting the cc pair
    CCPairManager.wait_for_deletion_completion(
        cc_pair_id=valid_cc_pair.id,
        user_performing_action=curator,  # noqa: F821,F841
    )

    CCPairManager.verify(
        cc_pair=valid_cc_pair,
        verify_deleted=True,
        user_performing_action=curator,  # noqa: F821,F841
    )
