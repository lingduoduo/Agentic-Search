"""
This file takes the happy path to adding a curator to a user group and then tests
the permissions of the curator manipulating credentials.
"""

import os

import pytest
from requests.exceptions import HTTPError

from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.managers.credential import CredentialManager
from tests.integration.common_utils.managers.user_group import UserGroupManager


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Curator and user group tests are enterprise only",
)
def test_credential_permissions(reset: None) -> None:  # noqa: ARG001
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

    # Curators should not be able to create a credential for a user group they are not a curator of
    with pytest.raises(HTTPError):
        CredentialManager.create(
            name="invalid_credential_2",
            source=DocumentSource.CONFLUENCE,
            groups=[user_group_1.id, user_group_2.id],
            curator_public=False,
            user_performing_action=curator,  # noqa: F821,F841
        )

    """Tests for things Curators should be able to do"""
    # Curators should be able to create a private credential for a user group they are a curator of
    valid_credential = CredentialManager.create(
        name="valid_credential",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        curator_public=False,
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Verify the created credential
    CredentialManager.verify(
        credential=valid_credential,
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Test editing the credential
    valid_credential.name = "updated_valid_credential"
    CredentialManager.edit(valid_credential, user_performing_action=curator)  # noqa: F821,F841

    # Verify the edit
    CredentialManager.verify(
        credential=valid_credential,
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Test deleting the credential

    # Verify the deletion
    CredentialManager.verify(
        credential=valid_credential,
        verify_deleted=True,
        user_performing_action=curator,  # noqa: F821,F841
    )

    # Curators should be able to create a public credential
    public_credential = CredentialManager.create(
        name="curator_public_credential",
        source=DocumentSource.CONFLUENCE,
        groups=[user_group_1.id],
        curator_public=True,
        user_performing_action=curator,  # noqa: F821,F841
    )
    CredentialManager.verify(
        credential=public_credential,
        user_performing_action=curator,  # noqa: F821,F841
    )
