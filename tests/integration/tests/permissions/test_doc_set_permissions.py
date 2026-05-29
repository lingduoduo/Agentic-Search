import os

import pytest
from requests.exceptions import HTTPError

from tests.integration.common_utils.types import AccessType
from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.managers.cc_pair import CCPairManager
from tests.integration.common_utils.managers.document_set import DocumentSetManager
from tests.integration.common_utils.managers.user_group import UserGroupManager


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Curator and user group tests are enterprise only",
)
def test_doc_set_permissions_setup(reset: None) -> None:  # noqa: ARG001
    # Creating an admin user (first user created is automatically an admin)

    # Creating a second user (curator)

    # Creating the first user group
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

    # Setting the curator as a curator for the first user group
    UserGroupManager.set_curator_status(
        test_user_group=user_group_1,
        user_to_set_as_curator=curator,  # noqa: F821,F841
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Creating a second user group
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

    # Admin creates a cc_pair
    private_cc_pair = CCPairManager.create_from_scratch(  # noqa: F821,F841
        access_type=AccessType.PRIVATE,
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Admin creates a public cc_pair
    public_cc_pair = CCPairManager.create_from_scratch(
        access_type=AccessType.PUBLIC,
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # END OF HAPPY PATH

    """Tests for things Curators/Admins should not be able to do"""

    # Test that curator cannot create a non-public document set for the group they don't curate
    with pytest.raises(HTTPError):
        DocumentSetManager.create(
            name="Invalid Document Set 1",
            is_public=False,
            groups=[user_group_2.id],
            cc_pair_ids=[public_cc_pair.id],
            user_performing_action=curator,  # noqa: F821,F841
        )

    # Test that curator cannot create a document set attached to both groups
    with pytest.raises(HTTPError):
        DocumentSetManager.create(
            name="Invalid Document Set 2",
            is_public=False,
            cc_pair_ids=[public_cc_pair.id],
            groups=[user_group_1.id, user_group_2.id],
            user_performing_action=curator,  # noqa: F821,F841
        )

    # Test that curator cannot create a document set with no groups
    with pytest.raises(HTTPError):
        DocumentSetManager.create(
            name="Invalid Document Set 3",
            is_public=False,
            cc_pair_ids=[public_cc_pair.id],
            groups=[],
            user_performing_action=curator,  # noqa: F821,F841
        )

    # Test that curator cannot create a document set with no cc_pairs
    with pytest.raises(HTTPError):
        DocumentSetManager.create(
            name="Invalid Document Set 4",
            is_public=False,
            cc_pair_ids=[],
            groups=[user_group_1.id],
            user_performing_action=curator,  # noqa: F821,F841
        )

    # Test that admin cannot create a document set with no cc_pairs
    with pytest.raises(HTTPError):
        DocumentSetManager.create(
            name="Invalid Document Set 4",
            is_public=False,
            cc_pair_ids=[],
            groups=[user_group_1.id],
            user_performing_action=admin_user,  # noqa: F821,F841
        )

    """Tests for things Curators should be able to do"""
    # Test that curator can create a document set for the group they curate
    valid_doc_set = DocumentSetManager.create(
        name="Valid Document Set",
        is_public=False,
        cc_pair_ids=[public_cc_pair.id],
        groups=[user_group_1.id],
        user_performing_action=curator,  # noqa: F821,F841
    )

    DocumentSetManager.wait_for_sync(
        document_sets_to_check=[valid_doc_set],
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Verify that the valid document set was created
    DocumentSetManager.verify(
        document_set=valid_doc_set,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Verify that only one document set exists
    assert len(all_doc_sets) == 1  # noqa: F821,F841

    # Add the private_cc_pair to the doc set on our end for later comparison

    # Confirm the curator can't add the private_cc_pair to the doc set
    with pytest.raises(HTTPError):
        DocumentSetManager.edit(
            document_set=valid_doc_set,
            user_performing_action=curator,  # noqa: F821,F841
        )
    # Confirm the admin can't add the private_cc_pair to the doc set
    with pytest.raises(HTTPError):
        DocumentSetManager.edit(
            document_set=valid_doc_set,
            user_performing_action=admin_user,  # noqa: F821,F841
        )

    # Verify the document set has not been updated in the db
    with pytest.raises(ValueError):
        DocumentSetManager.verify(
            document_set=valid_doc_set,
            user_performing_action=admin_user,  # noqa: F821,F841
        )

    # Add the private_cc_pair to the user group on our end for later comparison

    # Admin adds the cc_pair to the group the curator curates
    UserGroupManager.edit(
        user_group=user_group_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    UserGroupManager.wait_for_sync(
        user_groups_to_check=[user_group_1],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    UserGroupManager.verify(
        user_group=user_group_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Confirm the curator can now add the cc_pair to the doc set
    DocumentSetManager.edit(
        document_set=valid_doc_set,
        user_performing_action=curator,  # noqa: F821,F841
    )
    DocumentSetManager.wait_for_sync(
        document_sets_to_check=[valid_doc_set],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    # Verify the updated document set
    DocumentSetManager.verify(
        document_set=valid_doc_set,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
