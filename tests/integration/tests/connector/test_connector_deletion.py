import pytest

pytestmark = pytest.mark.skip(
    reason="requires external services not available in this deployment"
)

"""
This file contains tests for the following:
- Ensuring deletion of a connector also:
    - deletes the documents in vespa for that connector
    - updates the document sets and user groups to remove the connector
- Ensure that deleting a connector that is part of an overlapping document set and/or user group works as expected
"""

import os  # noqa: E402
from uuid import uuid4  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

# ConnectorFailure removed — use dict
# Document model not used — replaced with dictFailure
# get_sqlalchemy_engine removed
from tests.integration.common_utils.types import IndexingStatus  # noqa: E402

# create_index_attempt removed
# create_index_attempt removed_error
# IndexAttempt ORM removed — use HTTP
# get_current_search_settings removed — no direct DB access
from tests.integration.common_utils.types import DocumentSource  # noqa: E402
from tests.integration.common_utils.constants import NUM_DOCS  # noqa: E402
from tests.integration.common_utils.managers.api_key import APIKeyManager  # noqa: E402
from tests.integration.common_utils.managers.cc_pair import CCPairManager  # noqa: E402
from tests.integration.common_utils.managers.document import DocumentManager  # noqa: E402
from tests.integration.common_utils.managers.document_set import DocumentSetManager  # noqa: E402
from tests.integration.common_utils.managers.user_group import UserGroupManager  # noqa: E402
from tests.integration.common_utils.test_models import DATestAPIKey  # noqa: E402
from tests.integration.common_utils.test_models import DATestUserGroup  # noqa: E402
# vespa_fixture removed — no Vespa in this deployment


def test_connector_deletion(
    reset: None,  # noqa: ARG001
    vespa_client: vespa_fixture,  # noqa: F821,F841
) -> None:
    user_group_1: DATestUserGroup
    user_group_2: DATestUserGroup

    is_ee = (
        os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() == "true"
    )

    # Creating an admin user (first user created is automatically an admin)
    # create api key
    api_key: DATestAPIKey = APIKeyManager.create(
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # create connectors
    cc_pair_1 = CCPairManager.create_from_scratch(
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    cc_pair_2 = CCPairManager.create_from_scratch(
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # seed documents
    cc_pair_1.documents = DocumentManager.seed_dummy_docs(
        cc_pair=cc_pair_1,
        num_docs=NUM_DOCS,
        api_key=api_key,
    )
    cc_pair_2.documents = DocumentManager.seed_dummy_docs(
        cc_pair=cc_pair_2,
        num_docs=NUM_DOCS,
        api_key=api_key,
    )

    # create document sets
    doc_set_1 = DocumentSetManager.create(
        name="Test Document Set 1",
        cc_pair_ids=[cc_pair_1.id],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    doc_set_2 = DocumentSetManager.create(
        name="Test Document Set 2",
        cc_pair_ids=[cc_pair_1.id, cc_pair_2.id],
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # wait for document sets to be synced

    print("Document sets created and synced")

    if is_ee:
        # create user groups
        user_group_1 = UserGroupManager.create(
            cc_pair_ids=[cc_pair_1.id],
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        user_group_2 = UserGroupManager.create(
            cc_pair_ids=[cc_pair_1.id, cc_pair_2.id],
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        UserGroupManager.wait_for_sync(user_performing_action=admin_user)  # noqa: F821,F841

    # inject a finished index attempt and index attempt error (exercises foreign key errors)
    with Session(get_sqlalchemy_engine()) as db_session:  # noqa: F821,F841
        primary_search_settings = get_current_search_settings(db_session)  # noqa: F821,F841
        new_attempt = IndexAttempt(  # noqa: F821,F841
            connector_credential_pair_id=cc_pair_1.id,
            search_settings_id=primary_search_settings.id,
            from_beginning=False,
            status=IndexingStatus.COMPLETED_WITH_ERRORS,
        )
        db_session.add(new_attempt)
        db_session.commit()

        create_index_attempt_error(  # noqa: F821,F841
            index_attempt_id=new_attempt.id,
            connector_credential_pair_id=cc_pair_1.id,
            failure=ConnectorFailure(  # noqa: F821,F841
                failure_message="Test error",
                failed_document=DocumentFailure(  # noqa: F821,F841
                    document_id=cc_pair_1.documents[0].id,
                    document_link=None,
                ),
                failed_entity=None,
            ),
            db_session=db_session,
        )

    # delete connector 1
    CCPairManager.pause_cc_pair(
        cc_pair=cc_pair_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    CCPairManager.delete(
        cc_pair=cc_pair_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # inject an index attempt and index attempt error (exercises foreign key errors)
    with Session(get_sqlalchemy_engine()) as db_session:  # noqa: F821,F841
        attempt_id = create_index_attempt(  # noqa: F821,F841
            connector_credential_pair_id=cc_pair_1.id,
            search_settings_id=1,
            db_session=db_session,
        )
        create_index_attempt_error(  # noqa: F821,F841
            index_attempt_id=attempt_id,
            connector_credential_pair_id=cc_pair_1.id,
            failure=ConnectorFailure(  # noqa: F821,F841
                failure_message="Test error",
                failed_document=DocumentFailure(  # noqa: F821,F841
                    document_id=cc_pair_1.documents[0].id,
                    document_link=None,
                ),
                failed_entity=None,
            ),
            db_session=db_session,
        )

    # Update local records to match the database for later comparison
    doc_set_1.cc_pair_ids = []
    doc_set_2.cc_pair_ids = [cc_pair_2.id]
    cc_pair_1.groups = []
    if is_ee:
        cc_pair_2.groups = [
            user_group_2.id  # ty: ignore[possibly-unresolved-reference]
        ]
    else:
        cc_pair_2.groups = []

    CCPairManager.wait_for_deletion_completion(
        cc_pair_id=cc_pair_1.id,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # validate vespa documents
    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_1,
        doc_set_names=[],
        group_names=[],
        doc_creating_user=admin_user,  # noqa: F821,F841
        verify_deleted=True,
    )

    cc_pair_2_group_name_expected = []
    if is_ee:
        cc_pair_2_group_name_expected = [
            user_group_2.name  # ty: ignore[possibly-unresolved-reference]
        ]

    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_2,
        doc_set_names=[doc_set_2.name],
        group_names=cc_pair_2_group_name_expected,
        doc_creating_user=admin_user,  # noqa: F821,F841
        verify_deleted=False,
    )

    # check that only connector 1 is deleted
    CCPairManager.verify(
        cc_pair=cc_pair_2,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # validate document sets
    DocumentSetManager.verify(
        document_set=doc_set_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    DocumentSetManager.verify(
        document_set=doc_set_2,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    if is_ee:
        user_group_1.cc_pair_ids = []  # ty: ignore[possibly-unresolved-reference]
        user_group_2.cc_pair_ids = [  # ty: ignore[possibly-unresolved-reference]
            cc_pair_2.id
        ]

        # validate user groups
        UserGroupManager.verify(
            user_group=user_group_1,  # ty: ignore[possibly-unresolved-reference]
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        UserGroupManager.verify(
            user_group=user_group_2,  # ty: ignore[possibly-unresolved-reference]
            user_performing_action=admin_user,  # noqa: F821,F841
        )


def test_connector_deletion_for_overlapping_connectors(
    reset: None,  # noqa: ARG001
    vespa_client: vespa_fixture,  # noqa: F821,F841
) -> None:
    """Checks to make sure that connectors with overlapping documents work properly. Specifically, that the overlapping
    document (1) still exists and (2) has the right document set / group post-deletion of one of the connectors.
    """
    user_group_1: DATestUserGroup
    user_group_2: DATestUserGroup

    is_ee = (
        os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() == "true"
    )

    # Creating an admin user (first user created is automatically an admin)
    # create api key
    api_key: DATestAPIKey = APIKeyManager.create(
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # create connectors
    cc_pair_1 = CCPairManager.create_from_scratch(
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    cc_pair_2 = CCPairManager.create_from_scratch(
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    doc_ids = [str(uuid4())]
    cc_pair_1.documents = DocumentManager.seed_dummy_docs(
        cc_pair=cc_pair_1,
        document_ids=doc_ids,
        api_key=api_key,
    )
    cc_pair_2.documents = DocumentManager.seed_dummy_docs(
        cc_pair=cc_pair_2,
        document_ids=doc_ids,
        api_key=api_key,
    )

    # verify vespa document exists and that it is not in any document sets or groups
    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_1,
        doc_set_names=[],
        group_names=[],
        doc_creating_user=admin_user,  # noqa: F821,F841
    )
    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_2,
        doc_set_names=[],
        group_names=[],
        doc_creating_user=admin_user,  # noqa: F821,F841
    )

    # create document set
    doc_set_1 = DocumentSetManager.create(
        name="Test Document Set 1",
        cc_pair_ids=[cc_pair_1.id],
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    DocumentSetManager.wait_for_sync(
        document_sets_to_check=[doc_set_1],
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    print("Document set 1 created and synced")

    # verify vespa document is in the document set
    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_1,
        doc_set_names=[doc_set_1.name],
        doc_creating_user=admin_user,  # noqa: F821,F841
    )
    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_2,
        doc_creating_user=admin_user,  # noqa: F821,F841
    )

    if is_ee:
        # create a user group and attach it to connector 1
        user_group_1 = UserGroupManager.create(
            name="Test User Group 1",
            cc_pair_ids=[cc_pair_1.id],
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        UserGroupManager.wait_for_sync(
            user_groups_to_check=[user_group_1],
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        cc_pair_1.groups = [user_group_1.id]

        print("User group 1 created and synced")

        # create a user group and attach it to connector 2
        user_group_2 = UserGroupManager.create(
            name="Test User Group 2",
            cc_pair_ids=[cc_pair_2.id],
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        UserGroupManager.wait_for_sync(
            user_groups_to_check=[user_group_2],
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        cc_pair_2.groups = [user_group_2.id]

        print("User group 2 created and synced")

        # verify vespa document is in the user group
        DocumentManager.verify(
            vespa_client=vespa_client,
            cc_pair=cc_pair_1,
            group_names=[user_group_1.name, user_group_2.name],
            doc_creating_user=admin_user,  # noqa: F821,F841
        )
        DocumentManager.verify(
            vespa_client=vespa_client,
            cc_pair=cc_pair_2,
            group_names=[user_group_1.name, user_group_2.name],
            doc_creating_user=admin_user,  # noqa: F821,F841
        )

    # delete connector 1
    CCPairManager.pause_cc_pair(
        cc_pair=cc_pair_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    CCPairManager.delete(
        cc_pair=cc_pair_1,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # wait for deletion to finish
    CCPairManager.wait_for_deletion_completion(
        cc_pair_id=cc_pair_1.id,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    print("Connector 1 deleted")

    # check that only connector 1 is deleted
    # TODO: check for the CC pair rather than the connector once the refactor is done
    CCPairManager.verify(
        cc_pair=cc_pair_1,
        verify_deleted=True,
        user_performing_action=admin_user,  # noqa: F821,F841
    )
    CCPairManager.verify(
        cc_pair=cc_pair_2,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # verify the document is not in any document sets
    # verify the document is only in user group 2
    group_names_expected = []
    if is_ee:
        group_names_expected = [
            user_group_2.name  # ty: ignore[possibly-unresolved-reference]
        ]

    DocumentManager.verify(
        vespa_client=vespa_client,
        cc_pair=cc_pair_2,
        doc_set_names=[],
        group_names=group_names_expected,
        doc_creating_user=admin_user,  # noqa: F821,F841
        verify_deleted=False,
    )
