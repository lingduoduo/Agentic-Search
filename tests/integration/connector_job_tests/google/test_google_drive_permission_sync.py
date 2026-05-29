import pytest

pytestmark = pytest.mark.skip(reason="requires external services")

import json  # noqa: E402
import os  # noqa: E402
from collections.abc import Generator  # noqa: E402
from datetime import datetime  # noqa: E402
from datetime import timezone  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402

from tests.integration.common_utils.types import DocumentSource  # noqa: E402

# GoogleDriveService removed
# google_utils shared_constants removed
# google_utils shared_constants removed
from tests.integration.common_utils.types import InputType  # noqa: E402
from tests.integration.common_utils.types import AccessType  # noqa: E402
from tests.integration.common_utils.managers.cc_pair import CCPairManager  # noqa: E402
from tests.integration.common_utils.managers.connector import ConnectorManager  # noqa: E402
from tests.integration.common_utils.managers.credential import CredentialManager  # noqa: E402
from tests.integration.common_utils.managers.document_search import (  # noqa: E402
    DocumentSearchManager,
)
from tests.integration.common_utils.test_models import DATestCCPair  # noqa: E402
from tests.integration.common_utils.test_models import DATestConnector  # noqa: E402
from tests.integration.common_utils.test_models import DATestCredential  # noqa: E402
from tests.integration.common_utils.test_models import DATestUser  # noqa: E402

# vespa_fixture removed — no Vespa in this deployment
from tests.integration.connector_job_tests.google.google_drive_api_utils import (  # noqa: E402
    GoogleDriveManager,
)


@pytest.fixture()
def google_drive_test_env_setup() -> Generator[
    tuple[GoogleDriveService, str, DATestCCPair, DATestUser, DATestUser, DATestUser],  # noqa: F821,F841
    None,
    None,
]:
    # Creating an admin user (first user created is automatically an admin)
    # Creating a non-admin user
    # Creating a non-admin user

    service_account_key = os.environ["FULL_CONTROL_DRIVE_SERVICE_ACCOUNT"]
    drive_id: str | None = None
    drive_service: GoogleDriveService | None = None  # noqa: F821,F841

    try:
        credentials = {
            DB_CREDENTIALS_PRIMARY_ADMIN_KEY: admin_user.email,  # noqa: F821,F841
            DB_CREDENTIALS_DICT_SERVICE_ACCOUNT_KEY: service_account_key,  # noqa: F821,F841
        }

        # Setup Google Drive
        drive_service = GoogleDriveManager.create_impersonated_drive_service(
            json.loads(service_account_key),
            admin_user.email,  # noqa: F821,F841
        )
        test_id = str(uuid4())
        drive_id = GoogleDriveManager.create_shared_drive(
            drive_service,
            admin_user.email,  # noqa: F821
            test_id,  # noqa: F821,F841
        )

        # Setup Onyx infrastructure

        before = datetime.now(timezone.utc)
        credential: DATestCredential = CredentialManager.create(
            source=DocumentSource.GOOGLE_DRIVE,
            credential_json=credentials,
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        connector: DATestConnector = ConnectorManager.create(
            name="Google Drive Test",
            input_type=InputType.POLL,
            source=DocumentSource.GOOGLE_DRIVE,
            connector_specific_config={
                "shared_drive_urls": f"https://drive.google.com/drive/folders/{drive_id}"
            },
            access_type=AccessType.SYNC,
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        cc_pair: DATestCCPair = CCPairManager.create(
            credential_id=credential.id,
            connector_id=connector.id,
            access_type=AccessType.SYNC,
            user_performing_action=admin_user,  # noqa: F821,F841
        )
        CCPairManager.wait_for_indexing_completion(
            cc_pair=cc_pair,
            after=before,
            user_performing_action=admin_user,  # noqa: F821,F841
        )

        yield drive_service, drive_id, cc_pair, admin_user, test_user_1, test_user_2  # noqa: F821,F841

    except json.JSONDecodeError:
        pytest.skip("FULL_CONTROL_DRIVE_SERVICE_ACCOUNT is not valid JSON")
    finally:
        # Cleanup drive and file
        if drive_id is not None:
            GoogleDriveManager.cleanup_drive(drive_service, drive_id)


@pytest.mark.xfail(reason="Needs to be tested for flakiness")
def test_google_permission_sync(
    reset: None,  # noqa: ARG001
    vespa_client: vespa_fixture,  # noqa: ARG001,F821
    google_drive_test_env_setup: tuple[
        GoogleDriveService, str, DATestCCPair, DATestUser, DATestUser, DATestUser  # noqa: F821,F841
    ],
) -> None:
    (
        drive_service,
        drive_id,
        cc_pair,
        admin_user,
        test_user_1,
        test_user_2,
    ) = google_drive_test_env_setup

    # ----------------------BASELINE TEST----------------------

    # Create empty test doc in drive

    # Append text to doc
    doc_text_1 = "The secret number is 12345"
    GoogleDriveManager.append_text_to_doc(drive_service, doc_id_1, doc_text_1)  # noqa: F821,F841

    # run indexing
    CCPairManager.run_once(
        cc_pair, from_beginning=True, user_performing_action=admin_user
    )
    CCPairManager.wait_for_indexing_completion(
        cc_pair=cc_pair,
        after=before,  # noqa: F821
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # run permission sync
    CCPairManager.sync(
        cc_pair=cc_pair,
        user_performing_action=admin_user,
    )
    CCPairManager.wait_for_sync(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        number_of_updated_docs=1,
        user_performing_action=admin_user,
    )

    # Verify admin has access to document
    admin_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=admin_user
    )
    assert doc_text_1 in [result.strip("\ufeff") for result in admin_results]

    # Verify test_user_1 cannot access document
    user1_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=test_user_1
    )
    assert doc_text_1 not in [result.strip("\ufeff") for result in user1_results]

    # ----------------------GRANT USER 1 DOC PERMISSIONS TEST--------------------------

    # Grant user 1 access to document 1
    GoogleDriveManager.update_file_permissions(
        drive_service=drive_service,
        file_id=doc_id_1,  # noqa: F821,F841
        email=test_user_1.email,
        role="reader",
    )

    # Create a second doc in the drive which user 1 should not have access to
    doc_text_2 = "The secret number is 67890"
    GoogleDriveManager.append_text_to_doc(drive_service, doc_id_2, doc_text_2)  # noqa: F821,F841

    # Run indexing
    CCPairManager.run_once(
        cc_pair, from_beginning=True, user_performing_action=admin_user
    )
    CCPairManager.wait_for_indexing_completion(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        user_performing_action=admin_user,
    )

    # Run permission sync
    CCPairManager.sync(
        cc_pair=cc_pair,
        user_performing_action=admin_user,
    )
    CCPairManager.wait_for_sync(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        number_of_updated_docs=1,
        user_performing_action=admin_user,
    )

    # Verify admin can access both documents
    admin_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=admin_user
    )
    assert {doc_text_1, doc_text_2} == {
        result.strip("\ufeff") for result in admin_results
    }

    # Verify user 1 can access document 1
    user1_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=test_user_1
    )
    assert doc_text_1 in [result.strip("\ufeff") for result in user1_results]

    # Verify user 1 cannot access document 2
    user1_results_2 = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=test_user_1
    )
    assert doc_text_2 not in [result.strip("\ufeff") for result in user1_results_2]

    # ----------------------REMOVE USER 1 DOC PERMISSIONS TEST--------------------------

    # Remove user 1 access to document 1
    GoogleDriveManager.remove_file_permissions(
        drive_service=drive_service,
        file_id=doc_id_1,  # noqa: F821
        email=test_user_1.email,  # noqa: F821,F841
    )
    # Run permission sync
    CCPairManager.sync(
        cc_pair=cc_pair,
        user_performing_action=admin_user,
    )
    CCPairManager.wait_for_sync(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        number_of_updated_docs=1,
        user_performing_action=admin_user,
    )

    # Verify admin can access both documents
    admin_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=admin_user
    )
    assert {doc_text_1, doc_text_2} == {
        result.strip("\ufeff") for result in admin_results
    }

    # Verify user 1 cannot access either document
    user1_results = DocumentSearchManager.search_documents(
        query="secret numbers", user_performing_action=test_user_1
    )
    assert {result.strip("\ufeff") for result in user1_results} == set()

    # ----------------------GRANT USER 1 DRIVE PERMISSIONS TEST--------------------------

    # Grant user 1 access to drive
    GoogleDriveManager.update_file_permissions(
        drive_service=drive_service,
        file_id=drive_id,
        email=test_user_1.email,
        role="reader",
    )

    # Run permission sync
    CCPairManager.sync(
        cc_pair=cc_pair,
        user_performing_action=admin_user,
    )

    CCPairManager.wait_for_sync(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        number_of_updated_docs=2,
        user_performing_action=admin_user,
        # if we are only updating the group definition for this test we use this varaiable,
        # since it doesn't result in a vespa sync so we don't want to wait for it
    )

    # Verify user 1 can access both documents
    user1_results = DocumentSearchManager.search_documents(
        query="secret numbers", user_performing_action=test_user_1
    )
    assert {doc_text_1, doc_text_2} == {
        result.strip("\ufeff") for result in user1_results
    }

    # ----------------------MAKE DRIVE PUBLIC TEST--------------------------

    # Unable to make drive itself public as Google's security policies prevent this, so we make the documents public instead
    GoogleDriveManager.make_file_public(drive_service, doc_id_2)  # noqa: F821,F841

    # Run permission sync
    CCPairManager.sync(
        cc_pair=cc_pair,
        user_performing_action=admin_user,
    )
    CCPairManager.wait_for_sync(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        number_of_updated_docs=2,
        user_performing_action=admin_user,
    )

    # Verify all users can access both documents
    admin_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=admin_user
    )
    assert {doc_text_1, doc_text_2} == {
        result.strip("\ufeff") for result in admin_results
    }

    user1_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=test_user_1
    )
    assert {doc_text_1, doc_text_2} == {
        result.strip("\ufeff") for result in user1_results
    }

    user2_results = DocumentSearchManager.search_documents(
        query="secret number", user_performing_action=test_user_2
    )
    assert {doc_text_1, doc_text_2} == {
        result.strip("\ufeff") for result in user2_results
    }
