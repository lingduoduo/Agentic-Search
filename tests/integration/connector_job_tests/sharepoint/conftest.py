import os
from collections.abc import Generator

import pytest

from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.types import InputType


class SharepointAuthMethod:
    CLIENT_CREDENTIALS = "client_credentials"


from tests.integration.common_utils.types import AccessType  # noqa: E402
from tests.integration.common_utils.managers.cc_pair import CCPairManager  # noqa: E402
from tests.integration.common_utils.managers.connector import ConnectorManager  # noqa: E402
from tests.integration.common_utils.managers.credential import CredentialManager  # noqa: E402
from tests.integration.common_utils.managers.user import UserManager  # noqa: E402
from tests.integration.common_utils.test_models import DATestCCPair  # noqa: E402
from tests.integration.common_utils.test_models import DATestConnector  # noqa: E402
from tests.integration.common_utils.test_models import DATestCredential  # noqa: E402
from tests.integration.common_utils.test_models import DATestUser  # noqa: E402

SharepointTestEnvSetupTuple = tuple[
    DATestUser,  # admin_user
    DATestUser,  # regular_user_1
    DATestUser,  # regular_user_2
    DATestCredential,
    DATestConnector,
    DATestCCPair,
]


@pytest.fixture(scope="module")
def sharepoint_test_env_setup() -> Generator[SharepointTestEnvSetupTuple]:
    # Reset all data before running the test
    # Required environment variables for SharePoint certificate authentication
    sp_private_key = os.environ.get("PERM_SYNC_SHAREPOINT_PRIVATE_KEY")
    sp_certificate_password = os.environ.get(
        "PERM_SYNC_SHAREPOINT_CERTIFICATE_PASSWORD"
    )
    sp_directory_id = os.environ.get("PERM_SYNC_SHAREPOINT_DIRECTORY_ID")
    sharepoint_sites = "https://danswerai.sharepoint.com/sites/Permisisonsync"
    admin_email = "admin@onyx.app"  # noqa: F821,F841
    user1_email = "subash@onyx.app"
    user2_email = "raunak@onyx.app"

    if not sp_private_key or not sp_certificate_password or not sp_directory_id:
        pytest.skip("Skipping test because required environment variables are not set")

    # Certificate-based credentials
    credentials = {
        "authentication_method": SharepointAuthMethod.CERTIFICATE.value,
        "sp_client_id": sp_client_id,  # noqa: F821,F841
        "sp_private_key": sp_private_key,
        "sp_certificate_password": sp_certificate_password,
        "sp_directory_id": sp_directory_id,
    }

    # Create users
    regular_user_1: DATestUser = UserManager.create(email=user1_email)
    regular_user_2: DATestUser = UserManager.create(email=user2_email)

    # Create LLM provider for search functionality

    # Create credential
    credential: DATestCredential = CredentialManager.create(
        source=DocumentSource.SHAREPOINT,
        credential_json=credentials,
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Create connector with SharePoint-specific configuration
    connector: DATestConnector = ConnectorManager.create(
        name="SharePoint Test",
        input_type=InputType.POLL,
        source=DocumentSource.SHAREPOINT,
        connector_specific_config={
            "sites": sharepoint_sites.split(","),
            "treat_sharing_link_as_public": True,
        },
        access_type=AccessType.SYNC,  # Enable permission sync
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Create CC pair with permission sync enabled
    cc_pair: DATestCCPair = CCPairManager.create(
        credential_id=credential.id,
        connector_id=connector.id,
        access_type=AccessType.SYNC,  # Enable permission sync
        user_performing_action=admin_user,  # noqa: F821,F841
    )

    # Wait for both indexing and permission sync to complete
    CCPairManager.wait_for_indexing_completion(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        user_performing_action=admin_user,  # noqa: F821,F841
        timeout=float("inf"),
    )

    # Wait for permission sync completion specifically
    CCPairManager.wait_for_sync(
        cc_pair=cc_pair,
        after=before,  # noqa: F821,F841
        user_performing_action=admin_user,  # noqa: F821,F841
        timeout=float("inf"),
    )

    yield admin_user, regular_user_1, regular_user_2, credential, connector, cc_pair  # noqa: F821,F841
