"""Integration test fixtures.

Tests run against a live server at API_SERVER_URL (default localhost:8080).
No direct database access — all state is managed via the HTTP API.
"""

import os
from collections.abc import Callable

import pytest

os.environ["INTEGRATION_TESTS_MODE"] = "true"

from tests.integration.common_utils.constants import ADMIN_USER_NAME  # noqa: E402
from tests.integration.common_utils.constants import GENERAL_HEADERS  # noqa: E402
from tests.integration.common_utils.managers.api_key import APIKeyManager  # noqa: E402
from tests.integration.common_utils.managers.llm_provider import LLMProviderManager  # noqa: E402
from tests.integration.common_utils.managers.user import UserManager  # noqa: E402
from tests.integration.common_utils.managers.user import build_email  # noqa: E402
from tests.integration.common_utils.managers.user import DEFAULT_PASSWORD  # noqa: E402
from tests.integration.common_utils.reset import _seed_dev_license_if_set  # noqa: E402
from tests.integration.common_utils.reset import reset_all  # noqa: E402
from tests.integration.common_utils.reset import reset_all_multitenant  # noqa: E402
from tests.integration.common_utils.test_models import DATestAPIKey  # noqa: E402
from tests.integration.common_utils.test_models import DATestLLMProvider  # noqa: E402
from tests.integration.common_utils.test_models import DATestUser  # noqa: E402
from tests.integration.common_utils.test_models import SimpleTestDocument  # noqa: E402
from tests.integration.common_utils.types import UserRole  # noqa: E402

BASIC_USER_NAME = "basic_user"

DocumentBuilderType = Callable[[list[str]], list[SimpleTestDocument]]


def load_env_vars(env_file: str = ".env") -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, env_file)
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value.strip())
        print("Successfully loaded environment variables")
    except FileNotFoundError:
        print(f"File {env_file} not found")


load_env_vars()


@pytest.fixture(scope="session", autouse=True)
def seed_dev_license_for_session() -> None:
    """Seed dev license once at session start if ONYX_DEV_LICENSE is set."""
    _seed_dev_license_if_set()


@pytest.fixture
def reset() -> None:
    reset_all()


@pytest.fixture
def new_admin_user(reset: None) -> DATestUser:  # noqa: ARG001
    return UserManager.create(name=ADMIN_USER_NAME)


@pytest.fixture
def admin_user() -> DATestUser:
    try:
        user = UserManager.create(name=ADMIN_USER_NAME)
        if not UserManager.is_role(user, UserRole.ADMIN):
            print("Trying to reset")
            reset_all()
            user = UserManager.create(name=ADMIN_USER_NAME)
        return user
    except Exception as e:
        print(f"Failed to create admin user: {e}")

    try:
        user = UserManager.login_as_user(
            DATestUser(
                id="",
                email=build_email("admin_user"),
                password=DEFAULT_PASSWORD,
                headers=GENERAL_HEADERS,
                role=UserRole.ADMIN,
                is_active=True,
            )
        )
        if not UserManager.is_role(user, UserRole.ADMIN):
            reset_all()
            return UserManager.create(name=ADMIN_USER_NAME)
        return user
    except Exception as e:
        print(f"Failed to create or login as admin user: {e}")

    raise RuntimeError("Failed to create or login as admin user")


@pytest.fixture
def basic_user(admin_user: DATestUser) -> DATestUser:  # noqa: ARG001
    try:
        user = UserManager.create(name=BASIC_USER_NAME)
        if user.role != UserRole.BASIC:
            raise RuntimeError(
                f"Created user {BASIC_USER_NAME} does not have BASIC role"
            )
        return user
    except Exception as e:
        print(f"Failed to create basic user, trying to login: {e}")

    user = UserManager.login_as_user(
        DATestUser(
            id="",
            email=build_email(BASIC_USER_NAME),
            password=DEFAULT_PASSWORD,
            headers=GENERAL_HEADERS,
            role=UserRole.BASIC,
            is_active=True,
        )
    )
    if not UserManager.is_role(user, UserRole.BASIC):
        raise RuntimeError(f"User {BASIC_USER_NAME} does not have BASIC role")
    return user


@pytest.fixture(scope="session")
def reset_multitenant() -> None:
    reset_all_multitenant()


@pytest.fixture
def llm_provider(admin_user: DATestUser) -> DATestLLMProvider:
    return LLMProviderManager.create(user_performing_action=admin_user)


@pytest.fixture
def api_key(admin_user: DATestUser) -> DATestAPIKey:
    return APIKeyManager.create(user_performing_action=admin_user)


@pytest.fixture
def document_builder(admin_user: DATestUser) -> DocumentBuilderType:
    from tests.integration.common_utils.managers.cc_pair import CCPairManager
    from tests.integration.common_utils.managers.document import DocumentManager
    from tests.integration.common_utils.types import DocumentSource

    api_key_: DATestAPIKey = APIKeyManager.create(user_performing_action=admin_user)
    cc_pair_1 = CCPairManager.create_from_scratch(
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,
    )

    def _document_builder(contents: list[str]) -> list[SimpleTestDocument]:
        return [
            DocumentManager.seed_doc_with_content(
                cc_pair=cc_pair_1,
                content=content,
                api_key=api_key_,
            )
            for content in contents
        ]

    return _document_builder


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:  # noqa: ARG001
    print(f"\nTest start: {nodeid}")


def pytest_runtest_logfinish(nodeid: str, location: tuple) -> None:  # noqa: ARG001
    print(f"\nTest end: {nodeid}")
