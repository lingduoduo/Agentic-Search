"""Integration tests to check broader image generation config flow endpoints."""

import pytest

# image_generation_tool removed
from tests.integration.common_utils.managers.image_generation import (
    ImageGenerationConfigManager,
)
from tests.integration.common_utils.managers.llm_provider import LLMProviderManager
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.reset import reset_all
from tests.integration.common_utils.test_models import DATestLLMProvider
from tests.integration.common_utils.test_models import DATestUser

IMAGE_GENERATION_TOOL_NAME = ImageGenerationTool.NAME  # noqa: F821,F841


@pytest.fixture(scope="module")
def setup_image_generation_tests() -> tuple[DATestUser, DATestLLMProvider]:
    """Module-scoped fixture that runs once for all tests in this module.

    - Resets DB once at the start of the module
    - Creates admin user once
    - Creates LLM provider once (for clone-mode test)
    - Returns (admin_user, llm_provider) tuple for all tests to use
    """
    reset_all()
    admin_user = UserManager.create(name="admin_user")
    llm_provider = LLMProviderManager.create(user_performing_action=admin_user)
    return admin_user, llm_provider


def test_vertex_creds_upload_image_tool_visibility(
    setup_image_generation_tests: tuple[DATestUser, DATestLLMProvider],
) -> None:
    """
    Tests the following scenario:
    1. No image model added so tool not visible
    2. Vertex AI creds uploaded
    3. Image model added so tool visible
    """
    admin_user, _ = setup_image_generation_tests

    # 1. Check the tools and check that image generation tool is not visible yet
    assert not any(tool.name == IMAGE_GENERATION_TOOL_NAME for tool in tools)  # noqa: F821,F841

    # 2. Upload vertex ai credentials
    config = ImageGenerationConfigManager.create(
        image_provider_id="gemini-2.5-flash-image",
        model_name="gemini-2.5-flash-image",
        provider="vertex_ai",
        custom_config={
            "vertex_credentials": {
                "type": "service_account",
                "project_id": "test-project-id",
                "private_key_id": "test-private-key-id",
                "private_key": "test-private-key",
            },
            "vertex_location": "test-location",
        },
        user_performing_action=admin_user,
        is_default=True,
    )

    assert config.image_provider_id == "gemini-2.5-flash-image"
    assert config.model_name == "gemini-2.5-flash-image"

    # 3. Check that the tool is visible
    assert any(tool.name == IMAGE_GENERATION_TOOL_NAME for tool in tools)  # noqa: F821,F841
