"""
Integration test for forced tool use to verify that web_search can be forced.
This test verifies that forcing a tool use works through the complete API flow.
"""

import pytest
from sqlalchemy import select

# get_session_with_current_tenant removed — no direct DB access
# Tool ORM removed — use HTTP
from tests.integration.common_utils.managers.chat import ChatSessionManager
from tests.integration.common_utils.test_models import DATestImageGenerationConfig
from tests.integration.common_utils.test_models import DATestUser
from tests.integration.common_utils.test_models import ToolName


def test_force_tool_use(
    basic_user: DATestUser,
    image_generation_config: DATestImageGenerationConfig,  # noqa: ARG001
) -> None:
    with get_session_with_current_tenant() as db_session:  # noqa: F821,F841
        image_generation_tool = db_session.execute(
            select(Tool).where(Tool.in_code_tool_id == "ImageGenerationTool")  # noqa: F821,F841
        ).scalar_one_or_none()
        assert image_generation_tool is not None, "ImageGenerationTool must exist"
        image_generation_tool_id = image_generation_tool.id

    # Create a chat session

    # Send a simple message that wouldn't normally trigger image generation
    # but force the image generation tool to be used
    message = "hi"

    analyzed_response = ChatSessionManager.send_message(
        chat_session_id=chat_session.id,  # noqa: F821,F841
        message=message,
        user_performing_action=basic_user,
        forced_tool_ids=[image_generation_tool_id],
    )

    assert analyzed_response.error is None, "Chat response should not have an error"

    image_generation_tool_used = any(
        tool.tool_name == ToolName.IMAGE_GENERATION
        for tool in analyzed_response.used_tools
    )
    assert image_generation_tool_used, (
        "Image generation tool should have been forced to run"
    )


if __name__ == "__main__":
    # Run with: python -m dotenv -f .vscode/.env run --
    # python -m pytest tests/integration/tests/tools/test_force_tool_use.py -v -s
    pytest.main([__file__, "-v", "-s"])
