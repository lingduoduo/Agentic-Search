import json
import logging
import os

logger = logging.getLogger(__name__)

IMAGE_GENERATION_OUTPUT_FORMAT = os.environ.get(
    "IMAGE_GENERATION_OUTPUT_FORMAT", "b64_json"
)

CUSTOM_TOOL_PASS_THROUGH_HEADERS: list[str] | None = None
_CUSTOM_TOOL_PASS_THROUGH_HEADERS_RAW = os.environ.get(
    "CUSTOM_TOOL_PASS_THROUGH_HEADERS"
)
if _CUSTOM_TOOL_PASS_THROUGH_HEADERS_RAW:
    try:
        CUSTOM_TOOL_PASS_THROUGH_HEADERS = json.loads(
            _CUSTOM_TOOL_PASS_THROUGH_HEADERS_RAW
        )
    except Exception:
        logger.error(
            "Failed to parse CUSTOM_TOOL_PASS_THROUGH_HEADERS, must be a valid JSON object"
        )
