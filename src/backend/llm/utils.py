"""LLM utility functions.

Provides model lookup, token counting, error classification, and passthrough
kwargs helpers that the LiteLLM backend (multi_llm.py) depends on.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from .constants import LlmProviderNames

if TYPE_CHECKING:
    from .interfaces import LLM, LLMUserIdentity
    from .model_response import ModelResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
GEN_AI_MAX_TOKENS: int | None = (
    int(os.environ["GEN_AI_MAX_TOKENS"]) if "GEN_AI_MAX_TOKENS" in os.environ else None
)
GEN_AI_MODEL_FALLBACK_MAX_TOKENS = int(
    os.environ.get("GEN_AI_MODEL_FALLBACK_MAX_TOKENS", "32000")
)
GEN_AI_NUM_RESERVED_OUTPUT_TOKENS = int(
    os.environ.get("GEN_AI_NUM_RESERVED_OUTPUT_TOKENS", "512")
)
SEND_USER_METADATA_TO_LLM_PROVIDER = (
    os.environ.get("SEND_USER_METADATA_TO_LLM_PROVIDER", "false").lower() == "true"
)
LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS: dict[str, str] = {}

MAX_LITELLM_USER_ID_LENGTH = 64

_TWELVE_LABS_PEGASUS_MODEL_NAMES = [
    "us.twelvelabs.pegasus-1-2-v1:0",
    "us.twelvelabs.pegasus-1-2-v1",
    "twelvelabs/us.twelvelabs.pegasus-1-2-v1:0",
    "twelvelabs/us.twelvelabs.pegasus-1-2-v1",
]
_TWELVE_LABS_PEGASUS_OUTPUT_TOKENS = max(512, GEN_AI_MODEL_FALLBACK_MAX_TOKENS // 4)
CUSTOM_LITELLM_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    model_name: {
        "max_input_tokens": GEN_AI_MODEL_FALLBACK_MAX_TOKENS,
        "max_output_tokens": _TWELVE_LABS_PEGASUS_OUTPUT_TOKENS,
        "max_tokens": GEN_AI_MODEL_FALLBACK_MAX_TOKENS,
        "supports_reasoning": False,
        "supports_vision": False,
    }
    for model_name in _TWELVE_LABS_PEGASUS_MODEL_NAMES
}

# ---------------------------------------------------------------------------
# User-identity helpers
# ---------------------------------------------------------------------------


def truncate_litellm_user_id(user_id: str) -> str:
    """Truncate the LiteLLM `user` field to its maximum allowed length."""
    if len(user_id) <= MAX_LITELLM_USER_ID_LENGTH:
        return user_id
    logger.warning(
        "User ID exceeds %d chars (len=%d); truncating for LiteLLM compatibility.",
        MAX_LITELLM_USER_ID_LENGTH,
        len(user_id),
    )
    return user_id[:MAX_LITELLM_USER_ID_LENGTH]


def build_litellm_passthrough_kwargs(
    model_kwargs: dict[str, Any],
    user_identity: "LLMUserIdentity | None",
) -> dict[str, Any]:
    """Build kwargs passed directly to LiteLLM.

    Returns `model_kwargs` unchanged unless user metadata needs to be added,
    in which case a shallow copy is returned to avoid cross-request mutation.
    """
    if not (SEND_USER_METADATA_TO_LLM_PROVIDER and user_identity):
        return model_kwargs

    passthrough_kwargs = copy.deepcopy(model_kwargs)

    if user_identity.user_id:
        passthrough_kwargs["user"] = truncate_litellm_user_id(user_identity.user_id)

    if user_identity.session_id:
        existing_metadata = passthrough_kwargs.get("metadata")
        metadata: dict[str, Any] | None
        if existing_metadata is None:
            metadata = {}
        elif isinstance(existing_metadata, dict):
            metadata = copy.deepcopy(existing_metadata)
        else:
            metadata = None

        if metadata is not None:
            metadata["session_id"] = user_identity.session_id
            passthrough_kwargs["metadata"] = metadata

    return passthrough_kwargs


# ---------------------------------------------------------------------------
# Exception helpers
# ---------------------------------------------------------------------------


def _unwrap_nested_exception(error: Exception) -> Exception:
    """Traverse common exception wrappers to surface the underlying error."""
    visited: set[int] = set()
    current = error
    for _ in range(100):
        visited.add(id(current))
        candidate: Exception | None = None
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, Exception):
            candidate = cause
        elif (
            hasattr(current, "args")
            and len(getattr(current, "args")) == 1
            and isinstance(current.args[0], Exception)
        ):
            candidate = current.args[0]
        if candidate is None or id(candidate) in visited:
            break
        current = candidate
    return current


def litellm_exception_to_error_msg(
    e: Exception,
    llm: "LLM",
    fallback_to_error_msg: bool = False,
    custom_error_msg_mappings: dict[str, str] | None = None,
) -> tuple[str, str, bool]:
    """Convert a LiteLLM exception to a user-friendly message.

    Returns:
        (error_message, error_code, is_retryable)
    """
    from litellm.exceptions import (
        APIConnectionError,
        APIError,
        AuthenticationError,
        BadRequestError,
        BudgetExceededError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
        UnprocessableEntityError,
    )

    _mappings = custom_error_msg_mappings or LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS
    core_exception = _unwrap_nested_exception(e)
    error_msg = str(core_exception)
    error_code = "UNKNOWN_ERROR"
    is_retryable = True

    if _mappings:
        for pattern, custom_msg in _mappings.items():
            if pattern in error_msg:
                return custom_msg, "CUSTOM_ERROR", True

    if isinstance(core_exception, BadRequestError):
        error_msg = "Bad request: The server couldn't process your request."
        error_code = "BAD_REQUEST"
    elif isinstance(core_exception, AuthenticationError):
        error_msg = "Authentication failed: Please check your API key and credentials."
        error_code = "AUTH_ERROR"
        is_retryable = False
    elif isinstance(core_exception, PermissionDeniedError):
        error_msg = "Permission denied: You don't have access to this model."
        error_code = "PERMISSION_DENIED"
        is_retryable = False
    elif isinstance(core_exception, NotFoundError):
        error_msg = "Resource not found: The requested resource doesn't exist."
        error_code = "NOT_FOUND"
        is_retryable = False
    elif isinstance(core_exception, UnprocessableEntityError):
        error_msg = "Unprocessable entity: The server couldn't process your request."
        error_code = "UNPROCESSABLE_ENTITY"
    elif isinstance(core_exception, RateLimitError):
        provider_name = (
            llm.config.model_provider if llm is not None else "The LLM provider"
        )
        upstream_detail = str(core_exception).strip()
        if ":" in upstream_detail and upstream_detail.lower().startswith(
            "ratelimiterror"
        ):
            upstream_detail = upstream_detail.split(":", 1)[1].strip()
        upstream_lower = upstream_detail.lower()
        if (
            "insufficient_quota" in upstream_lower
            or "exceeded your current quota" in upstream_lower
        ):
            error_msg = f"{provider_name} quota exceeded: {upstream_detail}"
            error_code = "BUDGET_EXCEEDED"
            is_retryable = False
        else:
            error_msg = f"{provider_name} rate limit: {upstream_detail}"
            error_code = "RATE_LIMIT"
    elif isinstance(core_exception, ServiceUnavailableError):
        provider_name = (
            llm.config.model_provider if llm is not None else "The LLM provider"
        )
        if "Too many connections" in error_msg or "BedrockException" in error_msg:
            error_msg = (
                f"{provider_name} has too many connections. Please try again shortly."
            )
        else:
            error_msg = f"{provider_name} service error: {core_exception}"
        error_code = "SERVICE_UNAVAILABLE"
    elif isinstance(core_exception, ContextWindowExceededError):
        error_msg = "Context window exceeded: Your input is too long for the model."
        error_code = "CONTEXT_TOO_LONG"
        is_retryable = False
    elif isinstance(core_exception, ContentPolicyViolationError):
        error_msg = "Content policy violation: Please revise your input."
        error_code = "CONTENT_POLICY"
        is_retryable = False
    elif isinstance(core_exception, APIConnectionError):
        error_msg = (
            "API connection error: Failed to connect. Check your internet connection."
        )
        error_code = "CONNECTION_ERROR"
    elif isinstance(core_exception, BudgetExceededError):
        error_msg = (
            "Budget exceeded: You've exceeded your allocated budget for API usage."
        )
        error_code = "BUDGET_EXCEEDED"
        is_retryable = False
    elif isinstance(core_exception, Timeout):
        error_msg = "Request timed out. Please try again."
        error_code = "CONNECTION_ERROR"
    elif isinstance(core_exception, APIError):
        error_msg = f"API error: {core_exception}"
        error_code = "API_ERROR"
    elif not fallback_to_error_msg:
        error_msg = "An unexpected error occurred. Please try again later."
        error_code = "UNKNOWN_ERROR"

    return error_msg, error_code, is_retryable


def llm_response_to_string(message: "ModelResponse") -> str:
    if not isinstance(message.choice.message.content, str):
        raise RuntimeError("LLM message not in expected format.")
    return message.choice.message.content


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def check_number_of_tokens(
    text: str, encode_fn: Callable[[str], list] | None = None
) -> int:
    """Count tokens using the provided encoder or cl100k_base as fallback."""
    import tiktoken

    if encode_fn is None:
        encode_fn = tiktoken.get_encoding("cl100k_base").encode
    return len(encode_fn(text))


# ---------------------------------------------------------------------------
# Credential scrubbing
# ---------------------------------------------------------------------------

SENSITIVE_CUSTOM_CONFIG_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
        "vertex_credentials",
        "aws_secret_access_key",
        "aws_access_key_id",
        "aws_bearer_token_bedrock",
        "private_key",
        "api_key",
        "secret",
        "password",
        "token",
        "credential",
    }
)
_SCRUB_PLACEHOLDER = "[REDACTED]"


def is_sensitive_custom_config_key(key: str) -> bool:
    key_lower = key.lower()
    return any(f in key_lower for f in SENSITIVE_CUSTOM_CONFIG_KEY_FRAGMENTS)


def scrub_sensitive_values(message: str, secrets: Iterable[str | None]) -> str:
    if not message:
        return message
    scrubbed = message
    for secret in secrets:
        if not secret or len(secret) < 4:
            continue
        scrubbed = scrubbed.replace(secret, _SCRUB_PLACEHOLDER)
    return scrubbed


def collect_llm_credential_values(llm: "LLM | None") -> list[str]:
    if llm is None:
        return []
    secrets: list[str] = []
    if llm.config.api_key:
        secrets.append(llm.config.api_key)
    for key, value in (llm.config.custom_config or {}).items():
        if isinstance(value, str) and value and is_sensitive_custom_config_key(key):
            secrets.append(value)
    return secrets


def test_llm(llm: "LLM") -> str | None:
    """Probe an LLM and return ``None`` on success or a sanitized error message."""
    from .models import UserMessage

    secrets = collect_llm_credential_values(llm)
    error_msg: str | None = None
    for _ in range(2):
        try:
            llm.invoke(UserMessage(content="Do not respond"), max_tokens=50)
            return None
        except Exception as exc:
            logger.warning("Failed to call LLM: %s", exc)
            safe_msg, _, _ = litellm_exception_to_error_msg(
                exc, llm, fallback_to_error_msg=False
            )
            error_msg = scrub_sensitive_values(safe_msg, secrets)
    return error_msg


# ---------------------------------------------------------------------------
# Model map helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_model_map() -> dict:
    """Return a copy of litellm.model_cost with provider prefixes stripped as fallback keys."""
    import litellm

    DIVIDER = "/"
    original_map = dict(litellm.model_cost)
    starting_map = copy.deepcopy(original_map)
    for key in original_map:
        if DIVIDER in key:
            truncated_key = key.split(DIVIDER)[-1]
            if truncated_key in original_map:
                continue
            existing = starting_map.get(truncated_key)
            potential = original_map[key]
            if not existing or len(potential) > len(existing):
                starting_map[truncated_key] = potential

    for model_name, metadata in CUSTOM_LITELLM_MODEL_OVERRIDES.items():
        if model_name not in starting_map:
            starting_map[model_name] = copy.deepcopy(metadata)

    return starting_map


def _strip_extra_provider_from_model_name(model_name: str) -> str:
    return model_name.split("/")[1] if "/" in model_name else model_name


def _strip_colon_from_model_name(model_name: str) -> str:
    return ":".join(model_name.split(":")[:-1]) if ":" in model_name else model_name


def find_model_obj(model_map: dict, provider: str, model_name: str) -> dict | None:
    stripped = _strip_extra_provider_from_model_name(model_name)
    candidates = [
        model_name,
        stripped,
        _strip_colon_from_model_name(model_name),
        _strip_colon_from_model_name(stripped),
    ]
    for name in candidates:
        obj = model_map.get(f"{provider}/{name}")
        if obj:
            return obj
    for name in candidates:
        obj = model_map.get(name)
        if obj:
            return obj
    return None


def llm_max_input_tokens(
    model_map: dict,
    model_name: str,
    model_provider: str,
) -> int:
    if GEN_AI_MAX_TOKENS:
        return GEN_AI_MAX_TOKENS

    model_obj = find_model_obj(model_map, model_provider, model_name)
    if not model_obj:
        logger.warning(
            "Model '%s' not in LiteLLM; falling back to %d tokens.",
            model_name,
            GEN_AI_MODEL_FALLBACK_MAX_TOKENS,
        )
        return GEN_AI_MODEL_FALLBACK_MAX_TOKENS

    max_input = model_obj.get("max_input_tokens")
    if max_input is not None:
        return max_input
    max_total = model_obj.get("max_tokens")
    if max_total is not None:
        return max_total

    logger.warning(
        "No max tokens for '%s'; falling back to %d.",
        model_name,
        GEN_AI_MODEL_FALLBACK_MAX_TOKENS,
    )
    return GEN_AI_MODEL_FALLBACK_MAX_TOKENS


def get_llm_max_output_tokens(
    model_map: dict,
    model_name: str,
    model_provider: str,
) -> int:
    default = GEN_AI_MODEL_FALLBACK_MAX_TOKENS
    model_obj = model_map.get(f"{model_provider}/{model_name}") or model_map.get(
        model_name
    )
    if not model_obj:
        return default
    max_output = model_obj.get("max_output_tokens")
    if max_output is not None:
        return max_output
    max_total = model_obj.get("max_tokens")
    if max_total is not None:
        return int(max_total * 0.1)
    return default


def get_max_input_tokens(
    model_name: str,
    model_provider: str,
    output_tokens: int = GEN_AI_NUM_RESERVED_OUTPUT_TOKENS,
) -> int:
    litellm_model_map = get_model_map()
    input_toks = (
        llm_max_input_tokens(
            model_name=model_name,
            model_provider=model_provider,
            model_map=litellm_model_map,
        )
        - output_tokens
    )
    return max(input_toks, GEN_AI_MODEL_FALLBACK_MAX_TOKENS)


def get_bedrock_token_limit(model_id: str) -> int:
    """Determine token limit for a Bedrock model from ID suffix, LiteLLM, or hardcoded map."""
    from .constants import BEDROCK_MODEL_TOKEN_LIMITS

    model_id_lower = model_id.lower()

    context_match = re.search(r":(\d+)k\b", model_id_lower)
    if context_match:
        return int(context_match.group(1)) * 1000

    try:
        model_map = get_model_map()
        for key in [f"bedrock/{model_id}", model_id]:
            if key in model_map:
                info = model_map[key]
                limit = info.get("max_input_tokens") or info.get("max_tokens")
                if limit is not None:
                    return limit
    except Exception:
        pass

    for pattern, limit in sorted(
        BEDROCK_MODEL_TOKEN_LIMITS.items(), key=lambda x: -len(x[0])
    ):
        if pattern in model_id_lower:
            return limit

    return GEN_AI_MODEL_FALLBACK_MAX_TOKENS


def litellm_thinks_model_supports_image_input(
    model_name: str, model_provider: str
) -> bool:
    try:
        model_obj = find_model_obj(get_model_map(), model_provider, model_name)
        if not model_obj:
            return False
        return model_obj.get("supports_vision", False) or False
    except Exception:
        logger.exception(
            "Failed to get model object for %s/%s", model_provider, model_name
        )
        return False


def model_supports_image_input(model_name: str, model_provider: str) -> bool:
    return litellm_thinks_model_supports_image_input(model_name, model_provider)


def model_is_reasoning_model(model_name: str, model_provider: str) -> bool:
    import litellm

    model_map = get_model_map()
    try:
        model_obj = find_model_obj(model_map, model_provider, model_name)
        if model_obj and "supports_reasoning" in model_obj:
            reasoning = model_obj["supports_reasoning"]
            if reasoning is not None:
                return reasoning

        full_model_name = (
            f"{model_provider}/{model_name}"
            if model_provider not in model_name
            else model_name
        )
        return litellm.supports_reasoning(model=full_model_name)
    except Exception:
        logger.exception(
            "Failed to check reasoning support for %s/%s", model_provider, model_name
        )
        return False


def is_true_openai_model(model_provider: str, model_name: str) -> bool:
    """Return True only for native OpenAI models (not OpenAI-compatible proxies)."""
    if model_provider not in {
        str(LlmProviderNames.OPENAI),
        str(LlmProviderNames.LITELLM_PROXY),
        str(LlmProviderNames.AZURE),
    }:
        return False

    model_map = get_model_map()

    def _is_openai_provider(name: str) -> bool:
        entry = model_map.get(name)
        return bool(
            entry and entry.get("litellm_provider") == str(LlmProviderNames.OPENAI)
        )

    try:
        if f"{LlmProviderNames.OPENAI}/{model_name}" in model_map:
            return True
        if _is_openai_provider(model_name):
            return True
        if model_name.startswith(f"{LlmProviderNames.AZURE}/"):
            stripped = "/".join(model_name.split("/")[1:])
            if _is_openai_provider(stripped):
                return True
        return False
    except Exception:
        logger.exception(
            "Failed to determine if %s/%s is a true OpenAI model",
            model_provider,
            model_name,
        )
        return False


def model_needs_formatting_reenabled(model_name: str) -> bool:
    """True for OpenAI reasoning models that need markdown formatting re-enabled."""
    model_names = ["gpt-5.1", "gpt-5", "o3", "o1"]
    pattern = (
        r"(?:^|[\s\-/])("
        + "|".join(re.escape(name) for name in model_names)
        + r")(?:$|[\s\-/])"
    )
    return bool(re.search(pattern, model_name))
