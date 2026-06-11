"""HTTP hook execution for Agentic Search.

Hooks let deployments call an external service at a well-defined point, such as
before query execution. The implementation is dependency-light and explicit:
callers provide a hook config or registry, choose soft/hard failure behavior,
and receive a typed result.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


from pydantic import TypeAdapter
from pydantic import ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HookPoint(StrEnum):
    QUERY_PROCESSING = "query_processing"
    BEFORE_RETRIEVAL = "before_retrieval"
    AFTER_RETRIEVAL = "after_retrieval"
    ANSWER_GENERATED = "answer_generated"


class HookFailStrategy(StrEnum):
    SOFT = "soft"
    HARD = "hard"


@dataclass(frozen=True)
class HookConfig:
    """Configuration for one external hook endpoint."""

    hook_point: HookPoint | str
    endpoint_url: str
    api_key: str | None = None
    timeout_seconds: float = 5.0
    fail_strategy: HookFailStrategy | str = HookFailStrategy.SOFT
    is_active: bool = True

    def normalized_hook_point(self) -> HookPoint:
        return HookPoint(self.hook_point)

    def normalized_fail_strategy(self) -> HookFailStrategy:
        return HookFailStrategy(self.fail_strategy)


@dataclass(frozen=True)
class HookExecutionRecord:
    hook_point: HookPoint
    endpoint_url: str
    is_success: bool
    duration_ms: int
    status_code: int | None = None
    error_message: str | None = None
    is_reachable: bool | None = None


@dataclass(frozen=True)
class HookSkipped:
    reason: str = "no active hook configured"


@dataclass(frozen=True)
class HookSoftFailed:
    error_message: str
    status_code: int | None = None
    is_reachable: bool | None = None


class HookExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        is_reachable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.is_reachable = is_reachable


@dataclass
class HookRegistry:
    """In-memory hook lookup and execution log."""

    hooks: dict[HookPoint, HookConfig]
    execution_log: list[HookExecutionRecord]

    def __init__(self, hooks: list[HookConfig] | None = None) -> None:
        self.hooks = {}
        self.execution_log = []
        for hook in hooks or []:
            self.register(hook)

    def register(self, hook: HookConfig) -> None:
        self.hooks[hook.normalized_hook_point()] = hook

    def get(self, hook_point: HookPoint | str) -> HookConfig | None:
        hook = self.hooks.get(HookPoint(hook_point))
        if hook is None or not hook.is_active or not hook.endpoint_url:
            return None
        return hook

    def record(self, record: HookExecutionRecord) -> None:
        self.execution_log.append(record)


@dataclass(frozen=True)
class _HttpOutcome:
    is_success: bool
    status_code: int | None
    error_message: str | None
    response_payload: dict[str, Any] | None
    is_reachable: bool | None


def execute_hook(
    *,
    hook_point: HookPoint | str,
    payload: dict[str, Any],
    response_type: type[T] | None = None,
    hook: HookConfig | None = None,
    registry: HookRegistry | None = None,
) -> T | dict[str, Any] | HookSkipped | HookSoftFailed:
    """Execute a hook and validate the JSON response.

    ``HookSkipped`` means no active hook exists. ``HookSoftFailed`` means the
    hook failed but its fail strategy is soft. Hard failures raise
    ``HookExecutionError``.
    """

    selected_hook = hook or (registry.get(hook_point) if registry else None)
    if selected_hook is None:
        return HookSkipped()

    start = time.monotonic()
    outcome = _post_json_hook(selected_hook, payload)
    duration_ms = int((time.monotonic() - start) * 1000)
    record = HookExecutionRecord(
        hook_point=selected_hook.normalized_hook_point(),
        endpoint_url=selected_hook.endpoint_url,
        is_success=outcome.is_success,
        duration_ms=duration_ms,
        status_code=outcome.status_code,
        error_message=outcome.error_message,
        is_reachable=outcome.is_reachable,
    )
    if registry is not None:
        registry.record(record)

    if not outcome.is_success:
        return _handle_failure(selected_hook, outcome)

    if outcome.response_payload is None:
        raise RuntimeError(
            "Hook reported success but returned no payload — this is a bug."
        )
    try:
        return _validate_response(outcome.response_payload, response_type)
    except (TypeError, ValidationError, ValueError) as exc:
        validation_outcome = _HttpOutcome(
            is_success=False,
            status_code=outcome.status_code,
            error_message=f"Hook response validation failed: {exc}",
            response_payload=None,
            is_reachable=None,
        )
        return _handle_failure(selected_hook, validation_outcome)


def _post_json_hook(hook: HookConfig, payload: dict[str, Any]) -> _HttpOutcome:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if hook.api_key:
        headers["Authorization"] = f"Bearer {hook.api_key}"
    request = urllib.request.Request(
        hook.endpoint_url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=hook.timeout_seconds) as response:
            status_code = int(response.status)
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status_code = int(exc.code)
        auth_failed = status_code in (401, 403)
        return _HttpOutcome(
            is_success=False,
            status_code=status_code,
            error_message=f"Hook returned HTTP {status_code}: {body}",
            response_payload=None,
            is_reachable=False if auth_failed else None,
        )
    except urllib.error.URLError as exc:
        return _HttpOutcome(
            is_success=False,
            status_code=None,
            error_message=f"Hook endpoint unreachable: {exc.reason}",
            response_payload=None,
            is_reachable=False,
        )
    except TimeoutError as exc:
        return _HttpOutcome(
            is_success=False,
            status_code=None,
            error_message=f"Hook timed out after {hook.timeout_seconds}s: {exc}",
            response_payload=None,
            is_reachable=None,
        )
    except Exception as exc:
        logger.exception("Unexpected hook execution error")
        return _HttpOutcome(
            is_success=False,
            status_code=None,
            error_message=f"Hook call failed: {exc}",
            response_payload=None,
            is_reachable=None,
        )

    try:
        parsed = json.loads(raw_body or "{}")
    except json.JSONDecodeError as exc:
        return _HttpOutcome(
            is_success=False,
            status_code=status_code,
            error_message=f"Hook returned non-JSON response: {exc}",
            response_payload=None,
            is_reachable=None,
        )
    if not isinstance(parsed, dict):
        return _HttpOutcome(
            is_success=False,
            status_code=status_code,
            error_message=f"Hook returned non-object JSON: {type(parsed).__name__}",
            response_payload=None,
            is_reachable=None,
        )
    return _HttpOutcome(
        is_success=True,
        status_code=status_code,
        error_message=None,
        response_payload=parsed,
        is_reachable=True,
    )


@lru_cache(maxsize=64)
def _get_adapter(response_type: type[T]) -> TypeAdapter[T]:
    return TypeAdapter(response_type)


def _validate_response(
    payload: dict[str, Any],
    response_type: type[T] | None,
) -> T | dict[str, Any]:
    if response_type is None:
        return payload
    return _get_adapter(response_type).validate_python(payload)


def _handle_failure(
    hook: HookConfig,
    outcome: _HttpOutcome,
) -> HookSoftFailed:
    message = outcome.error_message or "Hook execution failed."
    if hook.normalized_fail_strategy() == HookFailStrategy.HARD:
        raise HookExecutionError(
            message,
            status_code=outcome.status_code,
            is_reachable=outcome.is_reachable,
        )
    logger.warning("Hook execution soft-failed: %s", message)
    return HookSoftFailed(
        error_message=message,
        status_code=outcome.status_code,
        is_reachable=outcome.is_reachable,
    )


__all__ = [
    "HookConfig",
    "HookExecutionError",
    "HookExecutionRecord",
    "HookFailStrategy",
    "HookPoint",
    "HookRegistry",
    "HookSkipped",
    "HookSoftFailed",
    "execute_hook",
]
