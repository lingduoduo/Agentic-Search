"""AgenticSearchError — structured exception for all backend business errors.

Raise ``AgenticSearchError`` instead of ``HTTPException`` in business code.
Register ``register_exception_handlers`` on the FastAPI app to convert it into
a JSON response with the standard ``{"error_code": "...", "detail": "..."}``
shape.

Usage::

    from src.backend.servers.error_handling.error_codes import ErrorCode
    from src.backend.servers.error_handling.exceptions import AgenticSearchError

    raise AgenticSearchError(ErrorCode.NOT_FOUND, "Session not found")
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from src.backend.servers.error_handling.error_codes import ErrorCode

logger = logging.getLogger(__name__)


class AgenticSearchError(Exception):
    def __init__(
        self,
        error_code: ErrorCode,
        detail: str | None = None,
        *,
        status_code_override: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        resolved_detail = detail or error_code.code
        super().__init__(resolved_detail)
        self.error_code = error_code
        self.detail = resolved_detail
        self._status_code_override = status_code_override
        self.headers = headers

    @property
    def status_code(self) -> int:
        return self._status_code_override or self.error_code.status_code


def _log_error(exc: AgenticSearchError) -> None:
    if exc.status_code >= 500:
        logger.error("AgenticSearchError %s: %s", exc.error_code.code, exc.detail)
    elif exc.status_code >= 400:
        logger.warning("AgenticSearchError %s: %s", exc.error_code.code, exc.detail)


def error_to_json_response(exc: AgenticSearchError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.error_code.detail(exc.detail),
        headers=exc.headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register a global handler that converts ``AgenticSearchError`` to JSON.

    Call after the app is created, before it starts serving.
    """

    @app.exception_handler(AgenticSearchError)
    async def _handle_error(
        request: Request,  # noqa: ARG001
        exc: AgenticSearchError,
    ) -> JSONResponse:
        _log_error(exc)
        return error_to_json_response(exc)
