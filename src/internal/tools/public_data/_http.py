"""Shared HTTP plumbing for the public data-source tools.

One place owns the timeout, the User-Agent, and the error shape, so each theme
module is only about its upstream's response format.

Absolute import of the aiohttp shim: this module sits one package deeper than
``tools.search``, and a four-dot relative import is needlessly hard to read.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

from src.context.retrieval.client import aiohttp

logger = logging.getLogger(__name__)

# Nominatim's usage policy requires an identifying User-Agent, and a generic
# browser string risks getting the whole project blocked. Individual callers
# may still override it (Yahoo rejects non-browser agents).
USER_AGENT = "AgenticSearch/1.0 (+https://github.com/linghypshen/Agentic-Search)"

DEFAULT_TIMEOUT_SECONDS = 10.0

# Upper bound on any single document body handed back to the model. Abstracts
# and article intros are otherwise long enough to crowd out the rollout budget.
MAX_CONTENT_CHARS = 1500


class PublicDataError(Exception):
    """An upstream call failed. ``guarded`` turns this into {"error": ...}."""


async def _fetch(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: Any = None,
    headers: dict | None = None,
    timeout_seconds: float,
    as_json: bool,
) -> Any:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method, url, params=params, data=data, headers=merged
            ) as response:
                if response.status >= 400:
                    raise PublicDataError(f"{url} returned HTTP {response.status}")
                body = await response.text()
    except PublicDataError:
        raise
    except Exception as exc:
        logger.debug("public data request to %s failed", url, exc_info=True)
        raise PublicDataError(f"request to {url} failed: {exc}") from exc

    if not as_json:
        return body
    # Read text then parse, rather than response.json(): several of these hosts
    # return JSON under a non-JSON content type, which aiohttp rejects.
    try:
        return json.loads(body)
    except ValueError as exc:
        raise PublicDataError(f"{url} returned a non-JSON body") from exc


async def get_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """GET *url* and parse the response as JSON. Raises PublicDataError."""
    return await _fetch(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=True,
    )


async def get_text(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """GET *url* and return the raw body. Raises PublicDataError."""
    return await _fetch(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=False,
    )


async def post_json(
    url: str,
    *,
    data: Any,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """POST *data* to *url* and parse the response as JSON."""
    return await _fetch(
        "POST",
        url,
        data=data,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=True,
    )


def guarded(fn: Callable) -> Callable:
    """Adapt a tool coroutine to the tool return contract.

    The wrapped function returns a plain ``dict``/``list``; this serializes it
    and converts any failure into ``{"error": ...}`` so one dead upstream
    degrades a single tool rather than the whole turn.
    """

    @functools.wraps(fn)
    async def _wrapped(**kwargs: Any) -> str:
        try:
            return json.dumps(await fn(**kwargs))
        except PublicDataError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - a tool must never raise
            logger.debug("tool %s failed", getattr(fn, "__name__", "?"), exc_info=True)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return _wrapped
