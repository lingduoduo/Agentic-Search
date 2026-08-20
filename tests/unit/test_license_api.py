"""The license claim endpoint must not block the event loop.

`claim_license` is an `async def` that talks to the cloud data plane over
blocking `requests`, with a 30s timeout. Awaited inline, one slow or hanging
license server freezes every other request the backend is serving — the same
defect fixed for answer synthesis in #547.

Thread identity is the assertion because it is the thing that actually differs:
an inline call runs on the event loop thread, an offloaded one cannot.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.internal.servers.license.api import create_license_router


def claim_endpoint(cloud_data_plane_url: str = "http://cloud.example"):
    """Pull `claim_license` back out of the router closure."""
    router = create_license_router(
        SimpleNamespace(cloud_data_plane_url=cloud_data_plane_url)
    )
    return next(
        route.endpoint for route in router.routes if route.name == "claim_license"
    )


def response_carrying_a_license() -> Mock:
    return Mock(
        raise_for_status=Mock(return_value=None),
        json=Mock(return_value={"license": "SIGNED-LICENSE"}),
    )


def verified_payload() -> Mock:
    return Mock(
        expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
        max_seats=10,
        tenant_id="tenant-1",
    )


@pytest.mark.asyncio
async def test_claiming_with_a_session_id_leaves_the_event_loop_thread_free() -> None:
    calling_thread: dict[str, int] = {}

    def record_thread(*_args, **_kwargs) -> Mock:
        calling_thread["id"] = threading.get_ident()
        return response_carrying_a_license()

    with (
        patch("src.internal.servers.license.api.requests.post", record_thread),
        patch(
            "src.internal.servers.license.api.verify_license_signature",
            return_value=verified_payload(),
        ),
        patch("src.internal.servers.license.api._save_license"),
    ):
        await claim_endpoint()(session_id="checkout-1", _=None)

    assert calling_thread["id"] != threading.get_ident()


@pytest.mark.asyncio
async def test_refreshing_a_stored_license_leaves_the_event_loop_thread_free() -> None:
    calling_thread: dict[str, int] = {}

    def record_thread(*_args, **_kwargs) -> Mock:
        calling_thread["id"] = threading.get_ident()
        return response_carrying_a_license()

    with (
        patch("src.internal.servers.license.api.requests.get", record_thread),
        patch(
            "src.internal.servers.license.api._load_license_data",
            return_value="STORED-LICENSE",
        ),
        patch(
            "src.internal.servers.license.api.verify_license_signature",
            return_value=verified_payload(),
        ),
        patch("src.internal.servers.license.api._save_license"),
    ):
        await claim_endpoint()(session_id=None, _=None)

    assert calling_thread["id"] != threading.get_ident()


@pytest.mark.asyncio
async def test_the_claimed_license_is_still_verified_and_stored() -> None:
    with (
        patch(
            "src.internal.servers.license.api.requests.post",
            return_value=response_carrying_a_license(),
        ),
        patch(
            "src.internal.servers.license.api.verify_license_signature",
            return_value=verified_payload(),
        ) as verify,
        patch("src.internal.servers.license.api._save_license") as save,
    ):
        result = await claim_endpoint()(session_id="checkout-1", _=None)

    verify.assert_called_once_with("SIGNED-LICENSE")
    assert save.call_args.args[0] == "SIGNED-LICENSE"
    assert result.success is True
    assert result.tenant_id == "tenant-1"


@pytest.mark.asyncio
async def test_an_unreachable_license_server_still_becomes_a_502() -> None:
    import requests as requests_module
    from fastapi import HTTPException

    with patch(
        "src.internal.servers.license.api.requests.post",
        side_effect=requests_module.ConnectionError("refused"),
    ):
        with pytest.raises(HTTPException) as raised:
            await claim_endpoint()(session_id="checkout-1", _=None)

    assert raised.value.status_code == 502
