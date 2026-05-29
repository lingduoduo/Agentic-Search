"""Tests for the billing service layer (src/servers/billing/service.py).

The service is a thin HTTP proxy — tests mock httpx to avoid real requests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from .conftest import make_mock_http_client
from .conftest import make_mock_response


class TestMakeBillingRequest:
    """Tests for _make_billing_request — the central HTTP helper."""

    @pytest.mark.asyncio
    async def test_makes_post_request_and_returns_json(self) -> None:
        from src.servers.billing.service import _make_billing_request

        mock_response = make_mock_response({"url": "https://checkout.stripe.com/s"})
        mock_client = make_mock_http_client("post", response=mock_response)

        with patch("httpx.AsyncClient", mock_client):
            result = await _make_billing_request(
                method="POST",
                path="/create-checkout-session",
                base_url="https://api.example.com/proxy",
                body={"billing_period": "monthly"},
            )

        assert result == {"url": "https://checkout.stripe.com/s"}

    @pytest.mark.asyncio
    async def test_makes_get_request_with_params(self) -> None:
        from src.servers.billing.service import _make_billing_request

        mock_response = make_mock_response({"subscribed": False})
        mock_client = make_mock_http_client("get", response=mock_response)

        with patch("httpx.AsyncClient", mock_client):
            result = await _make_billing_request(
                method="GET",
                path="/billing-information",
                base_url="https://api.example.com/proxy",
                params={"tenant_id": "123"},
            )

        assert result == {"subscribed": False}

    @pytest.mark.asyncio
    async def test_uses_license_bearer_auth(self) -> None:
        from src.servers.billing.service import _make_billing_request

        mock_response = make_mock_response({})
        mock_client = make_mock_http_client("post", response=mock_response)

        with patch("httpx.AsyncClient", mock_client):
            await _make_billing_request(
                method="POST",
                path="/test",
                base_url="https://api.example.com/proxy",
                license_data="my_license_token",
                body={},
            )

        post_call = mock_client.return_value.__aenter__.return_value.post
        _, kwargs = post_call.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer my_license_token"

    @pytest.mark.asyncio
    async def test_follows_redirects(self) -> None:
        from src.servers.billing.service import _make_billing_request

        mock_response = make_mock_response({})
        mock_client = make_mock_http_client("get", response=mock_response)

        with patch("httpx.AsyncClient", mock_client):
            await _make_billing_request(
                method="GET",
                path="/test",
                base_url="http://api.example.com/proxy",
            )

        mock_client.assert_called_once_with(timeout=30.0, follow_redirects=True)

    @pytest.mark.asyncio
    async def test_raises_http_exception_on_http_status_error(self) -> None:
        from fastapi import HTTPException

        from src.servers.billing.service import _make_billing_request

        mock_response = make_mock_response({"detail": "Bad gateway"})
        mock_response.status_code = 502
        error = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )
        mock_client = make_mock_http_client("post", side_effect=error)

        with patch("httpx.AsyncClient", mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await _make_billing_request(
                    method="POST",
                    path="/test",
                    base_url="https://api.example.com/proxy",
                    error_message="Request failed",
                )

        assert exc_info.value.status_code == 502
        assert "Bad gateway" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_http_exception_on_connection_error(self) -> None:
        from fastapi import HTTPException

        from src.servers.billing.service import _make_billing_request

        mock_client = make_mock_http_client(
            "post", side_effect=httpx.RequestError("connection refused")
        )

        with patch("httpx.AsyncClient", mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await _make_billing_request(
                    method="POST",
                    path="/test",
                    base_url="https://api.example.com/proxy",
                )

        assert exc_info.value.status_code == 502
        assert "billing service" in exc_info.value.detail.lower()


class TestCreateCheckoutSession:
    @pytest.mark.asyncio
    async def test_builds_body_and_returns_url(self) -> None:
        from src.servers.billing.service import create_checkout_session

        with patch(
            "src.servers.billing.service._make_billing_request",
            new_callable=AsyncMock,
            return_value={"url": "https://checkout.stripe.com/session"},
        ) as mock_req:
            result = await create_checkout_session(
                base_url="https://api.example.com/proxy",
                billing_period="annual",
                email="test@example.com",
                license_data="blob",
                redirect_url="https://app.example.com/success",
            )

        assert result.stripe_checkout_url == "https://checkout.stripe.com/session"
        body = mock_req.call_args[1]["body"]
        assert body["billing_period"] == "annual"
        assert body["email"] == "test@example.com"
        assert body["redirect_url"] == "https://app.example.com/success"

    @pytest.mark.asyncio
    async def test_omits_email_when_none(self) -> None:
        from src.servers.billing.service import create_checkout_session

        with patch(
            "src.servers.billing.service._make_billing_request",
            new_callable=AsyncMock,
            return_value={"url": "https://checkout.stripe.com/session"},
        ) as mock_req:
            await create_checkout_session(
                base_url="https://api.example.com/proxy",
                billing_period="monthly",
            )

        body = mock_req.call_args[1]["body"]
        assert "email" not in body


class TestCreateCustomerPortalSession:
    @pytest.mark.asyncio
    async def test_returns_portal_url(self) -> None:
        from src.servers.billing.service import create_customer_portal_session

        with patch(
            "src.servers.billing.service._make_billing_request",
            new_callable=AsyncMock,
            return_value={"url": "https://billing.stripe.com/portal"},
        ):
            result = await create_customer_portal_session(
                base_url="https://api.example.com/proxy",
                license_data="blob",
                return_url="https://app.example.com/billing",
            )

        assert result.stripe_customer_portal_url == "https://billing.stripe.com/portal"


class TestGetBillingInformation:
    @pytest.mark.asyncio
    async def test_returns_billing_info_when_subscribed(self) -> None:
        from src.servers.billing.models import BillingInformationResponse
        from src.servers.billing.service import get_billing_information

        with patch(
            "src.servers.billing.service._make_billing_request",
            new_callable=AsyncMock,
            return_value={"tenant_id": "t1", "status": "active", "seats": 10},
        ):
            result = await get_billing_information(
                base_url="https://api.example.com/proxy",
                license_data="blob",
            )

        assert isinstance(result, BillingInformationResponse)
        assert result.tenant_id == "t1"
        assert result.status == "active"

    @pytest.mark.asyncio
    async def test_returns_not_subscribed_when_subscribed_false(self) -> None:
        from src.servers.billing.models import SubscriptionStatusResponse
        from src.servers.billing.service import get_billing_information

        with patch(
            "src.servers.billing.service._make_billing_request",
            new_callable=AsyncMock,
            return_value={"subscribed": False},
        ):
            result = await get_billing_information(
                base_url="https://api.example.com/proxy",
            )

        assert isinstance(result, SubscriptionStatusResponse)
        assert result.subscribed is False


class TestUpdateSeatCount:
    @pytest.mark.asyncio
    async def test_updates_seats_and_returns_response(self) -> None:
        from src.servers.billing.service import update_seat_count

        with patch(
            "src.servers.billing.service._make_billing_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "current_seats": 15,
                "used_seats": 5,
                "message": "ok",
            },
        ) as mock_req:
            result = await update_seat_count(
                base_url="https://api.example.com/proxy",
                new_seat_count=15,
                license_data="blob",
            )

        assert result.success is True
        assert result.current_seats == 15
        assert mock_req.call_args[1]["body"]["new_seat_count"] == 15


class TestEndTrial:
    @pytest.mark.asyncio
    async def test_raises_501_for_self_hosted(self) -> None:
        from fastapi import HTTPException

        from src.servers.billing.service import end_trial

        with pytest.raises(HTTPException) as exc_info:
            await end_trial()

        assert exc_info.value.status_code == 501
