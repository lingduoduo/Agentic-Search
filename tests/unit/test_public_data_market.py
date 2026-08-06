"""Unit tests for the stock / crypto / currency tools. No live network."""

from __future__ import annotations

import asyncio
import json

from src.internal.tools.public_data import market
from src.internal.tools.public_data._http import PublicDataError


def _run(tool, **arguments):
    response, _raw, _meta = asyncio.run(tool.execute("default", arguments))
    return json.loads(response)


def test_stock_quote_maps_yahoo_meta(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, headers=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        return {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "currency": "USD",
                            "regularMarketPrice": 210.5,
                            "previousClose": 208.0,
                            "regularMarketDayHigh": 211.0,
                            "regularMarketDayLow": 207.5,
                            "regularMarketVolume": 1234,
                            "exchangeName": "NMS",
                        }
                    }
                ]
            }
        }

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(market.build_stock_quote_tool(), symbol="aapl")

    assert result["symbol"] == "AAPL"
    assert result["current_price"] == 210.5
    assert result["exchange"] == "NMS"
    assert captured["url"].endswith("/aapl")
    # Yahoo 4xxs a request without a browser-like agent.
    assert "Mozilla" in captured["headers"]["User-Agent"]


def test_stock_quote_empty_result_is_an_error(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return {"chart": {"result": []}}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    assert "error" in _run(market.build_stock_quote_tool(), symbol="ZZZZ")


def test_stock_quote_rejects_path_traversal(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not call out with an unvalidated symbol")

    monkeypatch.setattr(market, "get_json", _unreachable)

    assert "error" in _run(market.build_stock_quote_tool(), symbol="../../evil")


def test_crypto_price_maps_symbol_to_coingecko_id(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, **kwargs):
        captured.update(params)
        return {
            "bitcoin": {
                "usd": 64000.0,
                "usd_market_cap": 1.2e12,
                "usd_24h_vol": 3.0e10,
                "usd_24h_change": -1.5,
                "last_updated_at": 1700000000,
            }
        }

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(market.build_crypto_price_tool(), symbol="btc")

    assert captured["ids"] == "bitcoin"
    assert result["coin_id"] == "bitcoin"
    assert result["price"] == 64000.0
    assert result["change_24h_percent"] == -1.5
    assert result["last_updated"].startswith("20")


def test_crypto_price_unknown_coin_is_an_error(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return {}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    assert "error" in _run(market.build_crypto_price_tool(), symbol="notacoin")


def test_convert_currency_computes_amount(monkeypatch):
    async def _fake_get_json(url, **kwargs):
        assert url.endswith("/USD")
        return {"rates": {"EUR": 0.9}, "date": "2026-08-06"}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(
        market.build_currency_tool(),
        amount=10.0,
        from_currency="usd",
        to_currency="eur",
    )

    assert result["converted_amount"] == 9.0
    assert result["rate"] == 0.9
    assert result["from_currency"] == "USD"


def test_convert_currency_unknown_target_is_an_error(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return {"rates": {"EUR": 0.9}}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(
        market.build_currency_tool(),
        amount=1.0,
        from_currency="USD",
        to_currency="XYZ",
    )
    assert "error" in result


def test_convert_currency_rejects_non_iso_code(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not call out with an unvalidated code")

    monkeypatch.setattr(market, "get_json", _unreachable)

    result = _run(
        market.build_currency_tool(),
        amount=1.0,
        from_currency="../secrets",
        to_currency="EUR",
    )
    assert "error" in result


def test_currency_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("rates are down")

    monkeypatch.setattr(market, "get_json", _fail)

    result = _run(
        market.build_currency_tool(),
        amount=1.0,
        from_currency="USD",
        to_currency="EUR",
    )
    assert result == {"error": "rates are down"}


def test_market_tools_are_not_citeable():
    for factory in (
        market.build_stock_quote_tool,
        market.build_crypto_price_tool,
        market.build_currency_tool,
    ):
        assert factory().citeable is False
