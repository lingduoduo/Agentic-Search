"""Unit tests for the weather / geocoding / POI tools. No live network."""

from __future__ import annotations

import asyncio
import json

from src.internal.tools.public_data import geo
from src.internal.tools.public_data._http import PublicDataError


def _run(tool, **arguments):
    response, _raw, _meta = asyncio.run(tool.execute("default", arguments))
    return json.loads(response)


_FORECAST = {
    "current": {
        "time": "2026-08-06T12:00",
        "temperature_2m": 14.2,
        "apparent_temperature": 13.0,
        "relative_humidity_2m": 71,
        "precipitation": 0.0,
        "weather_code": 3,
        "wind_speed_10m": 11.5,
        "wind_direction_10m": 240,
    }
}


def test_weather_geocodes_then_fetches(monkeypatch):
    urls = []

    async def _fake_get_json(url, *, params=None, **kwargs):
        urls.append(url)
        if "geocoding" in url:
            return {
                "results": [
                    {
                        "latitude": 51.5,
                        "longitude": -0.13,
                        "name": "London",
                        "country": "United Kingdom",
                    }
                ]
            }
        return _FORECAST

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    result = _run(geo.build_weather_tool(), location="london")

    assert len(urls) == 2
    assert result["location"] == "London"
    assert result["country"] == "United Kingdom"
    assert result["temperature"] == 14.2
    assert result["description"] == "Overcast"


def test_weather_skips_geocoding_when_coordinates_given(monkeypatch):
    urls = []

    async def _fake_get_json(url, *, params=None, **kwargs):
        urls.append(url)
        return _FORECAST

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    result = _run(
        geo.build_weather_tool(), location="Somewhere", latitude=1.0, longitude=2.0
    )

    assert len(urls) == 1
    assert result["latitude"] == 1.0


def test_weather_unknown_location_is_an_error(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        return {"results": []}

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    assert "error" in _run(geo.build_weather_tool(), location="zzzzz")


def test_weather_unknown_code_falls_back(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        if "geocoding" in url:
            return {"results": [{"latitude": 0.0, "longitude": 0.0, "name": "X"}]}
        return {"current": dict(_FORECAST["current"], weather_code=999)}

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    assert _run(geo.build_weather_tool(), location="X")["description"] == "Unknown"


def test_search_location_flattens_nominatim(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, headers=None, **kwargs):
        captured["params"] = params
        return [
            {
                "display_name": "Eiffel Tower, Paris, France",
                "lat": "48.858",
                "lon": "2.294",
                "type": "attraction",
                "class": "tourism",
                "address": {"country": "France", "city": "Paris"},
            }
        ]

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    result = _run(geo.build_location_tool(), query="eiffel tower", country_code="FR")

    assert result["count"] == 1
    place = result["locations"][0]
    assert place["latitude"] == 48.858
    assert place["city"] == "Paris"
    assert captured["params"]["countrycodes"] == "fr"


def test_search_location_no_results(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return []

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    assert _run(geo.build_location_tool(), query="zzz")["count"] == 0


def test_nearby_places_parses_overpass_elements(monkeypatch):
    captured = {}

    async def _fake_post_json(url, *, data=None, **kwargs):
        captured["data"] = data
        return {
            "elements": [
                {
                    "id": 1,
                    "lat": 48.86,
                    "lon": 2.29,
                    "tags": {
                        "name": "Cafe Central",
                        "amenity": "cafe",
                        "addr:street": "Rue X",
                        "opening_hours": "08:00-18:00",
                    },
                }
            ]
        }

    monkeypatch.setattr(geo, "post_json", _fake_post_json)

    result = _run(
        geo.build_nearby_places_tool(),
        query="cafe",
        latitude=48.86,
        longitude=2.29,
    )

    assert result["count"] == 1
    assert result["places"][0]["name"] == "Cafe Central"
    assert result["places"][0]["type"] == "cafe"
    # Anchored so "cafe" does not also match "cafeteria_supplier".
    assert '~"^cafe$"' in captured["data"]["data"]


def test_nearby_places_rejects_overpass_injection(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not build a query from unvalidated input")

    monkeypatch.setattr(geo, "post_json", _unreachable)

    result = _run(
        geo.build_nearby_places_tool(),
        query='cafe"](around:1,0,0);out;//',
        latitude=0.0,
        longitude=0.0,
    )
    assert "error" in result


def test_nearby_places_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("overpass is busy")

    monkeypatch.setattr(geo, "post_json", _fail)

    result = _run(
        geo.build_nearby_places_tool(), query="cafe", latitude=0.0, longitude=0.0
    )
    assert result == {"error": "overpass is busy"}


def test_geo_tools_are_not_citeable():
    for factory in (
        geo.build_weather_tool,
        geo.build_location_tool,
        geo.build_nearby_places_tool,
    ):
        assert factory().citeable is False
