from src.internal.routing.construction.api import ApiRequestConstructor, ApiSpec
from src.internal.routing.route import RetrieverTarget, RouteDecision

_SPEC = ApiSpec("prices", "https://api.example.com/prices", ("symbol", "currency"))


def _route():
    return RouteDecision(
        domain="live",
        sources=["external_api"],
        retriever=RetrieverTarget.API,
        construction_target=RetrieverTarget.API,
    )


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages, **kwargs):
        return self._reply


def test_constructor_extracts_allowlisted_params():
    llm = _StubLLM('{"symbol": "A100", "currency": "USD", "evil": "drop"}')
    out = ApiRequestConstructor(llm, _SPEC).construct("price of A100 in USD", _route())
    assert out.target is RetrieverTarget.API
    assert out.payload["endpoint"] == "https://api.example.com/prices"
    assert out.payload["params"] == {
        "symbol": "A100",
        "currency": "USD",
    }  # 'evil' dropped


def test_constructor_degrades_on_bad_json():
    out = ApiRequestConstructor(_StubLLM("not json"), _SPEC).construct("x", _route())
    assert out.payload["params"] == {}
    assert out.payload["endpoint"] == "https://api.example.com/prices"
