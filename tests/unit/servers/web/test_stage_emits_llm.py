from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.servers.web import request_capture as rc


def test_complete_emits_llm_stage_when_capturing():
    llm = OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="gpt-4o-mini", api_key="sk")
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hi there"}}]}
    mock_resp.raise_for_status.return_value = None

    token = rc.start_capture("r", "q")
    try:
        with patch.object(llm._session, "post", return_value=mock_resp):
            llm.complete([{"role": "user", "content": "hello"}])
        cap = rc.active()
        llm_stages = [s for s in cap.stages if s.stage == "llm"]
        assert llm_stages, "expected an llm stage"
        assert llm_stages[0].payload["completion"] == "hi there"
        assert llm_stages[0].payload["model"] == "gpt-4o-mini"
        assert llm_stages[0].payload["messages"] == [
            {"role": "user", "content": "hello"}
        ]
    finally:
        rc.reset_capture(token)


def test_complete_no_capture_does_not_raise():
    llm = OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="gpt-4o-mini", api_key="sk")
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    mock_resp.raise_for_status.return_value = None
    with patch.object(llm._session, "post", return_value=mock_resp):
        assert llm.complete([{"role": "user", "content": "hi"}]) == "x"
