from src.context import (
    LLMResponse,
    SchemaUnsupportedError,
    StructuredCompletionMetadata,
    StructuredOutputCapability,
    StructuredOutputRequest,
    answer_draft_json_schema,
)


def test_answer_draft_schema_is_strict_and_copy_safe():
    first = answer_draft_json_schema()

    assert first["additionalProperties"] is False
    assert first["required"] == ["claims", "missing_information", "abstain"]
    assert first["properties"]["claims"]["items"]["additionalProperties"] is False

    first["properties"]["abstain"]["type"] = "string"
    assert answer_draft_json_schema()["properties"]["abstain"]["type"] == "boolean"


def test_structured_metadata_defaults_preserve_legacy_clients():
    response = LLMResponse(
        text='{"claims": [], "missing_information": [], "abstain": true}'
    )

    assert response.structured.applied is False
    assert response.structured.refused is False
    assert response.structured.incomplete_reason is None


def test_structured_output_contract_defaults():
    request = StructuredOutputRequest(name="answer_draft", schema={"type": "object"})

    assert request.strict is True
    assert StructuredOutputCapability.JSON_SCHEMA.value == "json_schema"
    assert StructuredOutputCapability.PROMPT_ONLY.value == "prompt_only"
    assert StructuredCompletionMetadata().requested is False
    assert issubclass(SchemaUnsupportedError, RuntimeError)
