# Provider-Enforced Structured RAG Output Design

## Goal

Strengthen grounded RAG generation by using provider-enforced JSON Schema when the configured provider and model support it, while retaining the existing prompt, parser, semantic evidence verifier, one-retry limit, and canonical abstention behavior.

## Current State

The grounded RAG prompt asks the model for an exact `AnswerDraft` JSON object. `parse_answer_draft` validates its keys and types, and `verify_answer_draft` validates cited evidence and claim support. This is safe after generation but still relies on the model following formatting instructions.

The general `LLM` interface and `LitellmLLM` accept `structured_response_format`, but the RAG-facing `LLMClient.complete` contract has no typed structured-output semantics. `OpenAICompatibleLLM.complete` ignores structured-output keyword arguments, returns only text, and discards refusal and incomplete-response state.

## Chosen Approach

Use capability-aware hybrid enforcement.

1. Define one canonical JSON Schema for the internal `AnswerDraft` shape.
2. Ask known-compatible providers for provider-enforced JSON Schema output.
3. Keep the structured-output prompt in every guarded request as semantic guidance.
4. Fall back to prompt-plus-local-validation only when the provider explicitly reports that JSON Schema is unsupported.
5. Do not downgrade on authentication, rate-limit, timeout, transport, server, or other provider failures.
6. Run the existing local parser and evidence verifier for every successful model response, including provider-constrained responses.

This change does not replace the provider stack with LiteLLM and does not assume every OpenAI-compatible endpoint implements OpenAI Structured Outputs.

## Contracts

### Canonical schema

Add an immutable canonical schema for this object:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["claims", "missing_information", "abstain"],
  "properties": {
    "claims": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["text", "evidence_ids"],
        "properties": {
          "text": {"type": "string", "minLength": 1},
          "evidence_ids": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"}
          }
        }
      }
    },
    "missing_information": {
      "type": "array",
      "items": {"type": "string"}
    },
    "abstain": {"type": "boolean"}
  }
}
```

Evidence IDs cannot be fully constrained by the static schema because valid IDs differ by request. `parse_answer_draft` remains responsible for rejecting unknown IDs.

### Capability

Use an explicit capability enum:

- `JSON_SCHEMA`: the adapter can request schema-constrained output.
- `PROMPT_ONLY`: the endpoint is known not to support schema enforcement.

Capability is resolved by the concrete adapter, not inferred inside the RAG pipeline from model-name substrings. Native OpenAI chat-completions configuration may use `JSON_SCHEMA`. Generic OpenAI-compatible endpoints default to `PROMPT_ONLY` unless explicitly configured as schema-capable.

### Completion request and result

Extend the RAG-facing completion boundary with optional structured-output configuration and a result that can retain:

- response text;
- whether provider schema enforcement was applied;
- refusal state;
- incomplete/truncated state and reason;
- a typed schema-unsupported outcome.

Existing simple fake clients that return `str` or `LLMResponse` remain compatible. New fields must be optional and defaulted.

## Data Flow

For each guarded generation attempt:

1. Build the existing structured prompt with evidence and ordered chat history.
2. If capability is `JSON_SCHEMA`, call `complete` with the canonical schema.
3. If the provider explicitly rejects schema support, retry that same attempt once without provider schema enforcement. This is a transport-level downgrade, not the corrective semantic retry.
4. If capability is `PROMPT_ONLY`, call without a schema.
5. Convert refusal or incomplete output into verifier feedback or abstention according to the existing bounded generation policy.
6. Parse the JSON locally.
7. Verify every claim against cited evidence.
8. Use at most one corrective semantic retry.
9. Render only supported claims or exactly `I don't know based on the available evidence.`

The schema downgrade must not increase the maximum number of semantic correction attempts. A single semantic attempt may contain one schema-capable request followed by one prompt-only retry only when the first response is explicitly schema-unsupported.

## Provider Adapter Behavior

`OpenAICompatibleLLM.complete` will:

- add Chat Completions `response_format.type = "json_schema"` with `strict: true` for schema-capable native OpenAI configuration;
- set `additionalProperties: false` through the canonical schema;
- preserve response message refusal and finish reason before returning;
- classify only provider responses that specifically identify unsupported `response_format` or `json_schema` as schema-unsupported;
- continue raising unrelated HTTP and transport errors;
- avoid logging the schema payload, evidence, prompt, or raw refusal text in new public traces.

Generic OpenAI-compatible proxies remain prompt-only by default because API-shape compatibility does not prove constrained-decoding support.

## Failure Handling

- **Schema unsupported:** downgrade once for that attempt and record the downgrade internally.
- **Malformed JSON in prompt-only mode:** use the existing corrective semantic retry.
- **Schema-constrained but locally invalid output:** treat it as malformed and retain local validation; provider guarantees do not replace defense in depth.
- **Unknown evidence IDs or unsupported claims:** use existing verifier feedback and the one semantic retry.
- **Refusal:** do not expose raw refusal content as an answer; produce canonical abstention with abstained verification metadata.
- **Token/content truncation:** treat as incomplete, allowing the one semantic retry; abstain if no valid supported draft results.
- **Authentication, rate limit, timeout, transport, or server error:** propagate normally; never silently switch enforcement modes.

## Observability

Add only non-sensitive aggregate metadata:

- requested structured-output mode;
- applied mode;
- schema downgrade boolean/count;
- refusal boolean;
- incomplete reason category;
- semantic retry count.

Do not record schemas, prompts, evidence bodies, model output, refusal text, tool arguments, or exception bodies in new traces.

## Compatibility

- Existing `LLMClient` implementations returning strings remain valid and behave as prompt-only.
- Existing public RAG and MCP response keys remain unchanged.
- Existing confidence, verification status, citation, abstention, tool evidence, and retry semantics remain unchanged.
- `GroundedGenerationConfig(enabled=False)` continues to use the legacy free-text path without requesting structured output.
- No external fact checker or new provider dependency is introduced.

## Testing

Tests will cover:

- canonical schema exactness and immutability/copy safety;
- native schema-capable request body;
- generic compatible endpoint defaulting to prompt-only;
- schema success followed by local semantic verification;
- explicit schema-unsupported downgrade;
- no downgrade for unrelated 4xx/5xx, timeout, or transport failures;
- refusal and incomplete response handling;
- malformed prompt-only output and the one semantic corrective retry;
- total semantic retries remaining capped at one despite schema downgrade;
- unchanged legacy mode and simple fake-client compatibility;
- absence of prompt, schema, evidence, output, and refusal text in new tracing metadata.

## Non-Goals

- Replacing `OpenAICompatibleLLM` with `LitellmLLM` for RAG.
- Guaranteeing factual correctness from JSON Schema alone.
- Supporting provider-native Anthropic or Gemini request shapes in this first adapter change.
- Dynamically probing arbitrary endpoints for schema support.
- Removing prompt formatting instructions or local validation.
