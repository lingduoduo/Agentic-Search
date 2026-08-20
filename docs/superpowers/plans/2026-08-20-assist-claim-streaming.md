# Assist Claim-Level Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream the Assist surface's answer to the browser one verified claim at a time, so time-to-first-text stops equalling time-to-completion.

**Architecture:** Assist's grounded path has the model emit a JSON *answer draft* of claims; the answer the user sees is the join of only the claims that pass evidence verification. So the streamable unit is the verified claim, not the token. An incremental reader parses complete claim objects out of the partial JSON, each is verified as it arrives, and only supported claims are emitted. Everything is gated behind an optional `on_claim` callback — when it is `None`, behaviour is byte-identical to today.

**Tech Stack:** Python 3.11+, FastAPI, pytest, React 19 + TypeScript + Vitest.

**Spec:** `docs/superpowers/specs/2026-08-20-assist-claim-streaming-design.md`

## Global Constraints

- **Branch:** `feat/assist-claim-streaming`. Never commit to `main`.
- **The invariant — append-only.** Once a claim is emitted, nothing later in the run may contradict, retract, or rewrite it. The final answer must equal the join of the emitted claims, in emission order. Every design decision below serves this; if a change would break it, stop and raise it rather than working around it.
- **`on_claim=None` must be byte-identical to today.** The existing non-streaming callers (MCP `ask_agentic_search`, the search pipeline) and every existing test must be unaffected. Verify by running the full suite, not by inspection.
- **Never add a required method to the `LLMClient` Protocol** (`src/context/models.py:33`). It is structurally typed; a required streaming method breaks every implementation and test double. Streaming support is probed with `getattr`, matching how `structured_output_capability` is already probed at `src/context/pipeline.py:171`.
- **Streaming is advisory, never authoritative.** The final answer always comes from the whole-text `parse_answer_draft` + verification path. Any streaming failure degrades to today's behaviour.
- **Lint before every commit:** `ruff check . --fix && ruff format .`. Frontend: `cd web && npm run typecheck`.
- Pre-commit hooks run `ruff-format` and can abort a commit; re-stage and retry if that happens.

---

### Task 1: `stream_complete` adapter on the LLM provider

`OpenAICompatibleLLM.stream()` already exists but is not usable here as-is, for two reasons. Its signature is provider-shaped (`structured_response_format: dict`, yields `ModelResponseStream` objects) while `complete`'s is `LLMClient`-shaped (`structured_output: StructuredOutputRequest`, returns text) — probing `stream` directly would push provider details into `pipeline.py`. **And `structured_response_format` is a dead parameter: it is accepted at `providers.py:154` and never written into the request body.** Streaming with a schema would silently produce free prose instead of a JSON draft, which would break claim parsing in a way that looks like a model failure. Fixing that dead parameter is part of this task.

**Files:**
- Modify: `src/internal/llm/providers.py` (the `stream` method at 150-204; add `stream_complete` after it)
- Test: `tests/unit/test_llm_providers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `OpenAICompatibleLLM.stream_complete(messages: LanguageModelInput, *, structured_output: StructuredOutputRequest | None = None, timeout_override: int | None = None) -> Iterator[str]` — yields plain text deltas. Task 5 probes for this method by name via `getattr(llm, "stream_complete", None)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_llm_providers.py`. Match the file's existing style for faking `requests.Session` — read the file first and reuse whatever fixture or monkeypatch pattern it already uses for `complete`; only fall back to the shape below if none exists.

```python
def test_stream_puts_the_json_schema_in_the_request_body(monkeypatch):
    """structured_response_format was accepted and silently dropped."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self): pass
        def iter_lines(self, decode_unicode=False):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield "data: [DONE]"
        def close(self): pass

    llm = OpenAICompatibleLLM(LLMConfig(model_provider="openai", model_name="m", api_key="k"))

    def _post(url, headers=None, json=None, stream=None, timeout=None):
        captured.update(json)
        return _Resp()

    monkeypatch.setattr(llm._session, "post", _post)
    list(llm.stream([{"role": "user", "content": "q"}],
                    structured_response_format={"type": "json_schema",
                                                "json_schema": {"name": "answer_draft"}}))
    assert captured["response_format"]["json_schema"]["name"] == "answer_draft"


def test_stream_complete_yields_text_deltas_and_applies_the_schema(monkeypatch):
    captured: dict = {}

    class _Resp:
        def raise_for_status(self): pass
        def iter_lines(self, decode_unicode=False):
            yield 'data: {"choices":[{"delta":{"content":"{\\"abstain\\""}}]}'
            yield 'data: {"choices":[{"delta":{"content":": false}"}}]}'
            yield "data: [DONE]"
        def close(self): pass

    llm = OpenAICompatibleLLM(LLMConfig(model_provider="openai", model_name="m", api_key="k"))
    monkeypatch.setattr(llm._session, "post",
                        lambda *a, **kw: (captured.update(kw["json"]), _Resp())[1])

    request = StructuredOutputRequest(name="answer_draft", schema={"type": "object"})
    chunks = list(llm.stream_complete([{"role": "user", "content": "q"}],
                                      structured_output=request))

    assert "".join(chunks) == '{"abstain": false}'
    assert captured["response_format"]["json_schema"]["name"] == "answer_draft"
    assert captured["stream"] is True


def test_stream_complete_omits_the_schema_when_the_provider_cannot_enforce_it(monkeypatch):
    """PROMPT_ONLY providers must not receive a response_format they will reject."""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self): pass
        def iter_lines(self, decode_unicode=False):
            yield "data: [DONE]"
        def close(self): pass

    llm = OpenAICompatibleLLM(LLMConfig(model_provider="ollama", model_name="m"))
    monkeypatch.setattr(llm._session, "post",
                        lambda *a, **kw: (captured.update(kw["json"]), _Resp())[1])

    list(llm.stream_complete([{"role": "user", "content": "q"}],
                             structured_output=StructuredOutputRequest(name="d", schema={})))
    assert "response_format" not in captured
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_llm_providers.py -k "stream_complete or schema_in_the_request_body" -v
```

Expected: the first fails on a missing `response_format` key, the other two fail with `AttributeError: 'OpenAICompatibleLLM' object has no attribute 'stream_complete'`.

- [ ] **Step 3: Honour `structured_response_format` in `stream`**

In `src/internal/llm/providers.py`, inside `stream`, after the `tool_choice` block that ends with `body["tool_choice"] = _TOOL_CHOICE_MAP.get(tc, "auto")`, add:

```python
        if structured_response_format:
            body["response_format"] = structured_response_format
```

- [ ] **Step 4: Add the `stream_complete` adapter**

Add immediately after the `stream` method. Keep the `LLMClient`-shaped signature — this is the seam `pipeline.py` probes, and mirroring `complete`'s parameter names is what makes the `getattr` probe safe against other implementations.

```python
    def stream_complete(
        self,
        messages: LanguageModelInput,
        *,
        structured_output: StructuredOutputRequest | None = None,
        timeout_override: int | None = None,
    ) -> Iterator[str]:
        """Stream a completion as plain text deltas.

        The `LLMClient`-shaped counterpart to `stream`: it takes the same
        `structured_output` request `complete` does and yields text, so callers
        never handle provider chunk objects. Probed by name in the answer
        pipeline; absent implementations simply fall back to `complete`.
        """
        response_format: dict | None = None
        if (
            structured_output
            and self.structured_output_capability
            is StructuredOutputCapability.JSON_SCHEMA
        ):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": structured_output.name,
                    "strict": structured_output.strict,
                    "schema": structured_output.schema,
                },
            }
        for chunk in self.stream(
            messages,
            structured_response_format=response_format,
            timeout_override=timeout_override,
        ):
            content = chunk.choice.delta.content
            if content:
                yield content
```

Confirm `Iterator` is imported in the module's typing imports; `stream` already returns `Iterator[ModelResponseStream]`, so it should be.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/unit/test_llm_providers.py -v
```

Expected: PASS, including all pre-existing tests in the file.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/llm/providers.py tests/unit/test_llm_providers.py
git commit -m "feat(llm): add stream_complete, and honour the schema stream already accepted

stream() took structured_response_format and never wrote it into the body, so a
schema-constrained stream silently produced free prose. Fixed, and wrapped in an
LLMClient-shaped stream_complete that yields text deltas -- the seam the answer
pipeline probes, so provider chunk objects stay in the provider."
```

---

### Task 2: Order the answer draft so `abstain` precedes `claims`

`abstain: true` discards every claim (`src/context/safety.py:95`). With `claims` first in the payload, a streaming reader learns the draft abstained only after streaming the whole claim list — it would have to retract everything. Reordering makes abstain decidable before the first claim arrives.

**Files:**
- Modify: `src/context/structured_output.py:34-63` (the `_ANSWER_DRAFT_JSON_SCHEMA` literal)
- Modify: `src/context/prompts.py:103-109`
- Test: `tests/unit/test_rag_structured_output.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `answer_draft_json_schema()` returns a schema whose `required` list and `properties` dict are ordered `abstain`, `missing_information`, `claims`. Task 4's reader relies on this ordering at runtime but must not require it.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_rag_structured_output.py`:

```python
def test_answer_draft_schema_puts_abstain_before_claims():
    """Streaming must decide abstain before any claim it would discard."""
    schema = answer_draft_json_schema()
    assert list(schema["properties"]) == ["abstain", "missing_information", "claims"]
    assert list(schema["required"]) == ["abstain", "missing_information", "claims"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/test_rag_structured_output.py::test_answer_draft_schema_puts_abstain_before_claims -v
```

Expected: FAIL — the current order is `["claims", "missing_information", "abstain"]`.

- [ ] **Step 3: Reorder the schema literal**

In `src/context/structured_output.py`, rewrite `_ANSWER_DRAFT_JSON_SCHEMA` so both `required` and `properties` list `abstain` first. Move the existing property definitions unchanged — only their order changes:

```python
_ANSWER_DRAFT_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    # Order is load-bearing: under strict structured output the provider emits
    # properties in schema order, and a streaming reader must know whether the
    # draft abstained before it reads the claims that abstaining discards.
    "required": ["abstain", "missing_information", "claims"],
    "properties": {
        "abstain": {"type": "boolean"},
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}
```

- [ ] **Step 4: Reorder the prompt instruction**

In `src/context/prompts.py`, the sentence at line 103 currently reads `"Return only one JSON object with exactly these keys: claims, missing_information, abstain."`. Change the key order to match the schema and state the ordering requirement, which is what carries this to `PROMPT_ONLY` providers that get no schema at all:

```python
        "Return only one JSON object with exactly these keys, in this order: "
        "abstain, missing_information, claims. Each claim must contain exactly "
```

Leave the rest of the instruction — the wording about claims containing `text` and `evidence_ids`, and about recording gaps rather than guessing — exactly as it is.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/unit/test_rag_structured_output.py tests/unit/test_rag_safety.py -v
```

Expected: PASS. `parse_answer_draft` compares key *sets* (`safety.py:47`), so it is order-insensitive and no existing test should change. If a test asserts on key order or on the exact prompt string, that assertion is now wrong — update it to the new order rather than reverting the source.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/context/structured_output.py src/context/prompts.py tests/unit/test_rag_structured_output.py
git commit -m "refactor(context): order the answer draft abstain-first

abstain:true discards every claim, so with claims first a streaming reader could
only learn the draft abstained after streaming everything it would have to
retract. Reordered in the schema and the prompt. parse_answer_draft compares key
sets, so old-order drafts still parse."
```

---

### Task 3: Extract `verify_claim` and `render_claim`

`verify_answer_draft` already loops over claims with no cross-claim dependency — each verdict is a function of the claim, the evidence map and the threshold. Extracting that body lets the streaming path verify one claim the moment it arrives, using *exactly* the code the batch path uses. This is a pure refactor: no behaviour change.

**Files:**
- Modify: `src/context/safety.py:113-194`
- Test: `tests/unit/test_rag_safety.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `verify_claim(claim: AnswerClaim, evidence_by_id: dict[str, EvidenceSource], *, overlap_threshold: float = 0.15) -> ClaimVerdict`
  - `render_claim(claim: AnswerClaim) -> str` — one claim rendered as it appears in the final answer.
  - `render_claims(claims: list[AnswerClaim]) -> str` — the join, or `CANONICAL_ABSTENTION` when empty.

  Task 4 uses none of these; Task 5 uses all three.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_rag_safety.py` (the file already has an `_evidence()` helper returning `D1` and `T1` — reuse it):

```python
def test_verify_claim_matches_the_batch_verdict():
    evidence = _evidence()
    by_id = {item.id: item for item in evidence}
    claim = AnswerClaim(text="FAISS is a vector similarity search library.",
                        evidence_ids=["D1"])

    single = verify_claim(claim, by_id)
    batch = verify_answer_draft(AnswerDraft(claims=[claim]), evidence)

    assert single.supported is batch.verdicts[0].supported
    assert single.overlap_scores == batch.verdicts[0].overlap_scores
    assert single.reason == batch.verdicts[0].reason


def test_verify_claim_rejects_unknown_evidence_ids():
    verdict = verify_claim(AnswerClaim(text="Anything.", evidence_ids=["NOPE"]), {})
    assert verdict.supported is False
    assert "unknown evidence IDs" in verdict.reason


def test_render_claim_matches_what_the_full_render_produces():
    claim = AnswerClaim(text="FAISS is a library.", evidence_ids=["D1", "T1"])
    assert render_claim(claim) == "FAISS is a library. [D1] [T1]"


def test_render_claims_abstains_when_empty():
    assert render_claims([]) == CANONICAL_ABSTENTION
```

Add `verify_claim`, `render_claim` and `render_claims` to the `from src.context.safety import (...)` block at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_rag_safety.py -k "verify_claim or render_claim" -v
```

Expected: FAIL with `ImportError: cannot import name 'verify_claim'`.

- [ ] **Step 3: Extract the three helpers**

In `src/context/safety.py`, add `verify_claim` above `verify_answer_draft`, lifting the body of the existing per-claim loop verbatim:

```python
def verify_claim(
    claim: AnswerClaim,
    evidence_by_id: dict[str, EvidenceSource],
    *,
    overlap_threshold: float = 0.15,
) -> ClaimVerdict:
    """Verify one claim against the evidence map.

    Extracted from `verify_answer_draft`'s loop so the streaming path can judge a
    claim the moment it arrives using the same code the batch path uses. Verdicts
    carry no cross-claim dependency, which is what makes this safe to split out.
    """
    missing = sorted(set(claim.evidence_ids) - evidence_by_id.keys())
    if missing:
        return ClaimVerdict(
            claim=claim,
            supported=False,
            reason=f"unknown evidence IDs: {', '.join(missing)}",
        )

    claim_tokens = _tokenize(claim.text)
    scores = {
        evidence_id: _overlap(claim_tokens, _tokenize(evidence_by_id[evidence_id].text))
        for evidence_id in claim.evidence_ids
    }
    supported = bool(scores) and all(
        score >= overlap_threshold for score in scores.values()
    )
    return ClaimVerdict(
        claim=claim,
        supported=supported,
        overlap_scores=scores,
        reason=None if supported else "insufficient lexical support",
    )
```

Then replace the loop in `verify_answer_draft` (the block from `for claim in draft.claims:` down to the closing of the second `verdicts.append(...)`, currently lines 117-146) with:

```python
    verdicts = [
        verify_claim(claim, evidence_by_id, overlap_threshold=overlap_threshold)
        for claim in draft.claims
    ]
    unknown_ids = {
        evidence_id
        for verdict in verdicts
        for evidence_id in set(verdict.claim.evidence_ids) - evidence_by_id.keys()
    }
```

Delete the now-orphaned `verdicts: list[ClaimVerdict] = []` and `unknown_ids: set[str] = set()` initialisers above it. Everything from `supported_claims = [...]` onward is unchanged.

- [ ] **Step 4: Extract the render helpers**

Replace `render_verified_answer` (line 187) with a delegating version, so a single claim renders identically whether it is streamed or joined at the end:

```python
def render_claim(claim: AnswerClaim) -> str:
    """Render one claim exactly as it appears in the final answer."""
    return f"{claim.text} {' '.join(f'[{item}]' for item in claim.evidence_ids)}"


def render_claims(claims: list[AnswerClaim]) -> str:
    """Join rendered claims, or abstain when there are none."""
    if not claims:
        return CANONICAL_ABSTENTION
    return " ".join(render_claim(claim) for claim in claims)


def render_verified_answer(result: VerificationResult) -> str:
    """Render only supported claims, or the canonical abstention."""
    return render_claims(result.supported_claims)
```

- [ ] **Step 5: Run the full safety and pipeline suites**

```bash
pytest tests/unit/test_rag_safety.py tests/unit/test_context_pipeline.py tests/unit/test_rag_pipeline_integration.py -v
```

Expected: PASS with no test modifications. This is a pure refactor — if any pre-existing test needs changing, the extraction changed behaviour and must be corrected.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/context/safety.py tests/unit/test_rag_safety.py
git commit -m "refactor(context): extract verify_claim and render_claim

Claim verdicts have no cross-claim dependency, so the loop body splits out
cleanly. The streaming path can then judge and render a claim on arrival using
exactly the code the batch path uses, rather than a parallel implementation that
could drift. Pure refactor."
```

---

### Task 4: The incremental draft reader

The one genuinely new component. It reads complete claim objects out of a partial JSON draft. It is **strictly advisory**: on anything unexpected it gives up permanently and the run falls back to today's behaviour. It never decides the final answer.

**Files:**
- Create: `src/context/streaming_draft.py`
- Test: `tests/unit/test_streaming_draft.py`

**Interfaces:**
- Consumes: `AnswerClaim` from `src.context.models`.
- Produces: `IncrementalDraftReader(evidence_ids: set[str])` with:
  - `feed(chunk: str) -> list[AnswerClaim]` — claims completed by this chunk, in order.
  - `.abstain: bool | None` — `None` until decided.
  - `.gave_up: bool` — once `True`, `feed` returns `[]` forever.

  Task 5 constructs the reader and calls `feed`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_streaming_draft.py`:

```python
"""Tests for the incremental answer-draft reader."""

from __future__ import annotations

from src.context.streaming_draft import IncrementalDraftReader

IDS = {"D1", "D2"}


def _feed_all(reader: IncrementalDraftReader, text: str, size: int = 7) -> list:
    """Feed in small chunks, to prove claim boundaries survive arbitrary splits."""
    out = []
    for index in range(0, len(text), size):
        out.extend(reader.feed(text[index : index + size]))
    return out


DRAFT = (
    '{"abstain": false, "missing_information": [], "claims": ['
    '{"text": "First claim.", "evidence_ids": ["D1"]}, '
    '{"text": "Second claim.", "evidence_ids": ["D1", "D2"]}]}'
)


def test_yields_each_claim_as_it_completes():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(reader, DRAFT)
    assert [claim.text for claim in claims] == ["First claim.", "Second claim."]
    assert claims[1].evidence_ids == ["D1", "D2"]
    assert reader.abstain is False
    assert reader.gave_up is False


def test_a_claim_is_not_yielded_until_its_closing_brace_arrives():
    reader = IncrementalDraftReader(IDS)
    partial = '{"abstain": false, "missing_information": [], "claims": [{"text": "Half'
    assert reader.feed(partial) == []
    assert reader.abstain is False


def test_abstaining_drafts_yield_nothing():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": true, "missing_information": [], "claims": ['
        '{"text": "Discarded.", "evidence_ids": ["D1"]}]}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_claims_before_abstain_yields_nothing():
    """Old key order: abstain is not decidable in time, so refuse to stream."""
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"claims": [{"text": "Too early.", "evidence_ids": ["D1"]}], '
        '"missing_information": [], "abstain": false}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_unknown_evidence_ids_give_up():
    """The whole-draft parse rejects these, so streaming must not race ahead."""
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": false, "missing_information": [], "claims": ['
        '{"text": "Bad cite.", "evidence_ids": ["NOPE"]}]}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_malformed_claim_gives_up_without_raising():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": false, "missing_information": [], "claims": ['
        '{"text": "Ok.", "evidence_ids": ["D1"], "extra": 1}]}',
    )
    assert claims == []
    assert reader.gave_up is True


def test_braces_inside_claim_text_do_not_split_the_object():
    reader = IncrementalDraftReader(IDS)
    claims = _feed_all(
        reader,
        '{"abstain": false, "missing_information": [], "claims": ['
        '{"text": "Use {\\"k\\": 1} in config.", "evidence_ids": ["D1"]}]}',
    )
    assert [claim.text for claim in claims] == ['Use {"k": 1} in config.']


def test_prose_prefix_gives_up():
    """Some providers wrap JSON in a fence or a sentence."""
    reader = IncrementalDraftReader(IDS)
    assert _feed_all(reader, 'Sure! ```json\n{"abstain": false') == []


def test_feed_after_giving_up_stays_silent():
    reader = IncrementalDraftReader(IDS)
    _feed_all(reader, '{"abstain": true, "missing_information": [], "claims": [')
    assert reader.gave_up is True
    assert reader.feed('{"text": "x", "evidence_ids": ["D1"]}]}') == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_streaming_draft.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.context.streaming_draft'`.

- [ ] **Step 3: Implement the reader**

Create `src/context/streaming_draft.py`:

```python
"""Incremental reader for a partially-received answer draft.

Assist's grounded path streams a JSON draft whose claims are shown only after
they verify. This reader extracts complete claim objects from the partial text so
each can be verified and emitted on arrival.

It is deliberately advisory. It never decides the final answer -- that always
comes from the whole-text `parse_answer_draft`. On anything it does not fully
recognise it gives up permanently and the run reverts to non-streaming
behaviour, because emitting a claim that later proves wrong is worse than
emitting nothing.
"""

from __future__ import annotations

import json
import re

from .models import AnswerClaim

_ABSTAIN = re.compile(r'"abstain"\s*:\s*(true|false)')
_CLAIMS = re.compile(r'"claims"\s*:\s*\[')


def _next_object(text: str, start: int) -> tuple[str, int] | None:
    """Return (object_text, end_index) for the first complete {...} at or after start.

    Returns None when the text is merely incomplete, so the caller waits for more.
    String-aware, so braces and escaped quotes inside claim text do not confuse
    the depth count.
    """
    index = start
    length = len(text)
    while index < length and text[index] in ", \t\r\n":
        index += 1
    if index >= length or text[index] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False
    cursor = index
    while cursor < length:
        char = text[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[index : cursor + 1], cursor + 1
        cursor += 1
    return None


class IncrementalDraftReader:
    """Yield complete claims from a draft as it streams in."""

    def __init__(self, evidence_ids: set[str]) -> None:
        self._evidence_ids = evidence_ids
        self._buffer = ""
        self._abstain: bool | None = None
        self._cursor: int | None = None
        self._gave_up = False

    @property
    def abstain(self) -> bool | None:
        return self._abstain

    @property
    def gave_up(self) -> bool:
        return self._gave_up

    def feed(self, chunk: str) -> list[AnswerClaim]:
        if self._gave_up:
            return []
        self._buffer += chunk

        if self._abstain is None:
            match = _ABSTAIN.search(self._buffer)
            if match is None:
                # Claims already opened but abstain is still unknown: the draft
                # is in the old key order, and a claim we emit now could be
                # discarded by an abstain we have not read yet.
                if _CLAIMS.search(self._buffer):
                    self._gave_up = True
                return []
            self._abstain = match.group(1) == "true"
            if self._abstain:
                # Every claim is discarded, so there is nothing to stream.
                self._gave_up = True
                return []

        if self._cursor is None:
            opening = _CLAIMS.search(self._buffer)
            if opening is None:
                # Only give up once we can tell this is not a draft at all;
                # a bare prefix may simply be incomplete.
                if self._buffer.lstrip()[:1] not in ("", "{"):
                    self._gave_up = True
                return []
            self._cursor = opening.end()

        claims: list[AnswerClaim] = []
        while True:
            found = _next_object(self._buffer, self._cursor)
            if found is None:
                return claims
            object_text, self._cursor = found
            try:
                claims.append(self._build_claim(object_text))
            except ValueError:
                self._gave_up = True
                return claims

    def _build_claim(self, object_text: str) -> AnswerClaim:
        """Apply exactly the per-claim rules `parse_answer_draft` applies."""
        value = json.loads(object_text)
        if not isinstance(value, dict) or set(value) != {"text", "evidence_ids"}:
            raise ValueError("each claim must contain exactly text and evidence_ids")
        text = value["text"]
        evidence_ids = value["evidence_ids"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("claim text must be a non-empty string")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or not all(isinstance(item, str) for item in evidence_ids)
        ):
            raise ValueError("claim evidence_ids must be a non-empty list of strings")
        if set(evidence_ids) - self._evidence_ids:
            raise ValueError("unknown evidence IDs")
        return AnswerClaim(text=text.strip(), evidence_ids=list(evidence_ids))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/test_streaming_draft.py -v
```

Expected: PASS, all ten.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/context/streaming_draft.py tests/unit/test_streaming_draft.py
git commit -m "feat(context): incremental reader for a partial answer draft

Extracts complete claim objects from a draft as it streams, so each can be
verified and shown on arrival. Advisory by construction: it gives up permanently
on abstain, on old-order drafts where abstain is not yet decidable, on unknown
evidence IDs, and on any malformed claim -- because emitting a claim that later
proves wrong is worse than emitting nothing. The whole-text parse stays
authoritative."
```

---

### Task 5: The `on_claim` seam and the append-only retry

Where the invariant is actually enforced. Two things change in `_generate_guarded_answer`: claims commit as they verify, and the final answer is built from **committed claims** rather than solely from the last attempt's result.

That second part is essential and easy to miss. Today attempt 2 wholly replaces attempt 1. If attempt 1 streamed claims A and C and then the draft failed to parse, an unmodified final-answer computation would return only attempt 2's claims — and the user would have been shown text that is not in the answer. Building the answer from the committed list is what closes that hole.

**Files:**
- Modify: `src/context/pipeline.py:51-62` (`generate_answer` signature), `:126-129` (the `_generate_guarded_answer` call), `:155-265` (`_generate_guarded_answer`)
- Test: `tests/unit/test_context_pipeline.py`

**Interfaces:**
- Consumes: `verify_claim`, `render_claim`, `render_claims` (Task 3); `IncrementalDraftReader` (Task 4); `stream_complete` (Task 1).
- Produces: `generate_answer(request, *, llm=None, on_claim: Callable[[str], None] | None = None)`. Task 6 passes `on_claim` through.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_context_pipeline.py`. Follow the file's existing fake-LLM conventions — read it first and reuse its evidence/context builders rather than inventing new ones.

```python
class _StreamingLLM:
    """An LLM whose drafts arrive in chunks, one draft per attempt."""

    structured_output_capability = StructuredOutputCapability.PROMPT_ONLY

    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.complete_calls = 0

    def complete(self, messages, **kwargs):
        self.complete_calls += 1
        return self._drafts.pop(0)

    def stream_complete(self, messages, **kwargs):
        draft = self._drafts.pop(0)
        for index in range(0, len(draft), 5):
            yield draft[index : index + 5]


def test_streamed_claims_are_exactly_the_final_answer():
    """The invariant: the answer is the join of what was emitted, in order."""
    draft = _draft_json(abstain=False, claims=[
        ("FAISS is a vector similarity search library.", ["D1"]),
    ])
    emitted: list[str] = []
    result = generate_answer(_request(), llm=_StreamingLLM([draft]),
                             on_claim=emitted.append)

    assert emitted
    assert result.answer == " ".join(emitted)


def test_committed_claims_survive_a_retry():
    """Append-only: attempt 1's supported claims stay, attempt 2 appends."""
    first = _draft_json(abstain=False, claims=[
        ("FAISS is a vector similarity search library.", ["D1"]),
        ("Unrelated invented assertion about nothing.", ["D1"]),
    ])
    second = _draft_json(abstain=False, claims=[
        ("FAISS is a vector similarity search library.", ["D1"]),
        ("The service currently has 12 active indexes.", ["T1"]),
    ])
    emitted: list[str] = []
    result = generate_answer(_request(), llm=_StreamingLLM([first, second]),
                             on_claim=emitted.append)

    assert result.answer == " ".join(emitted)
    assert emitted[0].startswith("FAISS is a vector similarity search library.")
    # The supported claim from attempt 1 is emitted once, not re-emitted.
    assert len(emitted) == len(set(emitted))


def test_unsupported_claims_are_never_emitted():
    draft = _draft_json(abstain=False, claims=[
        ("Completely unrelated invented assertion.", ["D1"]),
    ])
    emitted: list[str] = []
    generate_answer(_request(), llm=_StreamingLLM([draft, draft]),
                    on_claim=emitted.append)
    assert emitted == []


def test_abstaining_draft_emits_nothing():
    draft = _draft_json(abstain=True, claims=[
        ("FAISS is a vector similarity search library.", ["D1"]),
    ])
    emitted: list[str] = []
    result = generate_answer(_request(), llm=_StreamingLLM([draft]),
                             on_claim=emitted.append)
    assert emitted == []
    assert result.answer == CANONICAL_ABSTENTION


def test_llm_without_stream_complete_falls_back():
    """The probe must not require the method."""
    draft = _draft_json(abstain=False, claims=[
        ("FAISS is a vector similarity search library.", ["D1"]),
    ])

    class _NoStream:
        structured_output_capability = StructuredOutputCapability.PROMPT_ONLY
        def complete(self, messages, **kwargs):
            return draft

    emitted: list[str] = []
    result = generate_answer(_request(), llm=_NoStream(), on_claim=emitted.append)
    assert "FAISS" in result.answer


def test_on_claim_none_uses_the_blocking_path():
    draft = _draft_json(abstain=False, claims=[
        ("FAISS is a vector similarity search library.", ["D1"]),
    ])
    llm = _StreamingLLM([draft])
    generate_answer(_request(), llm=llm)
    assert llm.complete_calls == 1
```

Write the `_draft_json(abstain, claims)` and `_request()` helpers to match the file's existing fixtures — `_request()` must build an `AnswerGenerationRequest` over a context whose evidence IDs are `D1`/`T1`, matching `evidence_from_context`'s numbering.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_context_pipeline.py -k "streamed or committed or emitted or abstaining or fall_back or blocking_path" -v
```

Expected: FAIL with `TypeError: generate_answer() got an unexpected keyword argument 'on_claim'`.

- [ ] **Step 3: Thread `on_claim` through `generate_answer`**

In `src/context/pipeline.py`, change the signature:

```python
def generate_answer(
    request: AnswerGenerationRequest,
    *,
    llm: LLMClient | None = None,
    on_claim: Callable[[str], None] | None = None,
) -> AnswerGenerationResult:
```

Add `from collections.abc import Callable` to the imports if absent, and extend the safety import to include `render_claim`, `render_claims` and `verify_claim`. Then pass it at the `_generate_guarded_answer` call site (line 126):

```python
        ) = _generate_guarded_answer(request, llm, prompt, evidence, on_claim)
```

`on_claim` deliberately affects only the guarded branch: the extractive and ungrounded branches produce no claims.

- [ ] **Step 4: Implement commit-on-verify inside the attempt loop**

Change `_generate_guarded_answer`'s signature to accept `on_claim: Callable[[str], None] | None`. Before the `for attempt in range(max_attempts):` line, add the committed-claim state:

```python
    evidence_by_id = {item.id: item for item in evidence}
    committed: list[AnswerClaim] = []
    committed_keys: set[tuple[str, tuple[str, ...]]] = set()
    stream_fn = getattr(llm, "stream_complete", None) if on_claim else None

    def _commit(claim: AnswerClaim) -> None:
        """Emit a supported claim once. Emitted claims are permanent."""
        key = (claim.text, tuple(claim.evidence_ids))
        if key in committed_keys:
            return
        committed_keys.add(key)
        committed.append(claim)
        on_claim(render_claim(claim))
```

Inside the loop, replace the single `raw = llm.complete(...)` call (line 199, inside the existing `try:`) with a streaming branch that falls back:

```python
            if stream_fn is not None:
                reader = IncrementalDraftReader(set(evidence_by_id))
                parts: list[str] = []
                for delta in stream_fn(
                    active_prompt.messages,
                    **({"structured_output": schema_request} if schema_request else {}),
                ):
                    parts.append(delta)
                    for claim in reader.feed(delta):
                        verdict = verify_claim(
                            claim,
                            evidence_by_id,
                            overlap_threshold=request.grounded_generation.overlap_threshold,
                        )
                        if verdict.supported:
                            _commit(claim)
                raw = "".join(parts)
            else:
                raw = llm.complete(
                    active_prompt.messages,
                    **({"structured_output": schema_request} if schema_request else {}),
                )
```

The streaming branch yields a plain `str`, so the `isinstance(raw, LLMResponse)` blocks below it are simply skipped — structured-output metadata (`applied`, `refused`, `incomplete_reason`) is not observable on this path. That is acceptable: those fields are reporting only, and `on_claim` is opt-in.

Then, after the existing `result = verify_answer_draft(...)` call, commit any supported claim the reader missed — for instance when it gave up partway — so the committed list is complete before the loop breaks:

```python
        if on_claim is not None:
            for claim in result.supported_claims:
                _commit(claim)
```

Leave the retry condition (`if draft.abstain or not result.unsupported_claims: break`) exactly as it is.

- [ ] **Step 5: Build the final answer from the committed claims**

At the end of `_generate_guarded_answer`, the streaming path must return the join of what was emitted. Immediately before the existing `if result is None:` block, insert:

```python
    if on_claim is not None and committed:
        # The invariant: the answer is exactly what the user was shown. Built
        # from the committed list rather than the last attempt's result, because
        # a later attempt that parsed differently must not silently drop text
        # already on the user's screen.
        status = (
            VerificationStatus.VERIFIED
            if result is not None and not result.unsupported_claims
            else VerificationStatus.PARTIAL
        )
        return (
            render_claims(committed),
            result.confidence if result is not None else 0.0,
            status,
            attempt,
            requested,
            applied,
            downgraded,
            category,
        )
```

Ensure `AnswerClaim` and `VerificationStatus` are imported in `pipeline.py`, and add `from .streaming_draft import IncrementalDraftReader`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest tests/unit/test_context_pipeline.py tests/unit/test_rag_safety.py tests/unit/test_rag_pipeline_integration.py -v
```

Expected: PASS. Every pre-existing test must pass **unmodified** — they all call `generate_answer` without `on_claim`, which is the inert path. A pre-existing failure means the default path changed and must be fixed.

- [ ] **Step 7: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/context/pipeline.py tests/unit/test_context_pipeline.py
git commit -m "feat(context): commit claims as they verify, behind on_claim

A claim is emitted only once it verifies supported, so nothing shown is ever
retracted, and the corrective retry can only append. The final answer is built
from the committed list rather than the last attempt's result -- otherwise an
attempt that parsed differently would drop text already on the user's screen,
which is the exact failure the append-only invariant exists to prevent.

on_claim=None leaves every existing caller on the untouched blocking path."
```

---

### Task 6: Thread `on_claim` to the SSE route

**Files:**
- Modify: `src/agents/search/agentic_rag.py:191` (`run` signature), `:323-332` (the `to_thread` call)
- Modify: `src/internal/servers/web/app.py:640-683` (`_run_agentic_rag`), `:1886-1935` (`stream_agent`), `:1684-1700` (the `chat_loop` call site)
- Test: `tests/unit/test_agentic_rag.py`, `tests/unit/test_token_streaming.py`

**Interfaces:**
- Consumes: `generate_answer(..., on_claim=...)` (Task 5).
- Produces: SSE event `{"type": "claim", "text": str}`, emitted before the terminal `answer` event. Task 7 consumes it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_agentic_rag.py`:

```python
@pytest.mark.asyncio
async def test_agentic_rag_forwards_on_claim(monkeypatch):
    seen: dict = {}

    def _fake_generate_answer(request, *, llm=None, on_claim=None):
        seen["on_claim"] = on_claim
        return _stub_generation_result()

    monkeypatch.setattr("src.agents.search.agentic_rag.generate_answer",
                        _fake_generate_answer)

    callback = lambda text: None
    loop = AgenticRAGLoop(AgenticRAGConfig(max_rounds=1), llm=_stub_llm())
    await loop.run("q", on_claim=callback)

    assert seen["on_claim"] is callback
```

Reuse the file's existing stubs for the loop's LLM and retrieval. `_stub_generation_result()` must return an `AnswerGenerationResult` — the loop reads only `.answer`, `.citations` and `.context` off it (`agentic_rag.py:341-344`), so the minimum is:

```python
def _stub_generation_result():
    context = SearchContextBundle(query="q", documents=[])
    return AnswerGenerationResult(
        answer="stub", citations=[], context=context,
        prompt=build_chat_prompt("q", context),
    )
```

If `test_agentic_rag.py` already has an equivalent helper, use that instead of adding a second one.

Add to `tests/unit/test_token_streaming.py`:

```python
def test_stream_agent_emits_claim_events_before_the_answer():
    """A claim event carries text; the terminal answer event still arrives."""
    events = _collect_stream_events(query="what is faiss?", mode="chat_loop")
    types = [event["type"] for event in events]
    assert "claim" in types
    assert types.index("claim") < types.index("answer")
    assert all(isinstance(event["text"], str)
               for event in events if event["type"] == "claim")
```

Follow the file's existing helper for driving the SSE endpoint; if it has no `_collect_stream_events`, reuse whatever TestClient pattern it already uses. Note the model-load gotcha: web TestClient tests can hang loading `SEARCH_AGENT_MODEL` — `tests/conftest.py` already bakes in the skip, so run via plain `pytest` and check `examples/run_web_integration_tests.sh` if anything stalls.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_agentic_rag.py -k on_claim tests/unit/test_token_streaming.py -k claim_events -v
```

Expected: FAIL — `run()` rejects `on_claim`, and no `claim` event is emitted.

- [ ] **Step 3: Thread it through the loop**

In `src/agents/search/agentic_rag.py`, add `on_claim: Callable[[str], None] | None = None` as a keyword-only parameter to `run`, importing `Callable` from `collections.abc` if needed. Pass it into the existing offload — the `to_thread` call stays, since #547 added it precisely to keep this blocking call off the event loop:

```python
        gen_result = await asyncio.to_thread(
            generate_answer,
            AnswerGenerationRequest(
                question=question,
                context=merged,
                chat_history=chat_history or [],
                user_memory=user_memory,
            ),
            llm=self.llm,
            on_claim=on_claim,
        )
```

- [ ] **Step 4: Thread it through the route**

In `src/internal/servers/web/app.py`, add `on_claim=None` to `_run_agentic_rag`'s keyword arguments and forward it to `rag_loop.run(...)`. At the `chat_loop` call site (line ~1691), pass the callback through from the enclosing `_run_agent_impl` parameters, alongside the existing `on_turn` and `on_trace`.

In `stream_agent`, define the callback next to `on_turn`. `generate_answer` runs in a worker thread, so the callback is invoked off the event loop and **must** marshal back — `queue.put_nowait` from another thread is not safe:

```python
        loop = asyncio.get_running_loop()

        def _offer(item: dict) -> None:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass  # the terminal answer event still carries the full text

        def on_claim(text: str) -> None:
            # Called from the generate_answer worker thread, so hop back to the
            # loop before touching the queue.
            loop.call_soon_threadsafe(_offer, {"type": "claim", "text": text})
```

Extend the endpoint's docstring `Emits:` block with the new event:

```
          {"type": "claim",    "text": "..."}             — one verified claim
```

Pass `on_claim=on_claim` into the `_run_agent_impl(...)` call, and add the matching parameter to `_run_agent_impl`, defaulting to `None` so the non-streaming `POST /api/agent` caller is unaffected.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/unit/test_agentic_rag.py tests/unit/test_token_streaming.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the full backend suite**

```bash
pytest
```

Expected: PASS. This is the checkpoint that proves `on_claim=None` really is inert across every caller.

- [ ] **Step 7: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/agents/search/agentic_rag.py src/internal/servers/web/app.py tests/unit/test_agentic_rag.py tests/unit/test_token_streaming.py
git commit -m "feat(assist): emit verified claims as SSE events

Threads on_claim from the stream endpoint through _run_agentic_rag into the loop's
existing to_thread offload. The callback fires on the worker thread, so it hops
back via call_soon_threadsafe before touching the queue. The terminal answer and
done events are unchanged, so a client that ignores claim events behaves exactly
as before."
```

---

### Task 7: Render streamed claims in the Assist UI

**Files:**
- Modify: `web/src/types.ts:225-283` (add `SSEClaimEvent` to the union)
- Modify: `web/src/pages/AssistPage.tsx:141-160`
- Test: `web/src/__tests__/api.test.ts`, and the `AssistPage` test file if one exists

**Interfaces:**
- Consumes: the `{"type": "claim", "text": string}` SSE event (Task 6).

- [ ] **Step 1: Write the failing test**

Add to `web/src/__tests__/api.test.ts`, matching the file's existing `streamAgent` test shape:

```ts
it("yields claim events", async () => {
  mockFetchStream([
    'data: {"type":"claim","text":"FAISS is a library. [D1]"}\n\n',
    'data: {"type":"answer","text":"FAISS is a library. [D1]"}\n\n',
  ]);
  const types: string[] = [];
  for await (const event of streamAgent({ query: "q" })) types.push(event.type);
  expect(types).toEqual(["claim", "answer"]);
});
```

Use whatever fetch-mocking helper the file already defines rather than `mockFetchStream` if the name differs.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run src/__tests__/api.test.ts
```

Expected: FAIL — a type error on the unknown `claim` variant, or a mismatched event list.

- [ ] **Step 3: Add the event type**

In `web/src/types.ts`, add the interface beside `SSEAnswerEvent`:

```ts
export interface SSEClaimEvent {
  type: "claim";
  text: string;
}
```

and add `| SSEClaimEvent` to the `SSEEvent` union at line 277.

- [ ] **Step 4: Append claims in the Assist page**

In `web/src/pages/AssistPage.tsx`, add a branch before the existing `answer` branch. Claims arrive pre-rendered with their citation markers, so joining with a space matches exactly what the backend's `render_claims` produces:

```tsx
        } else if (event.type === "claim") {
          accumulatedAnswer = accumulatedAnswer
            ? `${accumulatedAnswer} ${event.text}`
            : event.text;
          setStreamingAnswer(accumulatedAnswer);
        } else if (event.type === "answer") {
```

The existing `answer` branch already overwrites `accumulatedAnswer` with the authoritative full text, so it reconciles any drift for free.

- [ ] **Step 5: Run the frontend checks**

```bash
cd web && npm run typecheck && npx vitest run
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts web/src/pages/AssistPage.tsx web/src/__tests__/api.test.ts
git commit -m "feat(web): append streamed claims into the Assist answer

Claims arrive pre-rendered with citation markers, so a space join reproduces the
backend's render_claims exactly. The terminal answer event still overwrites with
the authoritative text, reconciling any drift."
```

---

### Task 8: Verify end to end against a live stack

Unit tests cannot prove the model actually emits `abstain` first, which is the assumption the whole streaming path rests on. This task checks it against a real provider.

**Files:** `examples/verify_claim_streaming.py`, `tests/unit/test_verify_claim_streaming.py` — the checks below turned out to be worth keeping as a runnable script rather than a one-off shell session.

- [x] **Step 1: Start the stack**

```bash
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
cd web && npm run dev
```

`OPENAI_API_KEY` must be set, or `llm` is `None` at `app.py:1258` and Assist never reaches the grounded path at all.

- [x] **Step 2: Confirm claims stream**

```bash
curl -N -X POST http://127.0.0.1:7860/api/agent/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is FAISS?","top_k":5}'
```

Expected: one or more `{"type":"claim",...}` events arriving **before** the `answer` event, and the concatenation of their `text` fields, space-joined, equalling the `answer` event's `text`. That equality is the invariant — check it explicitly rather than eyeballing that text appeared.

- [x] **Step 3: Confirm the ordering assumption holds**

Check the server log or the Dev Console Request Inspector for the raw draft. Confirm the model emitted `abstain` before `claims`. **If it did not**, the reader gives up every time and claims never stream — the feature is silently inert. Report that rather than working around it: it means the schema ordering did not reach the provider, which is a Task 2 problem, not a reason to weaken the reader.

- [x] **Step 4: Confirm the browser renders incrementally**

Open `http://127.0.0.1:5173/assist`, ask a question, and confirm text appears progressively rather than all at once, and that no text visibly disappears or is rewritten once shown.

- [x] **Step 5: Report findings**

Report measured time-to-first-claim against time-to-completion. If they are equal, streaming is not working end to end regardless of what the unit tests say.

**Findings (run 2026-08-20 against a live OpenAI provider + demo retrieval on :8001).**

`python -m examples.verify_claim_streaming --query "..."` now performs Steps 2 and 3
and exits non-zero if the invariant breaks. Five timed runs plus one browser run,
all passing:

| query | claims | time-to-first-claim | time-to-answer | lead |
|---|---|---|---|---|
| "What is FAISS?" | 2 | 5.72s | 6.24s | 0.52s |
| "What is FAISS?" | 2 | 7.96s | 8.32s | 0.36s |
| "What is FAISS and how does it index vectors?" | 4 | 6.63s | 7.73s | 1.10s |
| hybrid retrieval / RRF / cross-encoder reranking | 2 | 8.50s | 11.86s | 3.36s |
| compare dense and sparse retrieval | 3 | 17.13s | 17.90s | 0.77s |

- **The invariant held on every run**: `answer == " ".join(claims)`, checked
  programmatically rather than by eye.
- **The abstain-first assumption holds on the live provider.** It needs no separate
  probe: the reader gives up unless `abstain` precedes `claims`, so claims arriving
  at all is the proof. Step 3 is therefore satisfied by Step 2 passing.
- **Step 4, the browser:** sampling the Answer region in-page at 100ms during a real
  `/assist` query gave three distinct growth states (171 → 383 → 573 chars over
  ~0.8s), each a strict prefix of the final text. Nothing is rewritten or removed
  once shown.
- **Time-to-first-claim is below time-to-completion on every run**, so the Definition
  of Done is met — but the honest headline is that the win is ~0.5-3.4s. Answer
  generation is only ~1s of a 6-18s request; **most Assist latency is upstream
  query-enhancement and retrieval**, not answer generation. That is where the next
  latency work belongs.

Two things that cost time and are worth writing down: the local `SEARCH_AGENT_MODEL`
(Qwen) fails to load without HF connectivity and that is **fine** — the grounded path
uses the OpenAI `llm`, not it. And in the UI, Enter in the Question textarea does not
submit; the Search button does.

---

## Definition of Done

- [x] `pytest` passes with no pre-existing test modified except where Task 2 changed a key-order or prompt-string assertion.
- [x] `cd web && npm run typecheck && npx vitest run` passes.
- [x] `ruff check . && ruff format --check .` clean.
- [x] Task 8 confirms claims stream from a live provider and that time-to-first-claim is materially below time-to-completion.
- [x] A PR is opened against `main` with both the spec and this plan referenced.
