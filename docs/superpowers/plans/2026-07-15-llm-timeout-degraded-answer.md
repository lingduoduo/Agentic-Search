# LLM Timeout Degraded Answer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an LLM timeout during guarded answer synthesis from a `502` with a leaked endpoint URL into a usable, abstain-shaped degraded answer.

**Architecture:** Mirror the existing `SchemaUnsupportedError` contract. `OpenAICompatibleLLM.complete` catches `requests.Timeout` and raises a typed `LLMTimeoutError` defined in `src/context/`; `_generate_guarded_answer` catches it and returns the existing 8-tuple immediately with a distinct answer string and a `"timeout"` category. No new architecture, no new trace attribute, no new status.

**Tech Stack:** Python 3.11+, `requests`, pytest, ruff.

Spec: `docs/superpowers/specs/2026-07-15-llm-timeout-degraded-answer-design.md`

## Global Constraints

- The new error is `LLMTimeoutError(RuntimeError)`, defined in `src/context/models.py` beside the `LLMClient` protocol, and re-exported from `src/context/__init__.py`. It must live in `src/context/` because the dependency arrow is `internal/llm → context`: `src/internal/llm/providers.py:23` already imports `SchemaUnsupportedError` from `src.context.structured_output`. Defining it in `src/internal/llm/` would invert that.
- The new answer constant is `TIMEOUT_DEGRADED_ANSWER = "I couldn't complete an answer in time. Please try again."`, defined in `src/context/safety.py` immediately beside `CANONICAL_ABSTENTION` (currently `safety.py:19`). It must **never** equal `CANONICAL_ABSTENTION`.
- Catch `requests.Timeout` only. **Do not** catch `requests.ConnectionError`, `requests.RequestException`, or bare `Exception`.
- **The `requests` hierarchy, verified empirically — do not re-derive it from intuition:**
  - `Timeout` is **not** a subclass of `HTTPError`; they are siblings under `RequestException`. This is precisely why the bug exists, and it means the two `except` clauses are independent and their order is not load-bearing.
  - `ReadTimeout` and `ConnectTimeout` are both subclasses of `Timeout`, so `except requests.Timeout` catches both. That is intended.
  - `ConnectTimeout` is **also** a subclass of `ConnectionError` (dual inheritance). So "connection errors are out of scope" applies only to **plain** `requests.ConnectionError`. Tests pinning the boundary must use plain `requests.ConnectionError` — using `ConnectTimeout` would assert the opposite of the intent and fail.
- The `LLMTimeoutError` message must be generic. It must **not** interpolate the original exception text or the endpoint URL — the whole point is to stop the leak, not relocate it.
- A timeout returns on the **first** attempt. It must not consume a retry.
- The degraded result keeps `structured_output_requested`, `structured_output_applied`, and `structured_output_downgraded` at whatever they were; only the answer, confidence, status, and category are fixed.
- Run `python3 -m ruff check` and `python3 -m ruff format` on every modified file before committing. Use `python3`, never `python`.

---

### Task 1: Typed error, degraded constant, and provider normalization

**Files:**
- Modify: `src/context/models.py` (add error beside the `LLMClient` protocol at lines 33-37)
- Modify: `src/context/__init__.py` (re-export, beside the `SchemaUnsupportedError` export at lines 43 and 82)
- Modify: `src/context/safety.py` (add constant beside `CANONICAL_ABSTENTION` at line 19)
- Modify: `src/internal/llm/providers.py` (import at ~line 23; catch in `complete()` at lines 286-318)
- Test: `tests/unit/test_llm_providers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `LLMTimeoutError(RuntimeError)` — importable as `from src.context.models import LLMTimeoutError` and `from src.context import LLMTimeoutError`. Task 2 catches this.
  - `TIMEOUT_DEGRADED_ANSWER: str` in `src/context/safety.py`. Task 2 returns this.

`tests/unit/test_llm_providers.py` is titled "Tests for OpenAICompatibleLLM" and already imports `MagicMock`, `patch`, `LLMConfig`, and `OpenAICompatibleLLM`. A timeout is not a structured-output concern, so the tests go here rather than in `test_llm_structured_output.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_llm_providers.py`. Add `import pytest` and `import requests` to the file's imports if absent, and `from src.context.models import LLMTimeoutError`:

```python
def _timeout_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider="openai",
            model_name="test-model",
            api_key="test-key",
        )
    )


def test_read_timeout_is_normalized_to_llm_timeout_error():
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ReadTimeout(
            "HTTPSConnectionPool(host='secret-endpoint.invalid', port=443): Read timed out."
        ),
    ):
        with pytest.raises(LLMTimeoutError):
            llm.complete([{"role": "user", "content": "hi"}])


def test_connect_timeout_is_normalized_to_llm_timeout_error():
    # ConnectTimeout subclasses BOTH Timeout and ConnectionError. It is a timeout,
    # so it must be handled — the ConnectionError exclusion does not reach it.
    llm = _timeout_llm()
    with patch.object(
        llm._session, "post", side_effect=requests.ConnectTimeout("connect timed out")
    ):
        with pytest.raises(LLMTimeoutError):
            llm.complete([{"role": "user", "content": "hi"}])


def test_llm_timeout_error_does_not_leak_endpoint_or_original_text():
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ReadTimeout(
            "HTTPSConnectionPool(host='secret-endpoint.invalid', port=443): Read timed out."
        ),
    ):
        with pytest.raises(LLMTimeoutError) as caught:
            llm.complete([{"role": "user", "content": "hi"}])
    assert "secret-endpoint.invalid" not in str(caught.value)
    assert "HTTPSConnectionPool" not in str(caught.value)


def test_plain_connection_error_still_propagates():
    # Scope boundary: a host that is down or refusing is a misconfiguration signal
    # and must stay loud. Plain ConnectionError is NOT a Timeout, so it escapes.
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ConnectionError("connection refused"),
    ):
        with pytest.raises(requests.ConnectionError):
            llm.complete([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_llm_providers.py -q`

Expected: FAIL. The `from src.context.models import LLMTimeoutError` import raises `ImportError: cannot import name 'LLMTimeoutError'`, erroring collection of the module. That single error is the expected RED signal for all four tests.

- [ ] **Step 3: Define the typed error**

In `src/context/models.py`, add directly below the `LLMClient` protocol (currently ends line 37):

```python
class LLMTimeoutError(RuntimeError):
    """The LLM call exceeded its timeout."""
```

- [ ] **Step 4: Re-export both new names**

In `src/context/__init__.py`, matching the file's existing one-import-per-line style:

```python
from .models import LLMTimeoutError
from .safety import TIMEOUT_DEGRADED_ANSWER
```

Place the first beside the existing `SchemaUnsupportedError` import (line 43) and the second beside the existing `CANONICAL_ABSTENTION` import (line 38). Add `"LLMTimeoutError"` and `"TIMEOUT_DEGRADED_ANSWER"` to `__all__`, beside `"SchemaUnsupportedError"` (line 82) and `"CANONICAL_ABSTENTION"` (line 93) respectively.

`TIMEOUT_DEGRADED_ANSWER` is re-exported for symmetry with `CANONICAL_ABSTENTION`, which `src/context/__init__.py:38,93` already exports and which `tests/unit/test_grounded_generation.py:12` imports from `src.context`. Task 2's tests import both from `src.context` the same way.

- [ ] **Step 5: Add the degraded answer constant**

In `src/context/safety.py`, directly below `CANONICAL_ABSTENTION` (line 19):

```python
TIMEOUT_DEGRADED_ANSWER = "I couldn't complete an answer in time. Please try again."
```

- [ ] **Step 6: Normalize the timeout in the provider**

In `src/internal/llm/providers.py`, add `LLMTimeoutError` to the existing `src.context` import block at line 23. Note it comes from `src.context.models`, not `src.context.structured_output`, so this may be a new import line:

```python
from src.context.models import LLMTimeoutError
```

In `complete()`, add an `except requests.Timeout` clause to the existing `try` (lines 287-318). Put it **before** the `except requests.HTTPError` clause for readability — the two are siblings so order does not affect behavior, but grouping the new short clause first keeps the long HTTPError block last:

```python
        try:
            resp = self._session.post(
                self._endpoint,
                headers=self._headers,
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.Timeout:
            raise LLMTimeoutError("LLM request timed out") from None
        except requests.HTTPError as exc:
```

`from None` suppresses the original exception's context, which is what keeps the endpoint URL out of any downstream rendering. The existing `SchemaUnsupportedError` raise at line 315 uses the same `from None` for the same reason.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_llm_providers.py tests/unit/test_llm_structured_output.py -q`

Expected: PASS. `test_llm_structured_output.py` must be included and must stay green — it holds `test_explicit_unsupported_schema_400_is_classified_without_raw_body`, which proves the `HTTPError` path is unchanged.

- [ ] **Step 8: Lint and commit**

Run: `python3 -m ruff check src/context/models.py src/context/__init__.py src/context/safety.py src/internal/llm/providers.py tests/unit/test_llm_providers.py && python3 -m ruff format src/context/models.py src/context/__init__.py src/context/safety.py src/internal/llm/providers.py tests/unit/test_llm_providers.py`

Expected: `All checks passed!`

```bash
git add src/context/models.py src/context/__init__.py src/context/safety.py src/internal/llm/providers.py tests/unit/test_llm_providers.py
git commit -m "feat: normalize LLM timeouts to a typed LLMTimeoutError"
```

---

### Task 2: Return a degraded answer instead of raising

**Files:**
- Modify: `src/context/pipeline.py` (`_generate_guarded_answer`, the `try` at lines 184-194)
- Test: `tests/unit/test_grounded_generation.py`

**Interfaces:**
- Consumes: `LLMTimeoutError` and `TIMEOUT_DEGRADED_ANSWER` from Task 1.
- Produces: no new public signature. `_generate_guarded_answer` returns its existing 8-tuple `(answer, confidence, verification_status, retry_count, requested, applied, downgraded, category)`.

Before writing tests, read the existing `refused` return at `src/context/pipeline.py:197-207` — the degraded return is the same shape with a different answer and category.

This test file already has everything needed. `SequenceLLM` (line 27) **raises** any response that is a `BaseException` (`if isinstance(response, BaseException): raise response`), so `SequenceLLM(LLMTimeoutError(...))` is the stub — do not write a new one. `_bundle()` (line 52) supplies context, `generate_answer` and `CANONICAL_ABSTENTION` are already imported (lines 12, 22), and `test_structured_refusal_abstains` (line 288) is the model for asserting on `structured_output_category`. Note `SequenceLLM` defaults to `StructuredOutputCapability.PROMPT_ONLY`, so `structured_output_requested` will be `False`.

- [ ] **Step 1: Write the failing tests**

Add `LLMTimeoutError` and `TIMEOUT_DEGRADED_ANSWER` to the existing `from src.context import (...)` block at the top of `tests/unit/test_grounded_generation.py`, keeping its alphabetical ordering. Then append:

```python
def test_llm_timeout_returns_degraded_answer_without_retrying():
    llm = SequenceLLM(LLMTimeoutError("LLM request timed out"))

    result = generate_answer(
        AnswerGenerationRequest(question="What is FAISS?", context=_bundle()), llm=llm
    )

    assert result.answer == TIMEOUT_DEGRADED_ANSWER
    assert result.confidence == 0.0
    assert result.verification_status is VerificationStatus.ABSTAINED
    assert result.abstained is True
    assert result.structured_output_category == "timeout"
    assert result.citations == []
    assert len(llm.calls) == 1  # a timeout must not consume a retry


def test_timeout_answer_is_distinct_from_evidence_abstention():
    assert TIMEOUT_DEGRADED_ANSWER != CANONICAL_ABSTENTION
    assert "evidence" not in TIMEOUT_DEGRADED_ANSWER.lower()
```

The `len(llm.calls) == 1` assertion is the one that pins "no retry" — `SequenceLLM` records every call, and a second attempt would also raise `IndexError` from the exhausted response list, so a regression here fails loudly either way.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_grounded_generation.py -q`

Expected: FAIL — `LLMTimeoutError` propagates out of `_generate_guarded_answer` instead of being converted, so the first test errors with `LLMTimeoutError: LLM request timed out` rather than asserting.

- [ ] **Step 3: Handle the timeout**

In `src/context/pipeline.py`, import `TIMEOUT_DEGRADED_ANSWER` in the existing local import inside `_generate_guarded_answer` (currently `from .safety import parse_answer_draft, render_verified_answer, verify_answer_draft`), and add `LLMTimeoutError` to the module's imports from `.models`.

Add an `except LLMTimeoutError` clause to the existing `try` around `llm.complete` (lines 184-194), returning immediately:

```python
        try:
            raw = llm.complete(
                active_prompt.messages,
                **({"structured_output": schema_request} if schema_request else {}),
            )
        except LLMTimeoutError:
            return (
                TIMEOUT_DEGRADED_ANSWER,
                0.0,
                VerificationStatus.ABSTAINED,
                attempt,
                requested,
                applied,
                downgraded,
                "timeout",
            )
        except SchemaUnsupportedError:
```

Returning rather than `continue` is what makes a timeout skip the retry.

Note the second `llm.complete` call inside the `except SchemaUnsupportedError` handler (line 194) is **not** covered by this clause — a timeout on the post-downgrade retry still propagates. That is acceptable and deliberate: it is a narrow path (provider rejected the schema, then timed out on the immediate re-call), and widening the `try` would restructure the downgrade logic this change is not meant to touch. Do not attempt it here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_grounded_generation.py -q`

Expected: PASS.

- [ ] **Step 5: Full suite and lint**

Run: `env HF_HOME=/tmp/agentic-search-hf-cache python3 -m pytest -q tests/unit`

Expected: PASS with no failures. The baseline on `main` is 2616 passed; expect that plus the tests added by Tasks 1 and 2. The `HF_HOME` override is required — without it a sandbox cache lock causes an unrelated failure.

Run: `python3 -m ruff check src/context/pipeline.py tests/unit/test_grounded_generation.py && python3 -m ruff format src/context/pipeline.py tests/unit/test_grounded_generation.py && git diff --check`

Expected: `All checks passed!` and `git diff --check` exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/context/pipeline.py tests/unit/test_grounded_generation.py
git commit -m "feat: return a degraded answer when LLM synthesis times out"
```

---

### Task 3: Document the timeout category

**Files:**
- Modify: `docs/retrieval.md` (the `### Result and operational metadata` section)

**Interfaces:**
- Consumes: the names and values from Tasks 1 and 2 — `TIMEOUT_DEGRADED_ANSWER`, category `"timeout"`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Read the existing section**

Run: `sed -n '/### Result and operational metadata/,/## Retrieval setup/p' docs/retrieval.md`

That section documents `confidence`, `verification_status` (`verified`, `partial`, `abstained`), `abstained`, `tool_evidence`, and `retry_count`, and states that tracing records verification status and confidence. It currently gives no way for a reader to tell a timeout apart from an evidence abstention.

- [ ] **Step 2: Add the paragraph**

Insert after the paragraph beginning "Existing result fields—" and before the paragraph beginning "Tracing records counts and categories":

```markdown
A timed-out synthesis call returns a degraded answer rather than an error. It is
abstention-shaped—`confidence` `0.0`, `verification_status` `abstained`, no
citations—so it flows through the same safe paths as any other abstention, but it
carries its own answer text and the `structured_output_category` `timeout`. The
distinction is deliberate: the canonical abstention asserts a conclusion about the
evidence, which a timeout cannot support, and collapsing the two would make an LLM
outage indistinguishable from a normal low-confidence answer. A timeout does not
consume a generation retry.
```

- [ ] **Step 3: Verify no other doc contradicts it**

Run: `grep -rn "verification_status\|abstained\|CANONICAL_ABSTENTION" docs/ | grep -v superpowers/archive`

Expected: any active doc that enumerates the abstention triggers or the `structured_output_category` values now also accounts for `timeout`, or does not enumerate them at all. Fix any active doc that lists them but omits this one. Report what you found either way.

- [ ] **Step 4: Commit**

```bash
git add docs/retrieval.md
git commit -m "docs: document the timeout degraded answer and category"
```

---

## Completion Gate

Before claiming completion:

1. `git status --short` contains no unintended files. Note `.superpowers/sdd/task-3-report.md`, `task-4-report.md`, `task-5-report.md`, and `final-fix-report.md` are **tracked** files from earlier plans despite `.superpowers/` being in `.gitignore`. Do not overwrite them — write task reports under distinct names.
2. `env HF_HOME=/tmp/agentic-search-hf-cache python3 -m pytest -q tests/unit` passes with no failures.
3. `ruff check`, `ruff format --check`, and `git diff --check` pass over every modified file.
4. `test_explicit_unsupported_schema_400_is_classified_without_raw_body` passes unmodified — the guard proving the `HTTPError` path is untouched.
5. `test_plain_connection_error_still_propagates` passes — the guard proving the new handler does not over-catch.
6. Write the task report per `docs/development/self-review-reports.md` and validate it with `python3 examples/validate_task_report.py REPORT_FILE --require-tdd` (omit `--require-tdd` for Task 3, which is docs-only).
7. Perform an independent whole-branch review against the merge base; fix every Critical or Important finding, then re-review.
