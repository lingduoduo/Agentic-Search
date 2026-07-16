# LLM Timeout Degraded Answer Design

## Goal

Convert an LLM timeout during guarded answer synthesis from a `502` error into a usable
degraded answer, without misrepresenting the timeout as an evidence-based abstention.

## Current State

An LLM timeout during synthesis reaches the caller as an HTTP `502` whose detail contains the
raw exception text, including the provider endpoint URL.

The failure is a gap between two narrow `except` clauses:

1. `OpenAICompatibleLLM.complete` (`src/internal/llm/providers.py:286-318`) sets
   `timeout = kwargs.get("timeout_override") or 30` and posts. Its only handler is
   `except requests.HTTPError`, which special-cases JSON-Schema-unsupported `400`s and
   re-raises everything else. A socket timeout raises `requests.exceptions.Timeout`
   (`ReadTimeout` / `ConnectTimeout`), which is **not** an `HTTPError`, so it bypasses the
   handler entirely and propagates raw.
2. `_generate_guarded_answer` (`src/context/pipeline.py:184-194`) wraps `llm.complete` in a
   `try` that catches **only** `SchemaUnsupportedError`.

The exception therefore escapes `answer_with_retrieval` and is caught by the generic handler
at `src/internal/servers/web/app.py:1746-1754`, which raises
`HTTPException(status_code=502, detail=str(exc))`.

Two consequences:

- The caller gets a hard failure where the system had every ingredient needed to degrade
  gracefully: retrieval evidence was already gathered, and the abstention machinery already
  exists.
- `detail=str(exc)` leaks the provider endpoint URL from the `requests` exception message.

`multi_llm.py` does normalize `Timeout` into an `LLMTimeoutError` (`src/internal/llm/multi_llm.py:115, 756-757`),
but that class is **never caught anywhere in `src/`**, and `LitellmLLM` exposes no `complete()`,
so it cannot satisfy the `LLMClient` protocol the RAG pipeline requires. That normalization is
decorative and does not affect this path.

## Chosen Approach

Mirror the existing `SchemaUnsupportedError` contract: the provider raises a typed error, and
the pipeline decides policy.

1. `providers.py` catches `requests.Timeout` and raises `LLMTimeoutError`.
2. `_generate_guarded_answer` catches `LLMTimeoutError` and returns a degraded result
   immediately.
3. `app.py` never sees an exception, so no `502` and no leaked URL.

This introduces no new architecture. The dependency arrow is already
`internal/llm → context`: `providers.py:23` imports `SchemaUnsupportedError` from
`src.context.structured_output`. The new error follows the same direction.

### Rejected alternative: reuse `CANONICAL_ABSTENTION`

`CANONICAL_ABSTENTION` is `"I don't know based on the available evidence."` — a claim *about
the evidence*. A timeout is not an evidence problem: the evidence may be perfectly good and
the model call simply failed. Returning that sentence for a timeout tells the user something
false, and makes an LLM outage indistinguishable from a normal low-confidence answer in the
user-facing response, which is how outages hide.

### Rejected alternative: keep the 502 and sanitize only the detail

Replacing `detail=str(exc)` with a generic message would close the URL leak and keep the
error loud and monitorable, but the caller still gets a hard failure when a usable degraded
response was available. That does not address the original defect.

### Rejected alternative: retry the timeout

`_generate_guarded_answer` has a retry loop capped at two attempts. Consuming a retry on a
timeout doubles worst-case latency (30s → 60s) on a path that has just demonstrated it is
slow. The retry budget exists for output-quality failures — unparseable or incomplete
output — not transport failures. A timeout also usually signals sustained overload rather
than a transient blip, so the retry would likely time out too. Return on the first timeout,
matching the existing `refused` path.

## Contracts

### `LLMTimeoutError`

Add to `src/context/models.py`, which is the home of the `LLMClient` protocol that the RAG
pipeline requires:

```python
class LLMTimeoutError(RuntimeError):
    """The LLM call exceeded its timeout."""
```

`src/context/models.py` is chosen over `src/context/structured_output.py` (where
`SchemaUnsupportedError` lives) because a timeout is a property of the client contract, not of
structured output.

Re-export from `src/context/__init__.py` alongside `SchemaUnsupportedError`.

### `TIMEOUT_DEGRADED_ANSWER`

Add to `src/context/safety.py`, immediately beside `CANONICAL_ABSTENTION` (`safety.py:19`):

```python
TIMEOUT_DEGRADED_ANSWER = "I couldn't complete an answer in time. Please try again."
```

It must not equal `CANONICAL_ABSTENTION`. The two are permanently distinct: one reports a
conclusion about evidence, the other reports a system failure.

### `OpenAICompatibleLLM.complete` (`src/internal/llm/providers.py`)

Catch `requests.Timeout` around the existing post and raise `LLMTimeoutError`. The message
must be generic and must not interpolate the exception text or the endpoint, since a caller
may surface it.

### The `requests` exception hierarchy, verified

The whole diagnosis and scope boundary rest on this hierarchy, so it was checked empirically
rather than assumed:

| Relationship | Result |
| --- | --- |
| `Timeout` is a subclass of `HTTPError` | **False** — this is the bug |
| `ReadTimeout` is a `Timeout` | True |
| `ConnectTimeout` is a `Timeout` | True |
| `ConnectionError` is a `Timeout` | **False** — the scope boundary holds |
| `ConnectTimeout` is a `ConnectionError` | **True** — dual inheritance |

Three consequences:

- `Timeout` and `HTTPError` are **siblings** under `RequestException`; neither catches the
  other. That is precisely why a timeout escapes today's `except requests.HTTPError`. It also
  means the two `except` clauses are independent and their order is not load-bearing — stated
  so a future reader does not assume otherwise.
- `except requests.Timeout` covers both `ReadTimeout` and `ConnectTimeout`, which is the
  intent.
- `ConnectTimeout` inherits from **both** `Timeout` and `ConnectionError`. So "connection
  errors are out of scope" is true only of *plain* `ConnectionError` (host down, connection
  refused). A `ConnectTimeout` is in scope and will be handled, because it is a timeout. Tests
  must use plain `requests.ConnectionError` to pin the boundary; using `ConnectTimeout` would
  assert the opposite of what is intended and fail.

### `_generate_guarded_answer` (`src/context/pipeline.py`)

Catch `LLMTimeoutError` and return the existing 8-tuple immediately:

| Field | Value |
| --- | --- |
| answer | `TIMEOUT_DEGRADED_ANSWER` |
| confidence | `0.0` |
| verification_status | `VerificationStatus.ABSTAINED` |
| retry_count | the current `attempt` |
| structured_output_requested | `requested` (preserved) |
| structured_output_applied | `applied` (preserved) |
| structured_output_downgraded | `downgraded` (preserved) |
| structured_output_category | `"timeout"` |

This is the same shape the `refused` path returns (`pipeline.py:197-207`), with a different
answer string and category.

## Behavior

- The result is abstain-*shaped*, so it flows through every existing safe path unchanged:
  `abstained` is derived from `verification_status` at `pipeline.py:120`, and no caller needs
  to learn a new state.
- `structured_output_category="timeout"` flows to the existing trace span
  (`pipeline.py:331-346`). No new trace attribute.
- The MCP chat adapter already surfaces `confidence`, `verification_status`, and `abstained`
  (`src/internal/mcp_server/tools/chat.py:172-177`), so it reports the degradation with no
  change.
- The web API drops that metadata before responding (`app.py:1737-1766`) — a known, separate
  gap — but now returns `200` with a usable answer instead of `502`.
- Citations and sources are empty, exactly as on the other abstention paths.

## Testing

Test-driven, RED before GREEN:

1. A session whose post raises `requests.ReadTimeout` → `complete()` raises `LLMTimeoutError`.
2. The raised `LLMTimeoutError` message contains neither the endpoint URL nor the original
   exception text — the leak must not simply move.
3. `_generate_guarded_answer` with a timing-out client → `TIMEOUT_DEGRADED_ANSWER`,
   `confidence == 0.0`, `VerificationStatus.ABSTAINED`, category `"timeout"`.
4. It does **not** retry: `llm.complete` is called exactly once.
5. `TIMEOUT_DEGRADED_ANSWER != CANONICAL_ABSTENTION`, and a timeout result is distinguishable
   from an evidence abstention by both answer and category.
6. Pipeline level: `answer_with_retrieval` with a timing-out client returns a result rather
   than raising.
7. A non-timeout exception still propagates — e.g. `requests.ConnectionError` from the same
   session still escapes `complete()`. This locks in the scope decision and proves the new
   handler does not over-catch.
8. `requests.HTTPError` handling is unchanged: the existing JSON-Schema-unsupported `400`
   still raises `SchemaUnsupportedError`.

## Out of Scope

Each is a separate finding; none is addressed here:

- **Plain `requests.ConnectionError`** (LLM host down or refusing). It produces the identical
  `502`-with-leaked-URL today. Excluded deliberately: a connection error usually means
  misconfiguration, and degrading it to a polite answer would hide that. Test 7 pins this
  boundary. Note the exclusion does **not** extend to `ConnectTimeout`, which is both a
  `ConnectionError` and a `Timeout`, and is therefore handled — see the hierarchy table above.
- **The `grounded_generation.enabled=False` legacy path** (`pipeline.py:87`). It opts out of
  the grounding and abstention machinery entirely and returns raw text with no confidence or
  verification status, so it has no degraded result to return. A timeout there still raises.
- **`OpenAICompatibleLLM.stream`** (`providers.py:175`). A different method, not used by the
  RAG synthesis path. Note its `timeout` covers only connect and time-to-first-byte, not the
  `iter_lines` body loop.
- **The duplicate `LLMTimeoutError` in `src/internal/llm/multi_llm.py:115`.** After this
  change two distinct classes share that name. The duplicate is dead — never caught, on a path
  whose client cannot satisfy the `LLMClient` protocol — so unifying it means editing dead
  code. It is left alone deliberately, and recorded here as a known collision to resolve when
  that path is either revived or removed. A future `except LLMTimeoutError` must import from
  `src.context`.
- **A total deadline on the agent run.** No timeout of any kind bounds a whole request today;
  the loops cap iterations, not elapsed time, and `AgentLoopBase.generate_response_ids`
  (`src/agents/core/base.py:230-247`) has no timeout at all. That is a much larger design
  change.
