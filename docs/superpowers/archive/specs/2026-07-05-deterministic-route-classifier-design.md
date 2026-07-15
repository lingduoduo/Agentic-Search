# Deterministic route classifier — design

## Problem

Running the same query from the web UI repeatedly returns answers from
*different* sources run-to-run — sometimes local vector/TF-IDF retrieval,
sometimes SerpAPI web search, sometimes the local LLM. Users read this as
"broken": identical input, non-reproducible behavior.

## Root cause

The `/api/agent` auto-routing path first classifies each query into one of
`chat` / `search` / `tool` via `classify_route`
(`src/internal/servers/web/intent_routing.py`). That classification is a single
LLM completion:

```python
response = llm.complete([ChatMessage(role="user", content=prompt)])
```

`OpenAICompatibleLLM.complete` (`src/internal/llm/providers.py`) never sends a
`temperature` field, so the server applies its **default (~1.0)**. The label
therefore varies across identical requests.

Because each strategy resolves to a *different* backend — and degrades
differently when a local model / LLM is absent (`app.py:_run_auto_routed`) — a
single classifier flip surfaces as a different answer *source*:

| Classifier label | With local model | Degraded |
| --- | --- | --- |
| `search` | SearchAgentLoop → local retrieval + local LLM | `_auto_search_pipeline` → SerpAPI fan-out |
| `chat`   | AgenticRAGLoop → local retrieval (RAG) | SerpAPI fan-out |

The search/chat *loops themselves* already decode at `temperature=0.0`; only the
entry-point classifier (and utility `.complete` calls) were left unpinned. Bare
term lookups (e.g. `FAISS`) are already deterministic — they skip the classifier
via `_is_bare_lookup` — which is why only some phrasings felt unstable.

## Fix

Make the strategy classification deterministic:

1. `OpenAICompatibleLLM.complete` forwards a `temperature` kwarg into the request
   body when provided (mirrors the existing `max_tokens` passthrough).
2. `classify_route` calls `llm.complete(..., temperature=0.0)`.

Same query → same strategy → same source, run-to-run.

## Scope / non-goals

- **In scope:** deterministic *routing* only.
- **Out of scope:** pinning answer-generation temperature (answer wording may
  still vary slightly); forcing a single fixed source; cross-restart model-load
  flapping. These were considered and explicitly deferred per user decision
  ("deterministic routing").

## Verification

- Unit: `complete()` forwards `temperature`; `classify_route` passes
  `temperature=0.0`.
- Regression: web-server, routing, LLM-provider, and execution-fallback suites
  stay green.
