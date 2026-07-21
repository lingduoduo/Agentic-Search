# Chat engine

[← Back to README](../README.md)

This guide covers the chat agent: conversational, retrieval-grounded answering.
For the authoritative deep dives, see [API request routing](request-routing.md)
and [Frontend development](frontend.md).

## Capabilities

- **Grounded conversation** — retrieval-grounded synthesis with citations, over
  either a single answer call (`chat_once`) or the iterative `AgenticRAGLoop`
  (`chat_loop`: query decomposition, HyDE, iterative retrieval, then synthesis).
- **Multi-turn memory** — bounded session history resolves follow-ups so a
  conversation carries context across turns.
- **React chat UI** — streaming responses, source inspection, and observability
  surfaces in the development frontend.

## Routing into chat

With `mode` omitted, `/api/agent` classifies each request as `chat`, `search`, or
`tool`. Conversational or generative requests route to `chat`, which runs the
grounded `AgenticRAGLoop`; `chat` is also the fallback when a `tool` route has no
usable result. Chat requires an LLM client for synthesis. See
[API request routing](request-routing.md) for the full decision order, explicit
`chat_once` / `chat_loop` modes, and response metadata.

## Direct chat surface (`/chat/send-chat-message`)

Beyond the `/api/agent` auto-router, chat has a direct endpoint that calls the
local model with no retrieval and no tools:

- `POST /chat/send-chat-message` — runs `PlainGenerationLoop` over the session
  history + the new message, streaming SSE `answer` then `done` (`error` on
  failure). Requires a local model (`SEARCH_AGENT_MODEL` /
  `SEARCH_AGENT_SERVER_URL`); returns **400** otherwise. `stream:false` returns
  one JSON `{ session_id, answer }`. The runner is
  `src/internal/servers/web/plain_chat_runner.py`.

In the web UI, the **Chat** tab drives this endpoint and renders a running
transcript of the session's turns (accumulated client-side as you chat).
