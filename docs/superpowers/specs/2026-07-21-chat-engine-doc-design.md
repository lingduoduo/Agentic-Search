# Move chat-agent material from the README into docs/chat-engine.md

**Date:** 2026-07-21
**Status:** Approved

## Problem

Following the search-engine extraction, the README still describes the chat agent
only through scattered "What it provides" bullets (conversation agents with
routing/memory; the React chat UI). `docs/chat-engine.md` exists but is empty, and
there is no single page describing the chat agent.

## Goal

Populate `docs/chat-engine.md` as a standalone overview of the chat agent — its
capabilities and how requests route into it — and add a short pointer section plus
docs-list entry in the README, mirroring the search-engine change. No product
behavior changes.

## Design

1. **New `docs/chat-engine.md`** with a README back-link:
   - *Capabilities* — grounded conversation via `chat_once` / `chat_loop`
     (`AgenticRAGLoop`), multi-turn memory, and the React chat UI.
   - *Routing into chat* — conversational/generative requests route to `chat`
     (grounded `AgenticRAGLoop`); `chat` is also the `tool` fallback; requires an
     LLM client. Links to `docs/request-routing.md` for the full decision order.
   - Cross-links to `docs/request-routing.md` and `docs/frontend.md`.
2. **README edits:**
   - Add a `## Chat engine` pointer section after `## Search engine`.
   - Add `docs/chat-engine.md` to the Documentation list.
   - Keep the short capability bullets in place.

No code, API, or schema changes.
