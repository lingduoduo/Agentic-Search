# Plan: chat-engine doc extraction

**Spec:** ../specs/2026-07-21-chat-engine-doc-design.md

1. Write `docs/chat-engine.md` (capabilities + routing-into-chat + cross-links) →
   verify: file renders, links resolve to existing docs.
2. Add a `## Chat engine` pointer section after `## Search engine` in the README →
   verify: `grep -n "chat-engine.md" README.md` shows the pointer.
3. Add `docs/chat-engine.md` to the README Documentation list → verify: entry
   present.
4. Sanity-check no dangling links → verify: `docs/request-routing.md` and
   `docs/frontend.md` exist on disk.
