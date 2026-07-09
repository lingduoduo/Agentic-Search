# Cap Full-Page Fetch Content — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/cap-full-page-fetch-content
Related: [[project_chat_orchestration]] (gap #3)

## Problem

`SearchAgentLoop._format_full_page_information` (`src/agents/search/search.py:661`)
inlines each fetched page's `contents` **verbatim, with no length cap**. Unlike
`ToolAgentLoop`, which caps tool responses at `max_tool_response_length = 2048`
(`tool_calling.py:174`), the `<fetch>` full-page path has no client-side size
guard — a single large fetched page can blow the response-token budget, and (via
the non-message-aware keep-tail token crop at `base.py:186`) push the system
prompt out of context.

Goal: cap the per-page fetched content that gets formatted into the `<full_page>`
observation, head-keeping the most relevant part.

## Non-goals

- No cap on search snippets (those are server-controlled via `topk`).
- No change to the base token-crop (`base.py:186`) — separate known gap.
- No change to `ToolAgentLoop`.
- No total/cross-page budget (per-page cap only — see decision below).

## Approach (all in `src/agents/search/search.py`)

1. **Config field** — add `max_full_page_chars: int = 4096` to `SearchAgentLoopConfig`
   (near `full_page_obs_template`). Full pages are the deliberate "deeper look"
   path, so allow more than a 2048 tool response but keep it bounded. `<= 0`
   disables the cap (escape hatch).

2. **Pure helper** — module-level `_truncate_page_content(text: str, limit: int) -> str`:
   - `limit <= 0` or `len(text) <= limit` → return `text` unchanged.
   - else → `text[:limit] + "…(truncated)"` (head-keep; a page's title/lead is
     usually the most relevant part).
   Pure → unit-testable with no model/loop.

3. **Apply** — in `_format_full_page_information`, replace the raw
   `sections.append(page.contents)` with
   `sections.append(_truncate_page_content(page.contents, cfg.max_full_page_chars))`.

### Decision: per-page cap (not total budget)

Cap each page's `contents` independently. This directly fixes "a large fetched
page blows the budget" and mirrors ToolAgentLoop's per-response cap. A total
budget across all pages fetched in one turn would be more robust when many pages
are fetched at once, but adds budget-division complexity; deferred (YAGNI).

## Success criteria

- A page whose `contents` exceeds `max_full_page_chars` is truncated to
  `limit` chars + a truncation marker in the `<full_page>` block.
- A page within the limit is unchanged.
- `max_full_page_chars <= 0` disables truncation.
- Existing search-agent tests stay green (default 4096 is large enough that
  existing small fixtures are unaffected).

## Testing (no model load)

Unit tests on the pure helper `_truncate_page_content`:
1. `len(text) <= limit` → unchanged.
2. `len(text) > limit` → `text[:limit]` + marker; result startswith the head and
   endswith the marker.
3. `limit <= 0` → unchanged (disabled).

Plus one formatter test: `_format_full_page_information` on a `SearchResult` with
oversized `contents` (built directly, no loop/model) truncates it, while a small
page is left intact. Construct the loop with a tiny `max_full_page_chars` to
exercise the cap deterministically.

## Risks

- Truncating mid-sentence could drop relevant later content; acceptable for a
  bounded observation, and `max_full_page_chars` is tunable / disableable.
