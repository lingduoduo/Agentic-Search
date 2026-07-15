# System-Preserving Token Crop — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/system-preserving-token-crop
Related: [[project_chat_orchestration]] (gap #2)

## Problem

`AgentLoopBase._build_prompt_ids_sync` (`src/agents/core/base.py:177`) renders the
message list through the chat template, encodes it, and keeps the **last**
`prompt_length` tokens: `list(prompt_ids)[-self.prompt_length:]` (:186, and again
in the fallback join path at :190). Because it keeps the *tail*, a conversation
that exceeds `prompt_length` (default 4096) tokens silently drops the **front** —
which is the system prompt. Every loop routes prompt construction through this
method, so the agent can lose its instructions on long chats.

Goal: when truncating, preserve the system prompt.

## Non-goals

- No message-level trimming / re-render loop (approach B) — higher blast radius
  on a shared method; deferred.
- No per-loop changes; no config flag (under-budget output is unchanged, over-budget
  strictly improves).
- No change to the tokenizer or chat template.

## Approach (A — token-level, system-preserving)

A pure helper decides the crop; `_build_prompt_ids_sync` supplies the token lists.

```python
def _crop_prompt_ids(full_ids: list[int], system_ids: list[int], budget: int) -> list[int]:
    if budget <= 0 or len(full_ids) <= budget:
        return full_ids                         # under budget → unchanged
    if not system_ids:
        return full_ids[-budget:]               # no system → current tail-crop
    if len(system_ids) >= budget:
        return system_ids[-budget:]             # degenerate: system alone overflows
    return system_ids + full_ids[-(budget - len(system_ids)):]
```

`_build_prompt_ids_sync`:
1. Compute `system_ids` = `encode` of the leading system message's raw `content`,
   or `[]` if `messages` is empty or `messages[0]["role"] != "system"`. (Raw
   content, not a lone chat-template render: simpler, avoids template-specific
   behavior when a system message is rendered alone, and still preserves the
   instruction text under an over-budget crop.)
2. Render `full_ids` as today (`add_generation_prompt=True`).
3. Return `_crop_prompt_ids(full_ids, system_ids, self.prompt_length)`.

Applied to **both** paths (chat-template and fallback join).

### Why A

- **Under budget → byte-identical to today** (the helper's first branch returns
  `full_ids` unchanged) — zero regression for normal-length prompts.
- **Over budget → system survives** + the most recent tokens (the tail already
  ends with the generation cue). The tail is mid-message, but the current code
  already cuts mid-token, so this is strictly better (it adds the system prefix).
- No duplication: when over budget, `full_ids[-(budget - len(system))]` are recent
  tokens, never the cropped-off system prefix.

## Success criteria

- `len(full) <= budget` → `full` unchanged (including `budget <= 0`).
- No system message → `full[-budget:]` (unchanged behavior).
- System present + over budget → result starts with `system_ids` and has length
  `budget`, and contains the tail of `full`.
- `len(system) >= budget` → `system[-budget:]`.
- Existing agent-loop / prompt tests stay green.

## Testing (no tokenizer for the core)

Unit tests on the pure `_crop_prompt_ids` (plain int lists):
1. under budget unchanged; `budget <= 0` unchanged.
2. no system → tail-crop.
3. system + over budget → `result[:len(system)] == system_ids`, `len(result) == budget`,
   and `result[len(system):] == full[-(budget-len(system)):]`.
4. system larger than budget → `system[-budget:]`.

One integration test: instantiate `AgentLoopBase` with a minimal dummy tokenizer
(fallback path, `encode(s) = list(s.encode())`, no `apply_chat_template`),
`prompt_length` tiny, a system message + long later messages; assert the encoded
system content is a prefix of `_build_prompt_ids_sync(...)`.

## Risks

- The mid-message tail cut can slightly garble the boundary between the system
  prefix and the tail — but the pre-existing behavior already cut mid-token, and
  the system instruction surviving is the higher-order correctness win.
