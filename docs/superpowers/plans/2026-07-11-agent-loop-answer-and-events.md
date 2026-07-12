# Plan: Agent-loop final answer + queue event delivery + RAG follow-up offload

Date: 2026-07-11
Spec: docs/superpowers/specs/2026-07-11-agent-loop-answer-and-events-design.md

## Steps

1. **A1 — record answer before caps** (`src/agents/tool/tool_calling.py`)
   - In `run()`, move the tool-call parse + `working_messages.append` +
     `final_answer = assistant_content` above the three stopping checks.
   - Keep `if not tool_calls: break` after the stopping checks.
   - Verify: new tests assert `final_answer` non-None on cap exit; existing
     tool-approval tests still pass (happy path unchanged).

2. **A2 — yield terminal events** (`src/internal/chat/queue_manager.py`)
   - In `listen()`, replace `publish(...) + break` with `yield self._event(...) +
     break` for TIMEOUT and STOP. Leave PING untouched.
   - Verify: consumer receives STOP / TIMEOUT as the last yielded event.

3. **A3 — offload follow-up** (`src/agents/search/agentic_rag.py`)
   - Make `_generate_followup` async; wrap `self.llm.complete` in
     `asyncio.wait_for(asyncio.to_thread(...), timeout=self.config.sufficiency_timeout_s)`;
     fail-open `[]` on exception/timeout. Await it at the call site.
   - Verify: timeout test returns `[]` fast; success test parses queries.

4. **Regression tests** — add to `tests/unit/test_tool_approval.py`,
   `tests/unit/test_queue_manager.py`, `tests/unit/test_agentic_rag.py`.

5. **Lint + test** — `ruff check . --fix && ruff format .`; `pytest tests/unit -q`.

## Success criteria

- 3 new regression tests fail before, pass after.
- `pytest tests/unit` green; ruff clean.
- No changes outside the three source files + three test files + spec/plan.
