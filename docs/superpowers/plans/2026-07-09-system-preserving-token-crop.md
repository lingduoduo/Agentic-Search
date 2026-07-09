# System-Preserving Token Crop Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Preserve the system prompt when `_build_prompt_ids_sync` truncates a long conversation to `prompt_length`.

**Architecture:** A pure `_crop_prompt_ids` helper + a `_encode_system_prefix` helper in `src/agents/core/base.py`, wired into both truncation paths of `_build_prompt_ids_sync`.

**Tech Stack:** Python.

## Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/system-preserving-token-crop`.
- Under-budget output must stay byte-identical to today (zero regression for normal prompts).
- Change only `base.py`; no per-loop changes, no config flag.
- Match repo ruff formatting.

---

### Task 1: `_crop_prompt_ids` helper + wiring + tests

**Files:**
- Modify: `src/agents/core/base.py`
- Test: `tests/unit/test_prompt_crop.py`

**Interfaces:**
- Produces: module-level `_crop_prompt_ids(full_ids: list[int], system_ids: list[int], budget: int) -> list[int]`; method `AgentLoopBase._encode_system_prefix(messages) -> list[int]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prompt_crop.py`:

```python
"""Unit tests for system-preserving prompt-id truncation."""

from __future__ import annotations

from src.agents.core.base import AgentLoopBase, AgentLoopConfig, _crop_prompt_ids
from tests.unit.test_agent_loop import DummyServerManager, DummyTokenizerWithEncode


def test_under_budget_unchanged():
    full = [1, 2, 3]
    assert _crop_prompt_ids(full, [9], 10) == full
    assert _crop_prompt_ids(full, [9], 0) == full  # budget <= 0 → unchanged


def test_no_system_tail_crop():
    full = list(range(10))
    assert _crop_prompt_ids(full, [], 4) == [6, 7, 8, 9]


def test_system_preserved_over_budget():
    system = [100, 101]
    full = list(range(20))  # far over budget
    out = _crop_prompt_ids(full, system, 6)
    assert len(out) == 6
    assert out[:2] == system
    assert out[2:] == full[-(6 - 2):]  # recent tail fills the rest


def test_system_larger_than_budget_degenerate():
    system = [1, 2, 3, 4, 5]
    full = list(range(50))
    out = _crop_prompt_ids(full, system, 3)
    assert out == system[-3:]


def test_build_prompt_ids_sync_keeps_system_prefix():
    loop = AgentLoopBase(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=40),
    )
    system_content = "SYSTEM RULES"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "u" * 200},  # forces the crop
    ]
    ids = loop._build_prompt_ids_sync(messages)
    assert len(ids) == 40
    assert ids[: len(system_content)] == [ord(c) for c in system_content]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_prompt_crop.py -v`
Expected: FAIL — `ImportError` for `_crop_prompt_ids`.

- [ ] **Step 3: Add the helper + system-prefix encoder**

In `src/agents/core/base.py`, add a module-level helper (near the top, after the
regex constants):

```python
def _crop_prompt_ids(
    full_ids: list[int], system_ids: list[int], budget: int
) -> list[int]:
    """Truncate `full_ids` to `budget` tokens, preserving the system prefix.

    Under budget → returned unchanged. Over budget → keep `system_ids` at the
    front and fill the remaining budget with the tail of `full_ids` (the most
    recent tokens, ending with the generation cue).
    """
    if budget <= 0 or len(full_ids) <= budget:
        return full_ids
    if not system_ids:
        return full_ids[-budget:]
    if len(system_ids) >= budget:
        return system_ids[-budget:]
    return system_ids + full_ids[-(budget - len(system_ids)):]
```

Add a method to `AgentLoopBase` (near `_build_prompt_ids_sync`):

```python
    def _encode_system_prefix(self, messages: list[dict[str, Any]]) -> list[int]:
        """Token ids of the leading system message, or [] if there is none."""
        if not messages or messages[0].get("role") != "system":
            return []
        system_msg = messages[0]
        chat_template = getattr(self.tokenizer, "chat_template", "__missing__")
        if hasattr(self.tokenizer, "apply_chat_template") and chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                [system_msg], add_generation_prompt=False, tokenize=False
            )
        else:
            text = system_msg.get("content", "")
        return list(self.tokenizer.encode(text))
```

- [ ] **Step 4: Wire both truncation paths**

Rewrite `_build_prompt_ids_sync` to use the helpers:

```python
    def _build_prompt_ids_sync(self, messages: list[dict[str, Any]]) -> list[int]:
        system_ids = self._encode_system_prefix(messages)
        chat_template = getattr(self.tokenizer, "chat_template", "__missing__")
        if hasattr(self.tokenizer, "apply_chat_template") and chat_template is not None:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = list(self.tokenizer.encode(prompt_text))
            return _crop_prompt_ids(prompt_ids, system_ids, self.prompt_length)

        joined = "\n".join(message.get("content", "") for message in messages)
        if hasattr(self.tokenizer, "encode"):
            prompt_ids = list(self.tokenizer.encode(joined))
            return _crop_prompt_ids(prompt_ids, system_ids, self.prompt_length)
        raise TypeError(
            "tokenizer must implement apply_chat_template(...) or encode(...)."
        )
```

- [ ] **Step 5: Run new tests + regression**

Run: `python3 -m pytest tests/unit/test_prompt_crop.py tests/unit/test_agent_loop.py -q`
Expected: PASS — new tests green; existing loop tests unchanged (under-budget prompts are byte-identical, and existing tests use short prompts well under 4096).

- [ ] **Step 6: Commit**

```bash
git add src/agents/core/base.py tests/unit/test_prompt_crop.py
git commit -m "feat(agents): preserve system prompt when cropping to prompt_length

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Pure `_crop_prompt_ids` (spec §Approach) → Step 3 + 4 helper tests. ✓
- `_encode_system_prefix` (both render paths) → Step 3. ✓
- Both truncation paths wired (spec §Approach) → Step 4. ✓
- Under-budget byte-identical + `budget<=0` (spec success criteria) → `test_under_budget_unchanged`. ✓
- No-system tail-crop unchanged → `test_no_system_tail_crop`. ✓
- Integration keeps system prefix → `test_build_prompt_ids_sync_keeps_system_prefix`. ✓
- Types consistent: `_crop_prompt_ids(list[int], list[int], int) -> list[int]` across def, tests, and both call sites. ✓
