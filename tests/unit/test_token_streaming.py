"""Token-level streaming: first token must arrive long before the answer.

The assist stream used to emit the answer as a single SSE event *after* the whole
agent run finished, so time-to-first-token equalled time-to-completion and
sub-second perceived latency was unreachable by construction. These tests pin the
seam that fixes it.

Scope is the single-generation routes -- `PlainGenerationLoop` and the RAG answer
path -- where tokens map one-to-one onto user-visible text. `SearchAgentLoop` is
deliberately excluded: it emits an XML protocol (`<plan>`, `<searches>`,
`<answer>`), so streaming its raw tokens would leak the protocol and the model's
intermediate reasoning to the user. That needs an incremental `<answer>` boundary
detector and is a separate change.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from src.agents.generation.plain import PlainGenerationLoop


class _StreamingManager:
    """A manager that can stream, yielding one known token at a time."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.stream_calls = 0
        self.generate_calls = 0

    async def generate(
        self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]
    ) -> list[int]:
        self.generate_calls += 1
        return [1, 2, 3]

    async def generate_stream(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        on_token,
    ) -> list[int]:
        self.stream_calls += 1
        for tok in self._tokens:
            await on_token(tok)
        return [1, 2, 3]


class _LegacyManager:
    """A manager with no ``generate_stream`` -- must still work, unchanged.

    This is the compatibility case that decided the design: `ServerManager` is a
    Protocol, and adding a required kwarg to `generate` would break every
    existing implementation and test double. A separate optional method keeps
    them working.
    """

    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(
        self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]
    ) -> list[int]:
        self.generate_calls += 1
        return [4, 5, 6]


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, **kw):
        return [ord(c) % 50 + 1 for c in str(text)][:64]

    def decode(self, ids, **kw):
        return "".join(chr((i % 26) + 97) for i in ids)

    def apply_chat_template(self, messages, **kw):
        joined = " ".join(str(m.get("content", "")) for m in messages)
        return self.encode(joined)


def _loop(manager):
    return PlainGenerationLoop(server_manager=manager, tokenizer=_Tokenizer())


@pytest.mark.asyncio
async def test_tokens_arrive_before_the_final_answer():
    """The point of the whole change: first token precedes completion."""
    order: list[str] = []
    manager = _StreamingManager(["Hel", "lo", " wor", "ld"])

    async def on_token(text: str) -> None:
        order.append(f"token:{text}")

    loop = _loop(manager)
    output = await loop.run(
        [{"role": "user", "content": "hi"}],
        {"max_tokens": 8},
        on_token=on_token,
    )
    order.append("answer")

    assert manager.stream_calls == 1, "streaming manager was not used"
    assert manager.generate_calls == 0, "fell back despite generate_stream existing"
    assert order[0] == "token:Hel", "no token arrived before the answer"
    assert order.index("token:Hel") < order.index("answer")
    assert [o for o in order if o.startswith("token:")] == [
        "token:Hel",
        "token:lo",
        "token: wor",
        "token:ld",
    ]
    assert output.final_answer is not None


@pytest.mark.asyncio
async def test_without_a_callback_the_old_path_is_used_unchanged():
    """`on_token=None` must reproduce today's behaviour exactly.

    Every trainer and every offline script calls this loop with no callback;
    none of them should start paying for a streaming path they do not consume.
    """
    manager = _StreamingManager(["a", "b"])
    loop = _loop(manager)

    output = await loop.run([{"role": "user", "content": "hi"}], {"max_tokens": 4})

    assert manager.stream_calls == 0, "streamed despite no callback"
    assert manager.generate_calls == 1
    assert output.response_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_manager_without_generate_stream_still_works():
    """Falls back rather than raising AttributeError."""
    manager = _LegacyManager()
    seen: list[str] = []

    async def on_token(text: str) -> None:
        seen.append(text)

    loop = _loop(manager)
    output = await loop.run(
        [{"role": "user", "content": "hi"}],
        {"max_tokens": 4},
        on_token=on_token,
    )

    assert manager.generate_calls == 1
    assert output.response_ids == [4, 5, 6]
    assert seen == [], "a non-streaming manager must not fabricate tokens"


@pytest.mark.asyncio
async def test_a_failing_token_callback_does_not_lose_the_answer():
    """A dead SSE client must not destroy an otherwise-complete generation.

    The callback is a delivery detail; the run has already done the expensive
    work by the time tokens flow.
    """
    manager = _StreamingManager(["x", "y"])

    async def on_token(text: str) -> None:
        raise RuntimeError("client disconnected")

    loop = _loop(manager)
    output = await loop.run(
        [{"role": "user", "content": "hi"}],
        {"max_tokens": 4},
        on_token=on_token,
    )

    assert output.response_ids == [1, 2, 3]
    assert output.final_answer is not None
