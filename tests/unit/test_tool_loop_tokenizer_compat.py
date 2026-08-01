"""ToolAgentLoop must accept a tokenizer that returns a mapping, not a list.

transformers 5.x changed `apply_chat_template(..., tokenize=True)` to return a
`BatchEncoding` (a Mapping of "input_ids"/"attention_mask") instead of a flat
list of ints. Iterating that mapping yields its *keys*, so the loop fed
`["input_ids", "attention_mask"]` into `torch.tensor(...)` and the backend died
with `ValueError: too many dimensions 'str'` before generating a single token.

The other loops are unaffected: they call the template with `tokenize=False`
and encode the resulting string themselves.
"""

from __future__ import annotations

import asyncio

from src.agents import ToolAgentLoop, ToolAgentLoopConfig


class _BatchEncoding(dict):
    """Stand-in for transformers' BatchEncoding: a Mapping keyed by input_ids."""


class _MappingTokenizer:
    """A transformers 5.x style tokenizer."""

    chat_template = ""

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(item) for item in ids)

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=True
    ):
        text = "\n".join(message.get("content", "") for message in messages)
        if not tokenize:
            return text
        ids = self.encode(text)
        return _BatchEncoding(input_ids=ids, attention_mask=[1] * len(ids))


class _Manager:
    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = iter(responses)
        self.prompts_seen = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        # The real backend does torch.tensor(prompt_ids) here, which is where a
        # str element blows up. Assert the contract at the boundary instead.
        self.prompts_seen.append(prompt_ids)
        assert all(isinstance(item, int) for item in prompt_ids), (
            f"prompt_ids must be ints, got {prompt_ids[:3]!r}"
        )
        return self.tokenizer.encode(next(self.responses))


def _loop(responses):
    tokenizer = _MappingTokenizer()
    manager = _Manager(tokenizer, responses)
    loop = ToolAgentLoop(
        tokenizer, manager, [], ToolAgentLoopConfig(response_length=4096)
    )
    return loop, manager


def test_prompt_ids_are_ints_when_the_template_returns_a_mapping():
    loop, _ = _loop(["done"])

    prompt_ids = loop._build_prompt_ids_with_tools_sync(
        [{"role": "user", "content": "hi"}]
    )

    assert prompt_ids == [ord(char) for char in "hi"]


def test_template_prefix_is_measured_in_tokens_not_mapping_keys():
    """len() of a BatchEncoding is its key count (2), which silently mis-strips."""
    loop, _ = _loop(["done"])

    assert loop._template_prefix_len == 0


def test_the_loop_runs_end_to_end_with_a_mapping_tokenizer():
    loop, manager = _loop(["done"])

    output = asyncio.run(loop.run([{"role": "user", "content": "hi"}], {}))

    assert output.final_answer == "done"
    assert manager.prompts_seen, "the model backend was never reached"
