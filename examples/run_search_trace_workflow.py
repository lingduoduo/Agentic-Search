"""Screenshot-style SearchAgentLoop workflow example.

This module demonstrates the trace pattern:

    <think>...</think>
    <search>...</search>
    <information>...</information>
    ...
    <answer>...</answer>

It uses fake model and search backends so the workflow is deterministic and can
be tested without loading an LLM or running a retrieval server.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from typing import Any

from src.agent_loop import SearchResult, build_search_sft_example

from .run_search_agent_loop import run_search_agent_loop_example


QUESTION = "Who is older, Jed Hoyer or John William Henry II?"
GROUND_TRUTH = "John William Henry II"


class DemoTokenizer:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class ScriptedServerManager:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        del request_id, prompt_ids, sampling_params
        response = self.responses[self.index]
        self.index += 1
        return [ord(char) for char in response]


class ScriptedSearchClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def retrieve(self, queries: list[str], topk: int | None = None):
        del topk
        self.calls.append(list(queries))
        return [[_result_for_query(query)] for query in queries]

    async def fetch_urls(self, urls: list[str]):
        del urls
        return []

    async def aclose(self) -> None:
        return None


def _result_for_query(query: str) -> SearchResult:
    lowered = query.lower()
    if "john" in lowered:
        return SearchResult(
            contents=(
                '"John Henry"\nJohn William Henry II was born on September 13, '
                "1949 in Quincy, Illinois."
            )
        )
    if "birth" in lowered:
        return SearchResult(
            contents='"Jed Hoyer"\nJed Hoyer was born on December 7, 1973.'
        )
    return SearchResult(
        contents=(
            '"Jed Hoyer"\nJed Hoyer joined the Red Sox at 28 and later became '
            "a Major League Baseball executive."
        )
    )


def build_scripted_responses() -> list[str]:
    return [
        (
            "<think>I need to determine if Jed Hoyer or John William Henry II "
            "is older. I'll search for their ages.</think>\n"
            "<search>Jed Hoyer or John William Henry II</search>"
        ),
        (
            "<think>I found Jed Hoyer is a person. Now I need John William "
            "Henry II's age.</think>\n"
            "<search>John William Henry II</search>"
        ),
        (
            "<think>John William Henry II was born in 1949. Now I need Jed "
            "Hoyer's birth year.</think>\n"
            "<search>Jed Hoyer birth year</search>"
        ),
        (
            "<think>Jed Hoyer was born in 1973, so John William Henry II is "
            "older.</think>\n"
            "<answer>John William Henry II</answer>"
        ),
    ]


def format_search_trace(question: str, ground_truth: str, output: Any) -> str:
    """Render the agent trajectory in the compact screenshot-style format."""

    lines = [f"Question: {question}", "", f"Ground Truth: {ground_truth}", ""]
    for message in output.trajectory_messages:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if message.get("role") == "assistant":
            lines.extend(_extract_action_blocks(content))
        elif message.get("role") == "user" and "<information>" in content:
            info = _extract_tag(content, "information")
            if info:
                lines.append(f"<information>{_compact(info)}</information>")
    return "\n".join(lines)


def _extract_action_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for tag in ("think", "search", "answer"):
        for value in re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL):
            blocks.append(f"<{tag}>{_compact(value)}</{tag}>")
    return blocks


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _compact(text: str, limit: int = 220) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 4].rstrip() + " ..."


async def run_workflow_demo() -> tuple[Any, str, ScriptedSearchClient]:
    tokenizer = DemoTokenizer()
    search_client = ScriptedSearchClient()
    output = await run_search_agent_loop_example(
        tokenizer=tokenizer,
        server_manager=ScriptedServerManager(build_scripted_responses()),
        question=QUESTION,
        search_client=search_client,
        max_turns=6,
        max_search_limit=3,
        sampling_params={"temperature": 0.0, "max_tokens": 128},
    )
    return output, format_search_trace(QUESTION, GROUND_TRUTH, output), search_client


async def run_sft_demo():
    output, _, _ = await run_workflow_demo()
    return build_search_sft_example(
        [{"role": "user", "content": QUESTION}],
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic search trace workflow."
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Print the SFT example built from the deterministic trace.",
    )
    args = parser.parse_args()

    if args.sft:
        example = asyncio.run(run_sft_demo())
        print("Prompt:", example.prompt_messages)
        print("\nCompletion:")
        print(example.completion)
    else:
        output, trace, _ = asyncio.run(run_workflow_demo())
        print(trace)
        print("\nMetrics:", output.metrics)


if __name__ == "__main__":
    main()
