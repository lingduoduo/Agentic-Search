"""Build an SFT example from a deterministic SearchAgentLoop trace."""

from __future__ import annotations

import asyncio

from src.agent_loop import build_search_sft_example

from .run_search_trace_workflow import QUESTION, run_workflow_demo


async def run_demo():
    output, _, _ = await run_workflow_demo()
    return build_search_sft_example(
        [{"role": "user", "content": QUESTION}],
        output,
    )


def main() -> None:
    example = asyncio.run(run_demo())
    print("Prompt:", example.prompt_messages)
    print("\nCompletion:")
    print(example.completion)


if __name__ == "__main__":
    main()
