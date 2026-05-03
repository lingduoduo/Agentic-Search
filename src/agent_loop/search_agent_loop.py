"""Multi-turn agentic search loop using <search>/<answer> XML tags."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .agent_loop import AgentLoopBase, AgentLoopConfig, AgentLoopOutput, register, simple_timer
from .context import AgentContext, SearchResult
from .search_client import SearchClient, SearchClientConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant that answers questions by searching for information.\n"
    "To search, write: <search>your query here</search>\n"
    "To give your final answer, write: <answer>your answer here</answer>\n"
    "Always search before answering if you need external knowledge."
)


@dataclass(frozen=True)
class SearchAgentLoopConfig(AgentLoopConfig):
    """Extends AgentLoopConfig with search-specific settings."""

    max_turns: int = 5
    search_url: str = "http://localhost:8000/retrieve"
    topk: int = 5
    search_timeout_seconds: int = 10
    search_max_retries: int = 3
    search_tag: str = "search"
    answer_tag: str = "answer"
    # Template for the observation injected after each search.
    # {content} is replaced by SearchContext.to_information_block().
    obs_template: str = "\n\n<information>\n{content}\n</information>\n\n"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@register("search_agent")
class SearchAgentLoop(AgentLoopBase):
    """Multi-turn agent loop that issues `<search>` queries and accumulates context.

    Each turn:
      1. Generates a response.
      2. Parses the first <search>...</search> or <answer>...</answer> tag.
      3. If <search>: calls the configured retrieval endpoint, injects
         <information>...</information> into the conversation, and continues.
      4. If <answer> (or no tag): stops and returns.

    The returned AgentLoopOutput.context is an AgentContext holding all search
    turns for downstream inspection or reward computation.
    """

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        search_config: SearchAgentLoopConfig | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        cfg = search_config or SearchAgentLoopConfig()
        super().__init__(
            tokenizer=tokenizer,
            server_manager=server_manager,
            config=AgentLoopConfig(
                prompt_length=cfg.prompt_length,
                response_length=cfg.response_length,
            ),
            loop=loop,
        )
        self.search_config = cfg
        self._action_re = re.compile(
            rf"<({re.escape(cfg.search_tag)}|{re.escape(cfg.answer_tag)})>"
            rf"(.*?)</\1>",
            re.DOTALL,
        )
        self._search_client = SearchClient(
            SearchClientConfig(
                url=cfg.search_url,
                topk=cfg.topk,
                timeout_seconds=cfg.search_timeout_seconds,
                max_retries=cfg.search_max_retries,
            )
        )

    def _parse_action(self, text: str) -> tuple[str | None, str]:
        """Return (tag_name, inner_content) or (None, '') if no tag matched."""
        m = self._action_re.search(text)
        if m:
            return m.group(1), m.group(2).strip()
        return None, ""

    def _retrieve_sync(self, query: str) -> list[SearchResult]:
        try:
            return self._search_client.retrieve_one(query)
        except Exception as exc:
            logger.warning("Search failed for query %r: %s", query, exc)
            return []

    async def _retrieve(self, query: str) -> list[SearchResult]:
        """Run the blocking HTTP call off the event loop."""
        loop = await self.get_loop()
        return await loop.run_in_executor(None, self._retrieve_sync, query)

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        request_id = uuid4().hex
        agent_ctx = AgentContext()

        # Prepend system prompt if not already present
        working_messages = list(messages)
        if self.search_config.system_prompt and (
            not working_messages or working_messages[0].get("role") != "system"
        ):
            working_messages = [
                {"role": "system", "content": self.search_config.system_prompt},
                *working_messages,
            ]

        all_response_ids: list[int] = []
        final_prompt_ids: list[int] = []
        num_turns = 0

        for turn in range(self.search_config.max_turns):
            with simple_timer(f"build_prompt_turn_{turn}", metrics):
                prompt_ids = await self.build_prompt_ids(working_messages)
            final_prompt_ids = prompt_ids

            with simple_timer(f"generate_turn_{turn}", metrics):
                response_ids = await self.generate_response_ids(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}_t{turn}",
                )

            all_response_ids.extend(response_ids)
            num_turns += 1

            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            action, content = self._parse_action(response_text)

            logger.debug("turn=%d action=%s content=%r", turn, action, content[:80] if content else "")

            # Any non-search action (answer or unrecognised) ends the loop
            if action != self.search_config.search_tag:
                break

            # --- search turn ---
            t0 = time.perf_counter()
            results = await self._retrieve(content)
            elapsed = time.perf_counter() - t0
            logger.debug("search returned %d results in %.2fs", len(results), elapsed)

            search_ctx = agent_ctx.add_turn(query=content, results=results)
            obs = self.search_config.obs_template.format(
                content=search_ctx.to_information_block()
            )

            working_messages.append({"role": "assistant", "content": response_text})
            working_messages.append({"role": "user", "content": obs})

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=all_response_ids,
            response_mask=self.build_response_mask(all_response_ids),
            num_turns=num_turns,
            metrics=metrics,
            request_id=request_id,
            context=agent_ctx,
        )
