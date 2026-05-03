"""Multi-turn research loop using XML tags for planning, search, and synthesis."""

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
from .context import AgentContext, SearchContext, SearchResult
from .evaluation import SearchEvaluationConfig, SearchResultEvaluator
from .search_client import SearchClient, SearchClientConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant that answers questions by searching for information.\n"
    "First write a concise research plan using <plan>...</plan>.\n"
    "Then search using either <search>single query</search> or "
    "<searches>one query per line</searches>.\n"
    "Queries inside <searches> are executed in parallel.\n"
    "After reviewing the evidence, either refine the keywords and search again "
    "or finish with <answer>...</answer>.\n"
    "When you answer, cite the evidence labels provided in the information block."
)


@dataclass(frozen=True)
class SearchAgentLoopConfig(AgentLoopConfig):
    """Extends AgentLoopConfig with search-specific settings."""

    max_turns: int = 5
    search_url: str = "http://localhost:8000/retrieve"
    topk: int = 5
    search_timeout_seconds: int = 10
    search_max_retries: int = 3
    plan_tag: str = "plan"
    search_tag: str = "search"
    searches_tag: str = "searches"
    answer_tag: str = "answer"
    plan_obs_template: str = (
        "\n\n<plan_feedback>\n"
        "Plan recorded. Continue by issuing one or more search queries.\n"
        "</plan_feedback>\n\n"
    )
    evaluation_obs_template: str = "\n\n<search_evaluation>\n{content}\n</search_evaluation>\n\n"
    answer_rejection_template: str = (
        "\n\n<answer_feedback>\n"
        "{content}\n"
        "</answer_feedback>\n\n"
    )
    require_sufficient_evidence_before_answer: bool = True
    # Stop rejecting <answer> after this many consecutive rejections (avoids infinite loops).
    max_answer_rejections: int = 3
    evaluation_config: SearchEvaluationConfig = SearchEvaluationConfig()
    # Template for the observation injected after each search.
    # {content} is replaced by the formatted batch-search information block.
    obs_template: str = "\n\n<information>\n{content}\n</information>\n\n"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@register("search_agent")
class SearchAgentLoop(AgentLoopBase):
    """Multi-turn agent loop for research planning, search, and synthesis.

    Each turn:
      1. Generates a response.
      2. Parses the first action tag: <plan>, <search>, <searches>, or <answer>.
      3. If <plan>: injects a lightweight acknowledgement and continues.
      4. If <search> or <searches>: executes one or more retrieval requests,
         injects <information>...</information> into the conversation, and continues.
      5. If <answer> (or no recognised tag): stops and returns.

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
        self._result_evaluator = SearchResultEvaluator(cfg.evaluation_config)
        action_tags = [
            cfg.plan_tag,
            cfg.search_tag,
            cfg.searches_tag,
            cfg.answer_tag,
        ]
        self._action_re = re.compile(
            rf"<({'|'.join(re.escape(tag) for tag in action_tags)})>"
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

    def _parse_actions(self, text: str) -> list[tuple[str, str]]:
        """Return all (tag, content) pairs found in *text*, in document order."""
        return [(m.group(1), m.group(2).strip()) for m in self._action_re.finditer(text)]

    async def _retrieve_many(self, queries: list[str]) -> list[list[SearchResult]]:
        try:
            return await self._search_client.retrieve(queries)
        except Exception as exc:
            logger.warning("Search failed for queries %r: %s", queries, exc)
            return [[] for _ in queries]

    def _parse_queries(self, content: str, action: str | None) -> list[str]:
        if action == self.search_config.search_tag:
            return [content] if content else []

        query_tags = re.findall(r"<query>(.*?)</query>", content, re.DOTALL)
        if query_tags:
            return [query.strip() for query in query_tags if query.strip()]

        queries: list[str] = []
        for line in content.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
            if cleaned:
                queries.append(cleaned)
        return queries

    def _format_round_information(
        self,
        round_index: int,
        search_contexts: list[SearchContext],
    ) -> str:
        if not search_contexts:
            return f"Round {round_index}\nNo information available"

        sections = [f"Round {round_index}"]
        for query_index, search_ctx in enumerate(search_contexts, 1):
            citation_prefix = f"R{round_index}Q{query_index}D"
            sections.append(f"Query {query_index}: {search_ctx.query}")
            sections.append(
                search_ctx.to_information_block(citation_prefix=citation_prefix)
            )
        return "\n".join(sections)

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

        plan_tag = self.search_config.plan_tag
        search_tags = {self.search_config.search_tag, self.search_config.searches_tag}
        answer_tag = self.search_config.answer_tag
        latest_evaluation = None
        consecutive_rejections = 0

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
            actions = self._parse_actions(response_text)

            logger.debug("turn=%d actions=%r", turn, [(t, c[:60]) for t, c in actions])

            working_messages.append({"role": "assistant", "content": response_text})

            # When the model generates nothing useful, check whether we still need evidence.
            # If evidence is already sufficient (or no gating), stop cleanly.
            # If insufficient, re-prompt for searches rather than silently exiting.
            if not actions:
                needs_more = (
                    self.search_config.require_sufficient_evidence_before_answer
                    and (latest_evaluation is None or not latest_evaluation.is_sufficient)
                    and consecutive_rejections < self.search_config.max_answer_rejections
                )
                if needs_more:
                    consecutive_rejections += 1
                    feedback = (
                        "No action detected. Evidence is still insufficient. "
                        "Issue a <searches> block to gather more evidence before answering."
                    )
                    working_messages.append({
                        "role": "user",
                        "content": self.search_config.answer_rejection_template.format(
                            content=feedback
                        ),
                    })
                    continue
                break

            answer_requested = any(tag == answer_tag for tag, _ in actions)
            # Collect every query from every <search>/<searches> block in this response.
            # This handles models that emit <plan>…</plan><searches>…</searches> in one shot.
            all_queries: list[str] = []
            for tag, content in actions:
                if tag in search_tags:
                    all_queries.extend(self._parse_queries(content, tag))

            if answer_requested and not all_queries:
                evidence_is_sufficient = (
                    latest_evaluation is not None and latest_evaluation.is_sufficient
                )
                if (
                    not self.search_config.require_sufficient_evidence_before_answer
                    or evidence_is_sufficient
                    or consecutive_rejections >= self.search_config.max_answer_rejections
                ):
                    break

                if latest_evaluation is None:
                    feedback = (
                        "Do not answer yet. You have not completed a search round. "
                        "Search first, then answer once the evidence is sufficient."
                    )
                else:
                    feedback = (
                        "Do not answer yet. The latest search evaluation was insufficient. "
                        "Refine the keywords, search again, and only answer after the evidence is sufficient."
                    )
                consecutive_rejections += 1
                working_messages.append({
                    "role": "user",
                    "content": self.search_config.answer_rejection_template.format(
                        content=feedback
                    ),
                })
                continue

            if not all_queries:
                # Only a <plan> (no searches yet) — acknowledge and let the model continue.
                working_messages.append(
                    {"role": "user", "content": self.search_config.plan_obs_template}
                )
                continue

            # A search was issued — reset the rejection counter.
            consecutive_rejections = 0

            # --- parallel search round ---
            t0 = time.perf_counter()
            results_by_query = await self._retrieve_many(all_queries)
            elapsed = time.perf_counter() - t0
            total_results = sum(len(r) for r in results_by_query)
            logger.debug(
                "search returned %d total results across %d queries in %.2fs",
                total_results,
                len(all_queries),
                elapsed,
            )

            if not any(results_by_query):
                working_messages.append({
                    "role": "user",
                    "content": self.search_config.obs_template.format(
                        content="No valid queries were provided. Try again with at least one search query."
                    ),
                })
                continue

            search_contexts = agent_ctx.add_round(
                queries=all_queries, results_by_query=results_by_query
            )
            round_index = agent_ctx.num_rounds
            latest_evaluation = self._result_evaluator.evaluate_round(search_contexts)
            obs = (
                self.search_config.evaluation_obs_template.format(
                    content=latest_evaluation.to_feedback_block()
                )
                + self.search_config.obs_template.format(
                    content=self._format_round_information(round_index, search_contexts)
                )
            )
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
