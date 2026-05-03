"""Multi-turn research loop using XML tags for planning, search, and synthesis."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from .agent_loop import AgentLoopBase, AgentLoopConfig, AgentLoopOutput, register, simple_timer
from .context import AgentContext, SearchContext, SearchResult
from .evaluation import SearchEvaluationConfig, SearchResultEvaluator, SearchRoundEvaluation
from .search_client import SearchClient, SearchClientConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))

def build_search_agent_instruction(max_search_limit: int, max_url_fetch: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches and fetch webpage content.\n\n"
        "Tools:\n"
        "- To write a brief research plan: use <plan>...</plan>.\n"
        "- To perform a search: use <search>single query</search> or <searches>one query per line</searches>.\n"
        "- To fetch detailed content from specific URLs returned by search: use <fetch>url1, url2</fetch>.\n"
        "- To provide the final response: use <answer>...</answer>.\n\n"
        "System feedback:\n"
        "- Search results are returned inside <information>...</information>.\n"
        "- Search-quality judgments are returned inside <search_evaluation>...</search_evaluation>.\n"
        "- Full webpage content is returned inside <full_page>...</full_page>.\n\n"
        f"You may use at most {max_search_limit} loop turns.\n"
        f"You may fetch up to {max_url_fetch} URLs in one fetch request.\n\n"
        "Workflow:\n"
        "1. Plan.\n"
        "2. Search, preferably with parallel queries when useful.\n"
        "3. Review the returned evidence.\n"
        "4. If the evidence is weak, search again with refined keywords.\n"
        "5. If snippets are not enough, fetch the most promising URLs.\n"
        "6. Answer only after the evidence is sufficient.\n\n"
        "Cite the evidence labels from the information blocks when you answer."
    )


DEFAULT_SYSTEM_PROMPT = build_search_agent_instruction(max_search_limit=5, max_url_fetch=3)


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
    fetch_tag: str = "fetch"
    answer_tag: str = "answer"
    max_url_fetch: int = 3
    plan_obs_template: str = (
        "\n\n<plan_feedback>\n"
        "Plan recorded. Continue by issuing one or more search queries.\n"
        "</plan_feedback>\n\n"
    )
    evaluation_obs_template: str = "\n\n<search_evaluation>\n{content}\n</search_evaluation>\n\n"
    full_page_obs_template: str = "\n\n<full_page>\n{content}\n</full_page>\n\n"
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
    system_prompt: str | None = None


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
        if cfg.system_prompt is None:
            cfg = replace(
                cfg,
                system_prompt=build_search_agent_instruction(
                    max_search_limit=cfg.max_turns,
                    max_url_fetch=cfg.max_url_fetch,
                ),
            )
        self.search_config = cfg
        self._result_evaluator = SearchResultEvaluator(cfg.evaluation_config)
        action_tags = [
            cfg.plan_tag,
            cfg.search_tag,
            cfg.searches_tag,
            cfg.fetch_tag,
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

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    async def _retrieve_many(self, queries: list[str]) -> list[list[SearchResult]]:
        try:
            return await self._search_client.retrieve(queries)
        except Exception as exc:
            logger.warning("Search failed for queries %r: %s", queries, exc)
            return [[] for _ in queries]

    async def _fetch_pages(self, urls: list[str]) -> list[SearchResult]:
        try:
            return await self._search_client.fetch_urls(urls)
        except Exception as exc:
            logger.warning("Page fetch failed for urls %r: %s", urls, exc)
            return []

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

    def _parse_urls(self, content: str) -> list[str]:
        return [part.strip() for part in re.split(r"[\n,]+", content) if part.strip()]

    def _collect_requested_queries_and_urls(
        self,
        actions: list[tuple[str, str]],
        search_tags: set[str],
        fetch_tag: str,
    ) -> tuple[list[str], list[str]]:
        queries: list[str] = []
        urls: list[str] = []
        for tag, content in actions:
            if tag in search_tags:
                queries.extend(self._parse_queries(content, tag))
            elif tag == fetch_tag:
                urls.extend(self._parse_urls(content))
        return self._dedupe_preserve_order(queries), self._dedupe_preserve_order(urls)

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

    def _format_full_page_information(self, pages: list[SearchResult]) -> str:
        if not pages:
            return "No page content available."

        sections: list[str] = []
        for index, page in enumerate(pages, 1):
            title = page.title or page.url or "No title"
            sections.append(f"Page {index}(Title: {title})")
            if page.url:
                sections.append(f"URL: {page.url}")
            sections.append(page.contents)
        return "\n".join(sections)

    def _append_user_observation(
        self,
        working_messages: list[dict[str, str]],
        content: str,
    ) -> None:
        working_messages.append({"role": "user", "content": content})

    def _build_answer_rejection_feedback(
        self,
        latest_evaluation: SearchRoundEvaluation | None,
    ) -> str:
        if latest_evaluation is None:
            return (
                "Do not answer yet. You have not completed a search round. "
                "Search first, then answer once the evidence is sufficient."
            )
        return (
            "Do not answer yet. The latest search evaluation was insufficient. "
            "Refine the keywords, search again, and only answer after the evidence is sufficient."
        )

    def _build_missing_action_feedback(self) -> str:
        return (
            "No action detected. Evidence is still insufficient. "
            "Issue a <searches> block to gather more evidence before answering."
        )

    def _build_search_observation(
        self,
        round_index: int,
        search_contexts: list[SearchContext],
        evaluation: SearchRoundEvaluation,
    ) -> str:
        return (
            self.search_config.evaluation_obs_template.format(
                content=evaluation.to_feedback_block()
            )
            + self.search_config.obs_template.format(
                content=self._format_round_information(round_index, search_contexts)
            )
        )

    def _build_full_page_observation(self, pages: list[SearchResult]) -> str:
        return self.search_config.full_page_obs_template.format(
            content=self._format_full_page_information(pages)
        )

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
        fetch_tag = self.search_config.fetch_tag
        answer_tag = self.search_config.answer_tag
        latest_evaluation = None
        consecutive_rejections = 0
        metrics["search_rounds"] = 0.0
        metrics["fetched_pages"] = 0.0
        metrics["answer_rejections"] = 0.0
        metrics["search_queries"] = 0.0

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
                    metrics["answer_rejections"] += 1
                    self._append_user_observation(
                        working_messages,
                        self.search_config.answer_rejection_template.format(
                            content=self._build_missing_action_feedback()
                        ),
                    )
                    continue
                break

            answer_requested = any(tag == answer_tag for tag, _ in actions)
            # Collect every query from every <search>/<searches> block in this response.
            # This handles models that emit <plan>…</plan><searches>…</searches> in one shot.
            all_queries, fetch_urls = self._collect_requested_queries_and_urls(
                actions,
                search_tags=search_tags,
                fetch_tag=fetch_tag,
            )

            if answer_requested and not all_queries and not fetch_urls:
                evidence_is_sufficient = (
                    latest_evaluation is not None and latest_evaluation.is_sufficient
                )
                if (
                    not self.search_config.require_sufficient_evidence_before_answer
                    or evidence_is_sufficient
                    or consecutive_rejections >= self.search_config.max_answer_rejections
                ):
                    break

                consecutive_rejections += 1
                metrics["answer_rejections"] += 1
                self._append_user_observation(
                    working_messages,
                    self.search_config.answer_rejection_template.format(
                        content=self._build_answer_rejection_feedback(latest_evaluation)
                    ),
                )
                continue

            if fetch_urls:
                limited_urls = fetch_urls[: self.search_config.max_url_fetch]
                pages = await self._fetch_pages(limited_urls)
                metrics["fetched_pages"] += len(pages)
                self._append_user_observation(
                    working_messages,
                    self._build_full_page_observation(pages),
                )
                continue

            if not all_queries:
                # Only a <plan> (no searches yet) — acknowledge and let the model continue.
                self._append_user_observation(
                    working_messages,
                    self.search_config.plan_obs_template,
                )
                continue

            # A search was issued — reset the rejection counter.
            consecutive_rejections = 0
            metrics["search_queries"] += len(all_queries)

            # --- parallel search round ---
            t0 = time.perf_counter()
            results_by_query = await self._retrieve_many(all_queries)
            elapsed = time.perf_counter() - t0
            metrics["search_rounds"] += 1
            total_results = sum(len(r) for r in results_by_query)
            logger.debug(
                "search returned %d total results across %d queries in %.2fs",
                total_results,
                len(all_queries),
                elapsed,
            )

            if not any(results_by_query):
                self._append_user_observation(
                    working_messages,
                    self.search_config.obs_template.format(
                        content="No valid queries were provided. Try again with at least one search query."
                    ),
                )
                continue

            search_contexts = agent_ctx.add_round(
                queries=all_queries, results_by_query=results_by_query
            )
            round_index = agent_ctx.num_rounds
            latest_evaluation = self._result_evaluator.evaluate_round(search_contexts)
            self._append_user_observation(
                working_messages,
                self._build_search_observation(
                    round_index=round_index,
                    search_contexts=search_contexts,
                    evaluation=latest_evaluation,
                ),
            )

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=all_response_ids,
            response_mask=self.build_response_mask(all_response_ids),
            num_turns=num_turns,
            metrics=metrics,
            request_id=request_id,
            context=agent_ctx,
        )
