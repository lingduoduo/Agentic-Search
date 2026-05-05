"""Multi-turn research loop using XML tags for planning, search, and synthesis."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from .agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    register,
    simple_timer,
)
from .context import AgentContext, SearchContext, SearchResult
from .evaluation import (
    SearchEvaluationConfig,
    SearchResultEvaluator,
    SearchRoundEvaluation,
)
from .search_client import SearchClient, SearchClientConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))

_TASK_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
_TASK_PREFIX_RE = re.compile(r"^\[(?P<task>[^\]]+)\]\s*(?P<query>.+)$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_QUERY_TAG_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_URL_SPLIT_RE = re.compile(r"[\n,]+")


def _normalize_task_id(raw: str) -> str:
    normalized = _TASK_ID_RE.sub("", raw.strip())
    return normalized or "T"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_search_agent_instruction(max_search_limit: int, max_url_fetch: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches and fetch webpage content.\n\n"
        "Tools:\n"
        "- To write a brief research plan: use <plan>...</plan>.\n"
        "- Before searching, decide whether internal knowledge is sufficient using <search_decision>answer</search_decision> or <search_decision>search</search_decision>.\n"
        "- To split a complex question into focused research tracks: use <subquestions>...</subquestions>.\n"
        "- To perform a search: use <search>single query</search> or <searches>one query per line</searches>.\n"
        "  For multi-track research, prefix a query with a task id like [T1] query text.\n"
        "- To fetch detailed content from specific URLs returned by search: use <fetch>url1, url2</fetch>.\n"
        "- To provide the final response: use <answer>...</answer>.\n\n"
        "System feedback:\n"
        "- Search results are returned inside <information>...</information>.\n"
        "- Search-quality judgments are returned inside <search_evaluation>...</search_evaluation>.\n"
        "- Full webpage content is returned inside <full_page>...</full_page>.\n\n"
        f"You may execute at most {max_search_limit} search rounds.\n"
        f"You may fetch up to {max_url_fetch} URLs in one fetch request.\n\n"
        "Workflow:\n"
        "1. Plan.\n"
        "2. Decide whether you can answer from internal knowledge or need external evidence.\n"
        "3. If the question has multiple parts, declare subquestions.\n"
        "4. Search, preferably with parallel queries when useful.\n"
        "5. Track evidence for each subquestion.\n"
        "6. If the evidence is weak, search again with refined keywords.\n"
        "7. If snippets are not enough, fetch the most promising URLs.\n"
        "8. Answer directly without search only when internal knowledge is sufficient; otherwise answer after evidence is sufficient for the overall question and each active subquestion.\n\n"
        "Cite the evidence labels from the information blocks when you answer."
    )


DEFAULT_SYSTEM_PROMPT = build_search_agent_instruction(
    max_search_limit=5, max_url_fetch=3
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
    decision_tag: str = "search_decision"
    subquestions_tag: str = "subquestions"
    search_tag: str = "search"
    searches_tag: str = "searches"
    fetch_tag: str = "fetch"
    answer_tag: str = "answer"
    max_url_fetch: int = 3
    # Explicit /fetch endpoint URL. When None, derived from search_url automatically.
    fetch_url: str | None = None
    plan_obs_template: str = (
        "\n\n<plan_feedback>\n"
        "Plan recorded. Continue by issuing one or more search queries.\n"
        "</plan_feedback>\n\n"
    )
    evaluation_obs_template: str = (
        "\n\n<search_evaluation>\n{content}\n</search_evaluation>\n\n"
    )
    full_page_obs_template: str = "\n\n<full_page>\n{content}\n</full_page>\n\n"
    subquestions_obs_template: str = (
        "\n\n<subquestions_feedback>\n{content}\n</subquestions_feedback>\n\n"
    )
    decision_obs_template: str = (
        "\n\n<decision_feedback>\n{content}\n</decision_feedback>\n\n"
    )
    answer_rejection_template: str = (
        "\n\n<answer_feedback>\n{content}\n</answer_feedback>\n\n"
    )
    require_sufficient_evidence_before_answer: bool = True
    # Stop rejecting <answer> after this many consecutive rejections (avoids infinite loops).
    max_answer_rejections: int = 3
    evaluation_config: SearchEvaluationConfig = SearchEvaluationConfig()
    obs_template: str = "\n\n<information>\n{content}\n</information>\n\n"
    system_prompt: str | None = None
    # Max search rounds (not queries). None defaults to max_turns.
    max_search_limit: int | None = None
    search_limit_template: str = (
        "\n\n<search_feedback>\n"
        "Search limit reached. Do not issue more searches; use the evidence already collected or refine with fetched pages.\n"
        "</search_feedback>\n\n"
    )
    repeated_query_template: str = (
        "\n\n<search_feedback>\n"
        "Repeated search skipped for these queries:\n{content}\n"
        "Refer to the earlier results instead of searching them again.\n"
        "</search_feedback>\n\n"
    )
    allow_internal_knowledge_answer: bool = True


@register("search_agent")
class SearchAgentLoop(AgentLoopBase):
    """Multi-turn agent loop for research planning, search, and synthesis.

    Each turn the model generates a response containing one or more XML action
    tags. The loop processes them in order — plan → subquestions → searches →
    fetch → answer — injecting observations back as a single user message before
    the next generation step.

    AgentLoopOutput.context is an AgentContext holding all search rounds for
    downstream inspection or reward computation.
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
        # Resolve deferred defaults in one replace() call.
        resolved: dict[str, Any] = {}
        if cfg.system_prompt is None:
            resolved["system_prompt"] = build_search_agent_instruction(
                max_search_limit=cfg.max_search_limit or cfg.max_turns,
                max_url_fetch=cfg.max_url_fetch,
            )
        if cfg.max_search_limit is None:
            resolved["max_search_limit"] = cfg.max_turns
        if resolved:
            cfg = replace(cfg, **resolved)

        self.search_config = cfg
        self._result_evaluator = SearchResultEvaluator(cfg.evaluation_config)
        # Per-task evaluator: same config but min_total_results relaxed to per-query minimum
        # so a single-query task isn't rejected just because it has fewer than min_total_results.
        task_eval_config = replace(
            cfg.evaluation_config,
            min_total_results=max(1, cfg.evaluation_config.min_results_per_query),
        )
        self._task_evaluator = SearchResultEvaluator(task_eval_config)

        action_tags = [
            cfg.plan_tag,
            cfg.decision_tag,
            cfg.subquestions_tag,
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
                fetch_url=cfg.fetch_url,
            )
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_actions(self, text: str) -> list[tuple[str, str]]:
        return [
            (m.group(1), m.group(2).strip()) for m in self._action_re.finditer(text)
        ]

    def _parse_queries(self, content: str, action: str | None) -> list[str]:
        if action == self.search_config.search_tag:
            return [content] if content else []
        query_tags = _QUERY_TAG_RE.findall(content)
        if query_tags:
            return [q.strip() for q in query_tags if q.strip()]
        return [
            cleaned
            for line in content.splitlines()
            if (cleaned := _LIST_PREFIX_RE.sub("", line).strip())
        ]

    def _parse_query_specifications(
        self,
        content: str,
        action: str | None,
    ) -> list[tuple[str | None, str]]:
        specs: list[tuple[str | None, str]] = []
        for raw in self._parse_queries(content, action):
            m = _TASK_PREFIX_RE.match(raw)
            if m:
                specs.append(
                    (_normalize_task_id(m.group("task")), m.group("query").strip())
                )
            else:
                specs.append((None, raw))
        return specs

    def _parse_urls(self, content: str) -> list[str]:
        return [p.strip() for p in _URL_SPLIT_RE.split(content) if p.strip()]

    def _parse_subquestions(
        self,
        content: str,
        existing_tasks: dict[str, str],
    ) -> dict[str, str]:
        parsed: dict[str, str] = {}
        next_index = len(existing_tasks) + 1
        for line in content.splitlines():
            cleaned = _LIST_PREFIX_RE.sub("", line).strip()
            if not cleaned:
                continue
            if ":" in cleaned:
                raw_id, desc = cleaned.split(":", 1)
                task_id, desc = _normalize_task_id(raw_id), desc.strip()
            else:
                task_id, desc = f"T{next_index}", cleaned
                next_index += 1
            if desc:
                parsed[task_id] = desc
        return parsed

    def _parse_search_decision(self, content: str) -> str | None:
        normalized = content.strip().lower()
        if normalized in {"answer", "internal", "internal_knowledge", "direct_answer"}:
            return "answer"
        if normalized in {"search", "retrieve", "external", "need_search"}:
            return "search"
        return None

    def _collect_requested_queries_and_urls(
        self,
        actions: list[tuple[str, str]],
        search_tags: set[str],
        fetch_tag: str,
    ) -> tuple[list[tuple[str | None, str]], list[str]]:
        raw_query_specs: list[tuple[str | None, str]] = []
        raw_urls: list[str] = []
        for tag, content in actions:
            if tag in search_tags:
                raw_query_specs.extend(self._parse_query_specifications(content, tag))
            elif tag == fetch_tag:
                raw_urls.extend(self._parse_urls(content))
        # Deduplicate while preserving order.
        deduped_specs = list(dict.fromkeys(raw_query_specs))
        return deduped_specs, _dedupe(raw_urls)

    def _partition_search_requests(
        self,
        query_specs: list[tuple[str | None, str]],
        executed_queries: set[str],
        rounds_used: int,
    ) -> tuple[list[tuple[str | None, str]], list[str], list[str]]:
        """Split query_specs into (allowed, repeated, overflow).

        A round is blocked as overflow when rounds_used has reached max_search_limit.
        Within an allowed round, individual repeated queries are still filtered out.
        """
        limit = self.search_config.max_search_limit or 0
        at_limit = limit > 0 and rounds_used >= limit

        allowed: list[tuple[str | None, str]] = []
        repeated: list[str] = []
        overflow: list[str] = []
        for task_id, query in query_specs:
            if query in executed_queries:
                repeated.append(query)
            elif at_limit:
                overflow.append(query)
            else:
                allowed.append((task_id, query))
        return allowed, repeated, overflow

    # ------------------------------------------------------------------
    # Network helpers (with per-run caches passed in)
    # ------------------------------------------------------------------

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

    async def _retrieve_with_cache(
        self,
        queries: list[str],
        cache: dict[str, list[SearchResult]],
    ) -> list[list[SearchResult]]:
        uncached = [q for q in queries if q not in cache]
        if uncached:
            rows = await self._retrieve_many(uncached)
            cache.update(zip(uncached, rows))
        return [list(cache.get(q, [])) for q in queries]

    async def _fetch_with_cache(
        self,
        urls: list[str],
        cache: dict[str, SearchResult],
    ) -> list[SearchResult]:
        uncached = [u for u in urls if u not in cache]
        if uncached:
            pages = await self._fetch_pages(uncached)
            cache.update({p.url: p for p in pages if p.url})
        return [cache[u] for u in urls if u in cache]

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _format_round_information(
        self,
        round_index: int,
        search_contexts: list[SearchContext],
    ) -> str:
        if not search_contexts:
            return f"Round {round_index}\nNo information available"
        sections = [f"Round {round_index}"]
        for i, ctx in enumerate(search_contexts, 1):
            sections.append(f"Query {i}: {ctx.query}")
            sections.append(
                ctx.to_information_block(citation_prefix=f"R{round_index}Q{i}D")
            )
        return "\n".join(sections)

    def _format_full_page_information(self, pages: list[SearchResult]) -> str:
        if not pages:
            return "No page content available."
        sections: list[str] = []
        for i, page in enumerate(pages, 1):
            title = page.title or page.url or "No title"
            sections.append(f"Page {i}(Title: {title})")
            if page.url:
                sections.append(f"URL: {page.url}")
            sections.append(page.contents)
        return "\n".join(sections)

    def _build_search_observation(
        self,
        round_index: int,
        search_contexts: list[SearchContext],
        evaluation: SearchRoundEvaluation,
    ) -> str:
        return self.search_config.evaluation_obs_template.format(
            content=evaluation.to_feedback_block()
        ) + self.search_config.obs_template.format(
            content=self._format_round_information(round_index, search_contexts)
        )

    def _build_answer_rejection_feedback(
        self,
        latest_evaluation: SearchRoundEvaluation | None,
        task_statuses: dict[str, bool],
        active_tasks: dict[str, str],
    ) -> str:
        if latest_evaluation is None:
            return (
                "Do not answer yet. You have not completed a search round. "
                "Search first, then answer once the evidence is sufficient."
            )
        base = (
            "Do not answer yet. The latest search evaluation was insufficient. "
            "Refine the keywords, search again, and only answer after the evidence is sufficient."
        )
        missing = [
            f"{tid}: {desc}"
            for tid, desc in active_tasks.items()
            if not task_statuses.get(tid, False)
        ]
        if missing:
            base += (
                " The following subquestions still need stronger evidence: "
                + "; ".join(missing)
            )
        return base

    def _build_decision_feedback(self, decision: str | None) -> str:
        if decision == "answer":
            return "Internal knowledge may be sufficient. Provide the answer directly if you are confident."
        if decision == "search":
            return "External knowledge is needed. Issue a <search> or <searches> action next."
        return (
            "Decide whether internal knowledge is sufficient. "
            "Use <search_decision>answer</search_decision> to answer directly or "
            "<search_decision>search</search_decision> before retrieving evidence."
        )

    def _evaluate_tasks(
        self,
        search_contexts: list[SearchContext],
    ) -> dict[str, SearchRoundEvaluation]:
        task_groups: dict[str, list[SearchContext]] = {}
        for ctx in search_contexts:
            if ctx.task_id:
                task_groups.setdefault(ctx.task_id, []).append(ctx)
        return {
            tid: self._task_evaluator.evaluate_round(ctxs)
            for tid, ctxs in task_groups.items()
        }

    def _has_sufficient_evidence(
        self,
        latest_evaluation: SearchRoundEvaluation | None,
        task_statuses: dict[str, bool],
        active_tasks: dict[str, str],
    ) -> bool:
        if latest_evaluation is None or not latest_evaluation.is_sufficient:
            return False
        return all(task_statuses.get(tid, False) for tid in active_tasks)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {
            "search_rounds": 0.0,
            "fetched_pages": 0.0,
            "answer_rejections": 0.0,
            "search_queries": 0.0,
            "active_subquestions": 0.0,
            "search_limit_hits": 0.0,
            "repeated_search_queries": 0.0,
            "search_cache_hits": 0.0,
            "page_cache_hits": 0.0,
            "decision_prompts": 0.0,
            "direct_answers": 0.0,
        }
        request_id = uuid4().hex
        agent_ctx = AgentContext()

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
        final_answer: str | None = None

        # Per-run state
        latest_evaluation: SearchRoundEvaluation | None = None
        active_tasks: dict[str, str] = {}
        task_statuses: dict[str, bool] = {}
        latest_search_decision: str | None = None
        consecutive_rejections = 0
        rounds_used = 0
        executed_queries: set[str] = set()
        search_cache: dict[str, list[SearchResult]] = {}
        page_cache: dict[str, SearchResult] = {}

        cfg = self.search_config
        decision_tag = cfg.decision_tag
        subquestions_tag = cfg.subquestions_tag
        search_tags = {cfg.search_tag, cfg.searches_tag}
        fetch_tag = cfg.fetch_tag
        answer_tag = cfg.answer_tag

        try:
            for turn in range(cfg.max_turns):
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

                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                actions = self._parse_actions(response_text)
                logger.debug(
                    "turn=%d actions=%r", turn, [(t, c[:60]) for t, c in actions]
                )

                working_messages.append({"role": "assistant", "content": response_text})

                # No recognised tag: re-prompt depending on where we are in the workflow.
                if not actions:
                    needs_more = (
                        cfg.require_sufficient_evidence_before_answer
                        and not self._has_sufficient_evidence(
                            latest_evaluation, task_statuses, active_tasks
                        )
                        and consecutive_rejections < cfg.max_answer_rejections
                    )
                    if needs_more:
                        consecutive_rejections += 1
                        metrics["answer_rejections"] += 1
                        if rounds_used == 0:
                            # No search has happened yet — ask the model to decide whether it needs one.
                            metrics["decision_prompts"] += 1
                            feedback = self._build_decision_feedback(None)
                        else:
                            feedback = (
                                "No action detected. Evidence is still insufficient. "
                                "Issue a <searches> block to gather more evidence before answering."
                            )
                        working_messages.append(
                            {
                                "role": "user",
                                "content": cfg.answer_rejection_template.format(
                                    content=feedback
                                ),
                            }
                        )
                        continue
                    break

                # Process <subquestions> declarations.
                declared_subquestions: dict[str, str] = {}
                for tag, content in actions:
                    if tag == decision_tag:
                        latest_search_decision = self._parse_search_decision(content)
                    if tag == subquestions_tag:
                        declared_subquestions.update(
                            self._parse_subquestions(
                                content, active_tasks | declared_subquestions
                            )
                        )
                if declared_subquestions:
                    active_tasks.update(declared_subquestions)
                    task_statuses.update({tid: False for tid in declared_subquestions})
                    agent_ctx.register_tasks(declared_subquestions)
                    metrics["active_subquestions"] = float(len(active_tasks))

                query_specs, fetch_urls = self._collect_requested_queries_and_urls(
                    actions, search_tags=search_tags, fetch_tag=fetch_tag
                )
                allowed_specs, repeated, overflow = self._partition_search_requests(
                    query_specs,
                    executed_queries=executed_queries,
                    rounds_used=rounds_used,
                )
                all_queries = [q for _, q in allowed_specs]
                query_task_ids = [tid for tid, _ in allowed_specs]

                # Tentatively record the answer from this turn.  Doing this
                # before the gating block ensures that a turn mixing <answer>
                # with <search> or <fetch> still contributes a candidate answer
                # if the loop exits without a later answer-only turn.
                answer_tag_contents = [c for t, c in actions if t == answer_tag]
                if answer_tag_contents:
                    final_answer = answer_tag_contents[0].strip()

                # Answer gating: block early answers if evidence is insufficient.
                if (
                    any(tag == answer_tag for tag, _ in actions)
                    and not all_queries
                    and not fetch_urls
                ):
                    if (
                        cfg.allow_internal_knowledge_answer
                        and rounds_used == 0
                        and latest_search_decision == "answer"
                        and not active_tasks
                    ):
                        metrics["direct_answers"] += 1.0
                        break
                    if (
                        not cfg.require_sufficient_evidence_before_answer
                        or self._has_sufficient_evidence(
                            latest_evaluation, task_statuses, active_tasks
                        )
                        or consecutive_rejections >= cfg.max_answer_rejections
                    ):
                        break
                    # Answer rejected — clear the tentative candidate so a
                    # discarded answer is not returned as the final answer.
                    final_answer = None
                    consecutive_rejections += 1
                    metrics["answer_rejections"] += 1
                    working_messages.append(
                        {
                            "role": "user",
                            "content": cfg.answer_rejection_template.format(
                                content=self._build_answer_rejection_feedback(
                                    latest_evaluation, task_statuses, active_tasks
                                )
                            ),
                        }
                    )
                    continue

                # Build observation for this turn (search + fetch combined into one message).
                turn_observations: list[str] = []

                if repeated:
                    metrics["repeated_search_queries"] += len(repeated)
                    turn_observations.append(
                        cfg.repeated_query_template.format(
                            content="\n".join(f"- {q}" for q in repeated)
                        )
                    )
                if overflow:
                    metrics["search_limit_hits"] += 1.0
                    turn_observations.append(cfg.search_limit_template)

                # Plan / subquestions only — no search or fetch.
                if not all_queries and not fetch_urls:
                    if declared_subquestions:
                        turn_observations.append(
                            cfg.subquestions_obs_template.format(
                                content="Registered subquestions:\n"
                                + "\n".join(
                                    f"- {tid}: {desc}"
                                    for tid, desc in declared_subquestions.items()
                                )
                            )
                        )
                    elif any(tag == decision_tag for tag, _ in actions):
                        metrics["decision_prompts"] += 1
                        turn_observations.append(
                            cfg.decision_obs_template.format(
                                content=self._build_decision_feedback(
                                    latest_search_decision
                                )
                            )
                        )
                    else:
                        turn_observations.append(cfg.plan_obs_template)
                    working_messages.append(
                        {"role": "user", "content": "".join(turn_observations)}
                    )
                    continue

                # Parallel search round (before fetch so evidence appears first).
                if all_queries:
                    consecutive_rejections = 0
                    metrics["search_queries"] += len(all_queries)
                    metrics["search_cache_hits"] += sum(
                        1 for q in all_queries if q in search_cache
                    )
                    rounds_used += 1  # count rounds, not individual queries
                    executed_queries.update(all_queries)

                    t0 = time.perf_counter()
                    results_by_query = await self._retrieve_with_cache(
                        all_queries, search_cache
                    )
                    elapsed = time.perf_counter() - t0
                    metrics["search_rounds"] += 1
                    logger.debug(
                        "search returned %d total results across %d queries in %.2fs",
                        sum(len(r) for r in results_by_query),
                        len(all_queries),
                        elapsed,
                    )

                    if any(results_by_query):
                        search_contexts = agent_ctx.add_round(
                            queries=all_queries,
                            results_by_query=results_by_query,
                            task_ids=query_task_ids,
                        )
                        latest_evaluation = self._result_evaluator.evaluate_round(
                            search_contexts
                        )
                        for tid, ev in self._evaluate_tasks(search_contexts).items():
                            task_statuses[tid] = ev.is_sufficient
                        turn_observations.append(
                            self._build_search_observation(
                                agent_ctx.num_rounds, search_contexts, latest_evaluation
                            )
                        )
                    else:
                        turn_observations.append(
                            cfg.obs_template.format(
                                content="No results returned. Try rephrasing the search query."
                            )
                        )

                # Page fetch (appended after search results in the same message).
                if fetch_urls:
                    limited = fetch_urls[: cfg.max_url_fetch]
                    metrics["page_cache_hits"] += sum(
                        1 for u in limited if u in page_cache
                    )
                    pages = await self._fetch_with_cache(limited, page_cache)
                    metrics["fetched_pages"] += len(pages)
                    turn_observations.append(
                        cfg.full_page_obs_template.format(
                            content=self._format_full_page_information(pages)
                        )
                    )

                working_messages.append(
                    {"role": "user", "content": "".join(turn_observations)}
                )
        finally:
            close_client = getattr(self._search_client, "aclose", None)
            if close_client is not None:
                close_result = close_client()
                if inspect.isawaitable(close_result):
                    await close_result

        # Derived metrics used by the reward function — computed once here so
        # callers don't have to re-derive them from the raw counts.
        # Use total *attempted* queries (executed + duplicates) as denominator
        # so the ratio stays in [0, 1] even when duplicates exceed new queries.
        total_attempted = metrics["search_queries"] + metrics["repeated_search_queries"]
        metrics["repeated_query_ratio"] = (
            metrics["repeated_search_queries"] / total_attempted
            if total_attempted
            else 0.0
        )
        metrics["subquestion_coverage_ratio"] = (
            sum(task_statuses.values()) / len(task_statuses) if task_statuses else 1.0
        )
        metrics["rounds_used"] = float(rounds_used)

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=all_response_ids,
            response_mask=self.build_response_mask(all_response_ids),
            num_turns=num_turns,
            metrics=metrics,
            request_id=request_id,
            context=agent_ctx,
            final_answer=final_answer,
        )
