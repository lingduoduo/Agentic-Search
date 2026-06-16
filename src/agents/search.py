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

from .base import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    register,
    simple_timer,
)
from ..context.search import AgentContext, SearchContext, SearchResult
from ..training.evaluation import (
    SearchEvaluationConfig,
    SearchResultEvaluator,
    SearchRoundEvaluation,
)
from ..context.retrieval.client import SearchClient, SearchClientConfig

# ---------------------------------------------------------------------------
# Search tool-call types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchToolCall:
    """The model's search decision — parsed from <search>...</search> output.

    Captures which queries to run this round, which were already seen
    (repeated), and which are blocked by the round budget (overflow).
    """

    queries: list[str]
    task_ids: list[str | None]
    repeated: list[str]
    overflow: list[str]

    @property
    def has_new_queries(self) -> bool:
        return bool(self.queries)


@dataclass(frozen=True)
class SearchRoundResult:
    """Outcome of executing one search round as a tool call."""

    search_contexts: list[SearchContext]
    evaluation: SearchRoundEvaluation | None
    observation: str


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))

_TASK_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
_TASK_PREFIX_RE = re.compile(r"^\[(?P<task>[^\]]+)\]\s*(?P<query>.+)$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_QUERY_TAG_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_URL_SPLIT_RE = re.compile(r"[\n,]+")
_SPACE_RE = re.compile(r"\s+")


def _normalize_task_id(raw: str) -> str:
    normalized = _TASK_ID_RE.sub("", raw.strip())
    return normalized or "T"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _normalize_result_fingerprint(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip().lower())


def _result_fingerprint(result: SearchResult) -> str:
    """Stable key for duplicate evidence across enterprise search backends."""
    for key in ("document_id", "doc_id", "chunk_id", "id"):
        value = result.metadata.get(key)
        if value:
            return f"{key}:{value}"
    if result.url:
        return "url:" + result.url.split("#", 1)[0].rstrip("/").lower()
    if result.title:
        return "title:" + _normalize_result_fingerprint(result.title)
    return "content:" + _normalize_result_fingerprint(result.contents[:512])


def build_search_agent_instruction(max_search_limit: int, max_url_fetch: int) -> str:
    return (
        "You are a search-capable reasoning assistant. Follow this XML workflow:\n\n"
        "<think>\n"
        "Decide whether the question can be answered directly or needs external evidence. "
        "State what evidence is missing and what search strategy to use. "
        "Do not include final-answer prose here.\n"
        "</think>\n"
        "<search>\n"
        "One precise search query when external evidence is needed.\n"
        "</search>\n"
        "<information>\n"
        "Environment-injected search results only. Never write or fabricate this block.\n"
        "</information>\n"
        "<answer>\n"
        "Final answer grounded in the available evidence.\n"
        "</answer>\n\n"
        "Extended actions (use when needed):\n"
        "- <search_decision>answer</search_decision> to skip search when internal knowledge is sufficient, "
        "or <search_decision>search</search_decision> before the first query.\n"
        "- <subquestions>one research subquestion per line</subquestions> when the task has multiple facets.\n"
        "- <searches>parallel precise queries, one per line</searches> for independent subquestions.\n"
        "- <fetch>comma or newline separated URLs</fetch> when snippets are insufficient.\n\n"
        "Behavior rules:\n"
        "- Open every turn with <think>. Keep it concise — decide the next single useful action.\n"
        "- Search when the answer depends on current, obscure, disputed, or source-specific facts.\n"
        "- For multi-track research, prefix queries with a task id: [T1] query text.\n"
        "- Write focused queries: key entity + date/timeframe + fact to verify.\n"
        "- Avoid duplicate queries; refine instead of repeating.\n"
        "- Use <searches> only when queries cover independent subquestions.\n"
        "- Fetch only the most promising URLs when snippets are not enough.\n"
        "- Answer only when evidence is sufficient for every active subquestion.\n"
        "- Always cite evidence labels from <information> blocks in the final answer.\n"
        "- If evidence is missing or conflicting, state what is known and avoid overclaiming.\n\n"
        "Environment-only tags (never output these yourself):\n"
        "- <information> — search results with citation labels.\n"
        "- <search_evaluation> — sufficiency verdict and weak-query hints.\n"
        "- <subquestions_feedback> — per-subquestion coverage status.\n"
        "- <full_page> — fetched page content.\n\n"
        f"Budget: at most {max_search_limit} search rounds and at most {max_url_fetch} URLs per fetch.\n\n"
        "Preferred turn order:\n"
        "1. <think> — reason about what is known and what is missing\n"
        "2. <search_decision> (optional) — skip to <answer> if internal knowledge suffices\n"
        "3. <subquestions> (optional) — declare sub-tasks for multi-facet questions\n"
        "4. <search> or <searches> — retrieve evidence\n"
        "5. Check <search_evaluation> and subquestion coverage\n"
        "6. <fetch> strong candidates if snippets are insufficient\n"
        "7. <answer> — once all subquestions have sufficient evidence"
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
    plan_tag: str = "think"
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
        "\n\n<think_feedback>\n"
        "Reasoning recorded. Continue by issuing one or more search queries.\n"
        "</think_feedback>\n\n"
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
    deduplicate_search_results: bool = True


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
            rf"<({'|'.join(re.escape(tag) for tag in action_tags)})>" rf"(.*?)</\1>",
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

    def _initial_metrics(self) -> dict[str, float]:
        return {
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
            "evidence_sufficient_rounds": 0.0,
            "evidence_insufficient_rounds": 0.0,
            "search_quality_score_sum": 0.0,
            "duplicate_search_results_removed": 0.0,
            "implicit_subquestions": 0.0,
            "research_followup_queries": 0.0,
        }

    def _with_system_prompt(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.search_config.system_prompt and (
            not messages or messages[0].get("role") != "system"
        ):
            return [
                {"role": "system", "content": self.search_config.system_prompt},
                *messages,
            ]
        return list(messages)

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

    def _deduplicate_results_by_source(
        self,
        results_by_query: list[list[SearchResult]],
    ) -> tuple[list[list[SearchResult]], int]:
        """Remove repeated sources from a multi-query evidence pack.

        Enterprise corpora often return the same document chunk for several
        follow-up queries. Keeping the first occurrence preserves citation
        labels while reducing context waste and duplicate synthesis pressure.
        Within one query, the higher-scored duplicate is retained.
        """
        seen: set[str] = set()
        deduped_rows: list[list[SearchResult]] = []
        removed = 0

        for row in results_by_query:
            best_by_key: dict[str, SearchResult] = {}
            key_order: list[str] = []
            for result in row:
                key = _result_fingerprint(result)
                current = best_by_key.get(key)
                if current is None:
                    best_by_key[key] = result
                    key_order.append(key)
                elif result.score > current.score:
                    removed += 1
                    best_by_key[key] = result
                else:
                    removed += 1

            row_out: list[SearchResult] = []
            for key in key_order:
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                row_out.append(best_by_key[key])
            deduped_rows.append(row_out)

        return deduped_rows, removed

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

    def _build_subquestion_status_feedback(
        self,
        active_tasks: dict[str, str],
        task_statuses: dict[str, bool],
        task_search_counts: dict[str, int] | None = None,
    ) -> str:
        if not active_tasks:
            return "No active subquestions."

        covered: list[str] = []
        missing: list[str] = []
        for tid, desc in active_tasks.items():
            line = f"{tid}: {desc}"
            if task_search_counts and task_search_counts.get(tid, 0):
                line += f" (searches: {task_search_counts[tid]})"
            if task_statuses.get(tid, False):
                covered.append(line)
            else:
                missing.append(line)

        sections: list[str] = []
        if covered:
            sections.append("Covered:\n" + "\n".join(f"- {item}" for item in covered))
        if missing:
            sections.append(
                "Needs more evidence:\n" + "\n".join(f"- {item}" for item in missing)
            )
        return "\n".join(sections)

    def _register_implicit_tasks(
        self,
        query_specs: list[tuple[str | None, str]],
        active_tasks: dict[str, str],
        task_statuses: dict[str, bool],
        agent_ctx: AgentContext,
    ) -> int:
        """Register task-prefixed searches as research tracks when needed."""
        implicit: dict[str, str] = {}
        for task_id, query in query_specs:
            if task_id and task_id not in active_tasks and task_id not in implicit:
                implicit[task_id] = query
        if not implicit:
            return 0
        active_tasks.update(implicit)
        task_statuses.update({tid: False for tid in implicit})
        agent_ctx.register_tasks(implicit)
        return len(implicit)

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
    # Search tool-call execution
    # ------------------------------------------------------------------

    async def _execute_search_round(
        self,
        tool_call: SearchToolCall,
        *,
        agent_ctx: AgentContext,
        search_cache: dict[str, list[SearchResult]],
        active_tasks: dict[str, str],
        task_statuses: dict[str, bool],
        metrics: dict[str, float],
    ) -> SearchRoundResult:
        """Execute the model's search tool call and return the observation.

        This is the tool_call() invoked when the model outputs
        <search>query</search> or <searches>...</searches>.  Handles retrieval,
        cache, evaluation, and observation formatting in one place.
        """
        queries = tool_call.queries
        task_ids = tool_call.task_ids
        cfg = self.search_config

        metrics["search_queries"] += len(queries)
        metrics["search_cache_hits"] += sum(1 for q in queries if q in search_cache)

        t0 = time.perf_counter()
        results_by_query = await self._retrieve_with_cache(queries, search_cache)
        if cfg.deduplicate_search_results:
            results_by_query, removed = self._deduplicate_results_by_source(
                results_by_query
            )
            metrics["duplicate_search_results_removed"] += removed
        metrics["search_rounds"] += 1
        logger.debug(
            "search returned %d total results across %d queries in %.2fs",
            sum(len(r) for r in results_by_query),
            len(queries),
            time.perf_counter() - t0,
        )

        if any(results_by_query):
            search_contexts = agent_ctx.add_round(
                queries=queries,
                results_by_query=results_by_query,
                task_ids=task_ids,
            )
            evaluation = self._result_evaluator.evaluate_round(search_contexts)
            sufficient_queries = sum(1 for qe in evaluation.queries if qe.is_sufficient)
            if evaluation.queries:
                metrics["search_quality_score_sum"] += sufficient_queries / len(
                    evaluation.queries
                )
            if evaluation.is_sufficient:
                metrics["evidence_sufficient_rounds"] += 1.0
            else:
                metrics["evidence_insufficient_rounds"] += 1.0
            for tid, ev in self._evaluate_tasks(search_contexts).items():
                task_statuses[tid] = ev.is_sufficient
            return SearchRoundResult(
                search_contexts=search_contexts,
                evaluation=evaluation,
                observation=self._build_search_observation(
                    agent_ctx.num_rounds, search_contexts, evaluation
                ),
            )

        metrics["evidence_insufficient_rounds"] += 1.0
        return SearchRoundResult(
            search_contexts=[],
            evaluation=None,
            observation=cfg.obs_template.format(
                content="No results returned. Try rephrasing the search query."
            ),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        on_turn=None,
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = self._initial_metrics()
        request_id = uuid4().hex
        agent_ctx = AgentContext()

        working_messages = self._with_system_prompt(list(messages))

        all_response_ids: list[int] = []
        final_prompt_ids: list[int] = []
        num_turns = 0
        final_answer: str | None = None
        action_trace_parts: list[str] = []

        # Per-run state
        latest_evaluation: SearchRoundEvaluation | None = None
        active_tasks: dict[str, str] = {}
        task_statuses: dict[str, bool] = {}
        latest_search_decision: str | None = None
        consecutive_rejections = 0
        rounds_used = 0
        executed_queries: set[str] = set()
        task_search_counts: dict[str, int] = {}
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

                response_text = self.decode_response_ids(response_ids)
                actions = self._parse_actions(response_text)
                logger.debug(
                    "turn=%d actions=%r", turn, [(t, c[:60]) for t, c in actions]
                )

                working_messages.append({"role": "assistant", "content": response_text})
                if actions:
                    action_trace_parts.append(response_text)

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
                implicit_tasks = self._register_implicit_tasks(
                    query_specs, active_tasks, task_statuses, agent_ctx
                )
                if implicit_tasks:
                    metrics["implicit_subquestions"] += implicit_tasks
                    metrics["active_subquestions"] = float(len(active_tasks))
                allowed_specs, repeated, overflow = self._partition_search_requests(
                    query_specs,
                    executed_queries=executed_queries,
                    rounds_used=rounds_used,
                )
                search_tool_call = SearchToolCall(
                    queries=[q for _, q in allowed_specs],
                    task_ids=[tid for tid, _ in allowed_specs],
                    repeated=repeated,
                    overflow=overflow,
                )

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
                    and not search_tool_call.has_new_queries
                    and not fetch_urls
                ):
                    if (
                        cfg.allow_internal_knowledge_answer
                        and rounds_used == 0
                        and latest_search_decision == "answer"
                        and not active_tasks
                    ):
                        metrics["direct_answers"] += 1.0
                        if on_turn is not None:
                            await on_turn(num_turns, None, 0)
                        break
                    if (
                        not cfg.require_sufficient_evidence_before_answer
                        or self._has_sufficient_evidence(
                            latest_evaluation, task_statuses, active_tasks
                        )
                        or consecutive_rejections >= cfg.max_answer_rejections
                    ):
                        if on_turn is not None:
                            await on_turn(num_turns, None, 0)
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

                if search_tool_call.repeated:
                    metrics["repeated_search_queries"] += len(search_tool_call.repeated)
                    turn_observations.append(
                        cfg.repeated_query_template.format(
                            content="\n".join(
                                f"- {q}" for q in search_tool_call.repeated
                            )
                        )
                    )
                if search_tool_call.overflow:
                    metrics["search_limit_hits"] += 1.0
                    turn_observations.append(cfg.search_limit_template)

                # Plan / subquestions only — no search or fetch.
                if not search_tool_call.has_new_queries and not fetch_urls:
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
                if search_tool_call.has_new_queries:
                    consecutive_rejections = 0
                    rounds_used += 1  # count rounds, not individual queries
                    for task_id in search_tool_call.task_ids:
                        if not task_id:
                            continue
                        if task_search_counts.get(task_id, 0):
                            metrics["research_followup_queries"] += 1.0
                        task_search_counts[task_id] = (
                            task_search_counts.get(task_id, 0) + 1
                        )
                    executed_queries.update(search_tool_call.queries)
                    round_result = await self._execute_search_round(
                        search_tool_call,
                        agent_ctx=agent_ctx,
                        search_cache=search_cache,
                        active_tasks=active_tasks,
                        task_statuses=task_statuses,
                        metrics=metrics,
                    )
                    latest_evaluation = round_result.evaluation
                    turn_observations.append(round_result.observation)
                    if on_turn is not None:
                        doc_count = sum(
                            len(sc.results) for sc in round_result.search_contexts
                        )
                        await on_turn(turn + 1, "search_routing_tool", doc_count)
                    if active_tasks:
                        turn_observations.append(
                            cfg.subquestions_obs_template.format(
                                content=self._build_subquestion_status_feedback(
                                    active_tasks, task_statuses, task_search_counts
                                )
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
                    agent_ctx.record_fetched_pages(pages)
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
        metrics["subquestions_covered"] = float(sum(task_statuses.values()))
        metrics["research_tasks_with_followup"] = float(
            sum(1 for count in task_search_counts.values() if count > 1)
        )
        metrics["rounds_used"] = float(rounds_used)
        metrics["budget_used_ratio"] = rounds_used / max(cfg.max_search_limit or 1, 1)
        metrics["search_quality_score"] = (
            metrics["search_quality_score_sum"] / metrics["search_rounds"]
            if metrics["search_rounds"]
            else 0.0
        )
        cited_contexts = agent_ctx.cited_search_contexts(final_answer or "")
        cited_task_ids = agent_ctx.cited_task_ids(final_answer or "")
        metrics["citation_count"] = float(
            len(agent_ctx.cited_result_ids(final_answer or ""))
        )
        metrics["cited_search_contexts"] = float(len(cited_contexts))
        metrics["cited_task_coverage_ratio"] = (
            len(cited_task_ids) / len(active_tasks) if active_tasks else 1.0
        )
        answer_allowed = False
        if final_answer:
            answer_allowed = bool(
                (
                    cfg.allow_internal_knowledge_answer
                    and metrics["direct_answers"] > 0
                    and rounds_used == 0
                )
                or not cfg.require_sufficient_evidence_before_answer
                or self._has_sufficient_evidence(
                    latest_evaluation, task_statuses, active_tasks
                )
            )
        metrics["answer_allowed"] = 1.0 if answer_allowed else 0.0
        final_evidence_sufficient = self._has_sufficient_evidence(
            latest_evaluation, task_statuses, active_tasks
        )
        metrics["final_evidence_sufficient"] = 1.0 if final_evidence_sufficient else 0.0
        useful_fetched_pages = 0.0
        if final_answer and agent_ctx.fetched_pages:
            cited_urls = {
                result.url
                for result in agent_ctx.cited_results(final_answer)
                if result.url
            }
            useful_fetched_pages = float(
                sum(1 for page in agent_ctx.fetched_pages if page.url in cited_urls)
            )
        metrics["useful_fetched_pages"] = useful_fetched_pages
        metrics["unnecessary_fetch_count"] = max(
            0.0, metrics["fetched_pages"] - useful_fetched_pages
        )
        answered_directly = metrics["direct_answers"] > 0.0 and rounds_used == 0
        metrics["answer_when_evidence_insufficient"] = (
            1.0
            if (
                final_answer and not answered_directly and not final_evidence_sufficient
            )
            else 0.0
        )
        search_limit = float(cfg.max_search_limit or 0)
        metrics["search_budget_exhausted_without_answer"] = (
            1.0
            if (search_limit > 0 and rounds_used >= search_limit and not final_answer)
            else 0.0
        )

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=all_response_ids,
            response_mask=self.build_response_mask(all_response_ids),
            num_turns=num_turns,
            metrics=metrics,
            request_id=request_id,
            context=agent_ctx,
            trajectory_messages=list(working_messages),
            action_trace="\n".join(action_trace_parts) if action_trace_parts else None,
            final_answer=final_answer,
        )
