"""Tests for SearchRewardFunction and GRPO advantage computation."""

from __future__ import annotations

import asyncio

import pytest

from src.agent_loop import (
    AgentLoopOutput,
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchRewardConfig,
    SearchRewardFunction,
)
from src.agent_loop.context import AgentContext, SearchResult


# ---------------------------------------------------------------------------
# Helpers shared with other test files
# ---------------------------------------------------------------------------


class DummyTokenizer:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids if 0 <= i < 0x110000)


class DummyServer:
    def __init__(self, responses: list[list[int]]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def generate(self, request_id, prompt_ids, sampling_params):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


class FakeSearchClient:
    def __init__(self, responses: dict) -> None:
        self._responses = responses
        self.closed = False

    async def retrieve(self, queries, topk=None):
        return [self._responses.get(q, []) for q in queries]

    async def fetch_urls(self, urls):
        return []

    async def aclose(self):
        self.closed = True


def _exact_match(answer: str, ground_truth: str) -> float:
    return 1.0 if answer.strip() == ground_truth.strip() else 0.0


# ---------------------------------------------------------------------------
# AgentContext.cited_result_ids
# ---------------------------------------------------------------------------


def _make_ctx_with_results() -> AgentContext:
    ctx = AgentContext()
    results_r1q1 = [
        SearchResult(contents="doc A", score=0.9, url="https://example.com/a"),
        SearchResult(contents="doc B", score=0.7, url="https://example.com/b"),
    ]
    results_r1q2 = [
        SearchResult(contents="doc C", score=0.8, url="https://example.com/c"),
    ]
    ctx.add_round(
        queries=["query1", "query2"],
        results_by_query=[results_r1q1, results_r1q2],
    )
    # Round 2
    results_r2q1 = [
        SearchResult(contents="doc D", score=0.6, url="https://example.com/d")
    ]
    ctx.add_round(queries=["query3"], results_by_query=[results_r2q1])
    return ctx


def test_cited_result_ids_returns_valid_citations():
    ctx = _make_ctx_with_results()
    answer = "The answer cites [R1Q1D1] and [R2Q1D1] for evidence."
    cited = ctx.cited_result_ids(answer)
    assert "R1Q1D1" in cited
    assert "R2Q1D1" in cited
    assert len(cited) == 2


def test_cited_result_ids_ignores_out_of_range_doc_index():
    ctx = _make_ctx_with_results()
    # R1Q1 has 2 docs; D99 doesn't exist
    answer = "Evidence from [R1Q1D99] is cited."
    cited = ctx.cited_result_ids(answer)
    assert len(cited) == 0


def test_cited_result_ids_empty_answer():
    ctx = _make_ctx_with_results()
    assert ctx.cited_result_ids("") == frozenset()


def test_cited_result_ids_no_citations_in_answer():
    ctx = _make_ctx_with_results()
    assert ctx.cited_result_ids("No citations here at all.") == frozenset()


def test_agent_context_num_results():
    ctx = _make_ctx_with_results()
    # 2 + 1 + 1 = 4 results across all rounds
    assert ctx.num_results == 4


# ---------------------------------------------------------------------------
# SearchRewardFunction.compute — individual components
# ---------------------------------------------------------------------------


def _output_with_answer(answer: str | None, **metric_overrides) -> AgentLoopOutput:
    ctx = _make_ctx_with_results()
    metrics = {
        "search_rounds": 1.0,
        "rounds_used": 1.0,
        "fetched_pages": 0.0,
        "repeated_search_queries": 0.0,
        "subquestion_coverage_ratio": 1.0,
        "repeated_query_ratio": 0.0,
    }
    metrics.update(metric_overrides)
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        metrics=metrics,
        context=ctx,
        final_answer=answer,
    )


def test_reward_correctness_weight():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=2.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
        )
    )
    output = _output_with_answer("Paris")
    reward = rf.compute(output, ground_truth="Paris", judge_fn=_exact_match)
    # correctness=1.0 * 2.0 + coverage=1.0 * 0.0 + search_pen=-0.0 = 2.0
    assert pytest.approx(reward, abs=0.01) == 2.0


def test_reward_zero_when_no_answer():
    rf = SearchRewardFunction(
        SearchRewardConfig(citation_support_weight=0.0, subquestion_coverage_weight=0.0)
    )
    output = _output_with_answer(None)
    reward = rf.compute(output, ground_truth="Paris", judge_fn=_exact_match)
    # correctness=0, citation=0, coverage=0, no penalties → 0
    assert reward == pytest.approx(0.0, abs=0.01)


def test_reward_duplicate_query_penalty():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            duplicate_query_penalty=-0.1,
        )
    )
    output = _output_with_answer("x", repeated_search_queries=3.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.3, abs=0.001)


def test_reward_unnecessary_fetch_penalty():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            unnecessary_search_penalty=0.0,
            budget_penalty=0.0,
            fetch_usefulness_reward=0.0,
            unnecessary_fetch_penalty=-0.1,
        )
    )
    output = _output_with_answer("x", unnecessary_fetch_count=2.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.2, abs=0.001)


def test_reward_answer_when_evidence_insufficient_penalty():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            unnecessary_search_penalty=0.0,
            budget_penalty=0.0,
            fetch_usefulness_reward=0.0,
            answer_when_evidence_insufficient_penalty=-0.2,
        )
    )
    output = _output_with_answer("x", answer_when_evidence_insufficient=1.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.2, abs=0.001)


def test_reward_search_budget_exhausted_without_answer_penalty():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            unnecessary_search_penalty=0.0,
            budget_penalty=0.0,
            fetch_usefulness_reward=0.0,
            search_budget_exhausted_without_answer_penalty=-0.2,
        )
    )
    output = _output_with_answer(None, search_budget_exhausted_without_answer=1.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.2, abs=0.001)


def test_reward_unnecessary_search_penalty():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=-0.1,
        )
    )
    # 3 rounds → penalty for 2 extra rounds
    output = _output_with_answer("x", rounds_used=3.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.2, abs=0.001)


def test_reward_budget_penalty_fires_above_threshold():
    cfg = SearchRewardConfig(
        correctness_weight=0.0,
        citation_support_weight=0.0,
        subquestion_coverage_weight=0.0,
        unnecessary_search_penalty=0.0,
        budget_penalty_threshold=0.8,
        budget_penalty=-0.5,
        max_search_rounds=5,
    )
    rf = SearchRewardFunction(cfg)
    # 4/5 = 0.8 → exactly at threshold → penalty fires
    output = _output_with_answer("x", rounds_used=4.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.5, abs=0.001)


def test_reward_budget_penalty_not_fired_below_threshold():
    cfg = SearchRewardConfig(
        correctness_weight=0.0,
        citation_support_weight=0.0,
        subquestion_coverage_weight=0.0,
        unnecessary_search_penalty=0.0,
        budget_penalty_threshold=0.8,
        budget_penalty=-0.5,
        max_search_rounds=5,
    )
    rf = SearchRewardFunction(cfg)
    # 3/5 = 0.6 < 0.8 → no penalty
    output = _output_with_answer("x", rounds_used=3.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(0.0, abs=0.001)


def test_reward_citation_support():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=1.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    # ctx has 4 results; cite 2 of them → 2/4 = 0.5
    answer = "See [R1Q1D1] and [R1Q2D1] for details."
    output = _output_with_answer(answer)
    reward = rf.compute(output, ground_truth="anything", judge_fn=lambda a, g: 0.0)
    assert reward == pytest.approx(0.5, abs=0.01)


def test_reward_subquestion_coverage():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=1.0,
            unnecessary_search_penalty=0.0,
        )
    )
    # coverage_ratio = 0.5 (half the tasks sufficient)
    output = _output_with_answer("x", subquestion_coverage_ratio=0.5)
    reward = rf.compute(output, ground_truth="x", judge_fn=lambda a, g: 0.0)
    assert reward == pytest.approx(0.5, abs=0.01)


def test_reward_components_matches_compute():
    rf = SearchRewardFunction()
    output = _output_with_answer("Paris", rounds_used=2.0, repeated_search_queries=1.0)
    total = rf.compute(output, ground_truth="Paris", judge_fn=_exact_match)
    components = rf.reward_components(
        output, ground_truth="Paris", judge_fn=_exact_match
    )
    assert components["total"] == pytest.approx(total, abs=1e-9)


def test_reward_with_null_context():
    rf = SearchRewardFunction(SearchRewardConfig(citation_support_weight=0.5))
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        metrics={
            "rounds_used": 1.0,
            "repeated_search_queries": 0.0,
            "fetched_pages": 0.0,
            "subquestion_coverage_ratio": 1.0,
        },
        context=None,
        final_answer="Paris",
    )
    reward = rf.compute(output, ground_truth="Paris", judge_fn=_exact_match)
    # correctness=1.0*1.0=1.0, citation=0 (ctx None), coverage=1.0*0.2=0.2
    assert reward == pytest.approx(1.2, abs=0.01)


def test_fetch_reward_requires_citation_not_just_nonempty_answer():
    """fetch_usefulness_reward must only fire when fetched content is actually cited."""
    cfg = SearchRewardConfig(
        correctness_weight=0.0,
        citation_support_weight=0.0,
        subquestion_coverage_weight=0.0,
        unnecessary_search_penalty=0.0,
        fetch_usefulness_reward=0.5,
    )
    rf = SearchRewardFunction(cfg)
    # Answer is non-empty but cites nothing from the context
    output = _output_with_answer("An answer with no citations.", fetched_pages=2.0)
    output.context.record_fetched_pages(
        [SearchResult(contents="full page", url="https://example.com/a")]
    )
    reward = rf.compute(output, ground_truth="anything", judge_fn=lambda a, g: 0.0)
    assert reward == pytest.approx(0.0, abs=0.001)

    # Answer cites a retrieved result whose URL was fetched.
    output_with_cite = _output_with_answer(
        "[R1Q1D1] proves the answer.", fetched_pages=2.0
    )
    output_with_cite.context.record_fetched_pages(
        [SearchResult(contents="full page", url="https://example.com/a")]
    )
    reward_with_cite = rf.compute(
        output_with_cite, ground_truth="anything", judge_fn=lambda a, g: 0.0
    )
    assert reward_with_cite == pytest.approx(0.5, abs=0.001)


def test_fetch_reward_requires_overlap_between_fetched_and_cited_urls():
    cfg = SearchRewardConfig(
        correctness_weight=0.0,
        citation_support_weight=0.0,
        subquestion_coverage_weight=0.0,
        search_quality_weight=0.0,
        unnecessary_search_penalty=0.0,
        fetch_usefulness_reward=0.5,
    )
    rf = SearchRewardFunction(cfg)
    output = _output_with_answer("[R1Q1D1] cited", fetched_pages=1.0)
    output.context.record_fetched_pages(
        [SearchResult(contents="other page", url="https://example.com/z")]
    )
    reward = rf.compute(output, ground_truth="anything", judge_fn=lambda a, g: 0.0)
    assert reward == pytest.approx(0.0, abs=0.001)


def test_reward_search_quality_uses_evaluator_metrics():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=1.0,
            unnecessary_search_penalty=0.0,
        )
    )
    output = _output_with_answer(
        "x",
        search_rounds=2.0,
        search_quality_score=0.5,
        final_evidence_sufficient=1.0,
    )
    reward = rf.compute(output, ground_truth="x", judge_fn=lambda a, g: 0.0)
    assert reward == pytest.approx(0.75, abs=0.001)


# ---------------------------------------------------------------------------
# GRPO advantage computation
# ---------------------------------------------------------------------------


def test_advantages_normalised_within_group():
    rf = SearchRewardFunction()
    rewards = [1.0, 3.0, 2.0]
    group_ids = ["g1", "g1", "g1"]
    adv = rf.compute_batch_advantages(rewards, group_ids)
    # Within-group: mean=2, std=sqrt(2/3)≈0.816
    assert pytest.approx(adv[0], abs=0.01) == (1.0 - 2.0) / (0.8165 + 1e-8)
    assert pytest.approx(adv[2], abs=0.01) == (2.0 - 2.0) / (0.8165 + 1e-8)


def test_advantages_across_groups_independent():
    rf = SearchRewardFunction()
    # Two separate groups — normalisation is per-group, not global.
    rewards = [10.0, 0.0, 5.0, 5.0]
    group_ids = ["g1", "g1", "g2", "g2"]
    adv = rf.compute_batch_advantages(rewards, group_ids)
    # g2 samples have identical rewards → std=0 → both advantages ≈ 0
    assert adv[2] == pytest.approx(0.0, abs=1e-6)
    assert adv[3] == pytest.approx(0.0, abs=1e-6)
    # g1 samples: mean=5, std=5 → adv[0] ≈ +1, adv[1] ≈ -1
    assert adv[0] == pytest.approx(1.0, abs=0.01)
    assert adv[1] == pytest.approx(-1.0, abs=0.01)


def test_advantages_single_sample_group_is_zero():
    rf = SearchRewardFunction()
    rewards = [5.0, 1.0, 3.0]
    group_ids = ["g1", "g2", "g1"]
    adv = rf.compute_batch_advantages(rewards, group_ids)
    # g2 has one sample → advantage 0
    assert adv[1] == 0.0


def test_advantages_length_mismatch_raises():
    rf = SearchRewardFunction()
    with pytest.raises(ValueError, match="same length"):
        rf.compute_batch_advantages([1.0, 2.0], ["g1"])


# ---------------------------------------------------------------------------
# Integration: final_answer captured by SearchAgentLoop
# ---------------------------------------------------------------------------


def test_search_agent_loop_captures_final_answer():
    tokenizer = DummyTokenizer()
    answer_text = "<answer>The capital is Paris.</answer>"
    encoded = tokenizer.encode(answer_text)

    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServer([encoded]),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            require_sufficient_evidence_before_answer=False,
            allow_internal_knowledge_answer=True,
        ),
    )
    loop._search_client = FakeSearchClient({})

    output = asyncio.run(
        loop.run([{"role": "user", "content": "What is the capital of France?"}], {})
    )

    assert output.final_answer is not None
    assert "Paris" in output.final_answer


def test_search_agent_loop_emits_derived_metrics():
    tokenizer = DummyTokenizer()
    answer_ids = tokenizer.encode("<answer>done</answer>")

    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServer([answer_ids]),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            require_sufficient_evidence_before_answer=False,
            allow_internal_knowledge_answer=True,
        ),
    )
    loop._search_client = FakeSearchClient({})

    output = asyncio.run(loop.run([{"role": "user", "content": "hello"}], {}))

    assert "repeated_query_ratio" in output.metrics
    assert "subquestion_coverage_ratio" in output.metrics
    assert "rounds_used" in output.metrics
