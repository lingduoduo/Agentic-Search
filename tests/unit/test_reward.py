"""Tests for SearchRewardFunction and GRPO advantage computation."""

from __future__ import annotations

import asyncio

import pytest

from src import (
    AgentLoopOutput,
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchRewardConfig,
    SearchRewardFunction,
)
from src import (
    AgentContext,
    SearchResult,
    normalize_answer_text,
    simple_sparse_correctness_reward,
)
from src.model.post_training.reward import format_compliance_reward


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

    async def retrieve(self, queries, topk=None, filters=None):
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


def test_normalize_answer_text_lowercases_and_collapses_whitespace():
    assert normalize_answer_text("  John   Hopfield \n") == "john hopfield"


def test_simple_sparse_correctness_reward_exact_match_gets_one():
    assert simple_sparse_correctness_reward(
        " John Hopfield ", "john hopfield"
    ) == pytest.approx(1.0)


def test_simple_sparse_correctness_reward_contains_gold_gets_partial_credit():
    assert simple_sparse_correctness_reward(
        "The winner was John Hopfield and Geoffrey Hinton.",
        "John Hopfield",
    ) == pytest.approx(0.7)


def test_simple_sparse_correctness_reward_non_match_gets_zero():
    assert simple_sparse_correctness_reward("Paris", "London") == pytest.approx(0.0)


def test_simple_sparse_with_search_penalty_matches_first_pass_formula():
    rf = SearchRewardFunction(
        SearchRewardConfig.simple_sparse_with_search_penalty(per_search_penalty=-0.02)
    )
    output = _output_with_answer("Paris", rounds_used=3.0)

    reward = rf.compute(output, "Paris", simple_sparse_correctness_reward)
    components = rf.reward_components(output, "Paris", simple_sparse_correctness_reward)

    assert reward == pytest.approx(1.0 - 0.02 * 3.0)
    assert components["terminal_reward"] == pytest.approx(1.0)
    assert components["per_search_penalty"] == pytest.approx(-0.06)
    assert components["total"] == pytest.approx(0.94)


def test_simple_sparse_builder_disables_complex_second_pass_terms():
    cfg = SearchRewardConfig.simple_sparse_with_search_penalty()
    assert cfg.citation_support_weight == 0.0
    assert cfg.search_quality_weight == 0.0
    assert cfg.duplicate_query_penalty == 0.0
    assert cfg.fetch_usefulness_reward == 0.0


# ---------------------------------------------------------------------------
# Phase 2 reward: second_pass preset + unsupported_claim_penalty
# ---------------------------------------------------------------------------


def test_second_pass_config_enables_exactly_the_phase2_terms():
    cfg = SearchRewardConfig.second_pass()
    assert cfg.per_search_penalty == pytest.approx(-0.02)
    assert cfg.citation_support_weight == pytest.approx(0.1)
    assert cfg.unsupported_claim_penalty == pytest.approx(-0.1)
    assert cfg.duplicate_query_penalty == pytest.approx(-0.05)
    # Everything else off
    assert cfg.subquestion_coverage_weight == 0.0
    assert cfg.search_quality_weight == 0.0
    assert cfg.unnecessary_search_penalty == 0.0
    assert cfg.fetch_usefulness_reward == 0.0


def test_second_pass_reward_matches_formula():
    """Phase 2 total = correctness - 0.02*searches + 0.1*citation - 0.05*dup_queries."""
    rf = SearchRewardFunction(SearchRewardConfig.second_pass())
    ctx = _make_ctx_with_results()
    # Cite one of the four retrieved results in the answer.
    answer = "The answer cites [R1Q1D1] for evidence."
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=2,
        metrics={
            "rounds_used": 2.0,
            "repeated_search_queries": 1.0,
            "subquestion_coverage_ratio": 1.0,
        },
        context=ctx,
        final_answer=answer,
    )
    components = rf.reward_components(output, "anything", lambda a, g: 1.0)

    # correctness = 1.0 * 1.0 = 1.0
    assert components["correctness"] == pytest.approx(1.0)
    # search penalty = -0.02 * 2 = -0.04
    assert components["per_search_penalty"] == pytest.approx(-0.04)
    # citation support = 1 cited / 4 total = 0.25; weight 0.1 → 0.025
    assert components["citation_support"] == pytest.approx(0.025, abs=0.001)
    # unsupported claim: answer DID cite → no penalty
    assert components["unsupported_claim_penalty"] == pytest.approx(0.0)
    # duplicate query: 1 repeated * -0.05 = -0.05
    assert components["duplicate_query_penalty"] == pytest.approx(-0.05)


def test_unsupported_claim_penalty_fires_when_searched_but_no_citation():
    """Agent searched, got results, but cited nothing — penalty should fire."""
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
            unsupported_claim_penalty=-0.1,
        )
    )
    ctx = _make_ctx_with_results()  # has 4 retrieved results
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        metrics={"rounds_used": 1.0, "repeated_search_queries": 0.0},
        context=ctx,
        final_answer="An answer with no citations.",
    )
    components = rf.reward_components(output, "anything", lambda a, g: 1.0)
    assert components["unsupported_claim_penalty"] == pytest.approx(-0.1)


def test_unsupported_claim_penalty_does_not_fire_when_answer_cites_result():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
            unsupported_claim_penalty=-0.1,
        )
    )
    ctx = _make_ctx_with_results()
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        metrics={"rounds_used": 1.0, "repeated_search_queries": 0.0},
        context=ctx,
        final_answer="According to [R1Q1D1] the answer is clear.",
    )
    components = rf.reward_components(output, "anything", lambda a, g: 1.0)
    assert components["unsupported_claim_penalty"] == pytest.approx(0.0)


def test_unsupported_claim_penalty_does_not_fire_when_no_search_was_performed():
    """Agent answered directly without searching — no fabrication signal."""
    rf = SearchRewardFunction(SearchRewardConfig(unsupported_claim_penalty=-0.1))
    ctx = _make_ctx_with_results()
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        # rounds_used = 0 → agent did not search
        metrics={"rounds_used": 0.0, "repeated_search_queries": 0.0},
        context=ctx,
        final_answer="Direct answer with no citations.",
    )
    components = rf.reward_components(output, "anything", lambda a, g: 1.0)
    assert components["unsupported_claim_penalty"] == pytest.approx(0.0)


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


def test_format_compliance_reward_accepts_search_agent_citations():
    assert format_compliance_reward("Use alpha evidence [R1Q1D1].") == pytest.approx(
        1.0
    )


def test_third_pass_format_reward_accepts_search_agent_citations():
    rf = SearchRewardFunction(
        SearchRewardConfig.third_pass_with_format(
            correctness_weight=0.0,
            per_search_penalty=0.0,
            citation_support_weight=0.0,
            unsupported_claim_penalty=0.0,
            duplicate_query_penalty=0.0,
            format_reward_weight=0.1,
        )
    )
    output = _output_with_answer("Use alpha evidence [R1Q1D1].")

    components = rf.reward_components(output, "anything", lambda a, g: 0.0)

    assert components["format_reward"] == pytest.approx(0.1)


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


def test_sparse_final_only_reward_ignores_process_shaping():
    rf = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    output = _output_with_answer(
        "Paris",
        rounds_used=4.0,
        repeated_search_queries=3.0,
        search_quality_score=0.1,
        final_evidence_sufficient=0.0,
        subquestion_coverage_ratio=0.25,
        unnecessary_fetch_count=2.0,
        answer_when_evidence_insufficient=1.0,
    )
    total = rf.compute(output, ground_truth="Paris", judge_fn=_exact_match)
    components = rf.reward_components(
        output, ground_truth="Paris", judge_fn=_exact_match
    )

    assert total == pytest.approx(1.0, abs=1e-9)
    assert components["reward_mode"] == "sparse_final_only"
    assert components["terminal_reward"] == pytest.approx(1.0, abs=1e-9)
    assert components["shaping_total"] == pytest.approx(0.0, abs=1e-9)
    assert components["total"] == pytest.approx(1.0, abs=1e-9)


def test_sparse_final_only_builder_zeroes_non_terminal_weights():
    cfg = SearchRewardConfig.sparse_final_only(correctness_weight=2.5)

    assert cfg.reward_mode == "sparse_final_only"
    assert cfg.correctness_weight == pytest.approx(2.5)
    assert cfg.citation_support_weight == 0.0
    assert cfg.subquestion_coverage_weight == 0.0
    assert cfg.search_quality_weight == 0.0
    assert cfg.unnecessary_search_penalty == 0.0
    assert cfg.duplicate_query_penalty == 0.0
    assert cfg.budget_penalty == 0.0
    assert cfg.unnecessary_fetch_penalty == 0.0
    assert cfg.answer_when_evidence_insufficient_penalty == 0.0
    assert cfg.search_budget_exhausted_without_answer_penalty == 0.0
    assert cfg.fetch_usefulness_reward == 0.0


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


# ---------------------------------------------------------------------------
# compute_token_rewards — sparse reward at the final action token
# ---------------------------------------------------------------------------


def _output_with_tokens(
    answer: str | None,
    response_ids: list[int],
    response_mask: list[int],
    **metric_overrides,
) -> AgentLoopOutput:
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
        response_ids=response_ids,
        response_mask=response_mask,
        num_turns=1,
        metrics=metrics,
        context=None,
        final_answer=answer,
    )


def test_token_rewards_placed_at_last_action_token():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    # mask: search(1,1), obs(0,0), answer(1,1) — reward goes at index 5
    output = _output_with_tokens("Paris", [10, 11, 0, 0, 20, 21], [1, 1, 0, 0, 1, 1])
    token_rewards = rf.compute_token_rewards(output, "Paris", _exact_match)
    assert len(token_rewards) == 6
    assert token_rewards[5] == pytest.approx(1.0)
    assert all(r == 0.0 for r in token_rewards[:5])


def test_token_rewards_zero_reward_all_zeros():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    output = _output_with_tokens("Wrong", [10, 11, 20, 21], [1, 1, 1, 1])
    token_rewards = rf.compute_token_rewards(output, "Paris", _exact_match)
    assert all(r == 0.0 for r in token_rewards)


def test_token_rewards_empty_response_returns_empty():
    rf = SearchRewardFunction()
    output = _output_with_tokens("Paris", [], [])
    assert rf.compute_token_rewards(output, "Paris", _exact_match) == []


def test_token_rewards_no_mask_falls_back_to_last_token():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    output = _output_with_tokens("Paris", [1, 2, 3], [])
    token_rewards = rf.compute_token_rewards(output, "Paris", _exact_match)
    assert token_rewards[2] == pytest.approx(1.0)
    assert token_rewards[0] == 0.0
    assert token_rewards[1] == 0.0


def test_token_rewards_all_zero_mask_falls_back_to_last_token():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    output = _output_with_tokens("Paris", [1, 2, 3], [0, 0, 0])
    token_rewards = rf.compute_token_rewards(output, "Paris", _exact_match)
    assert token_rewards[2] == pytest.approx(1.0)


def test_token_rewards_sparse_mode_only_final_answer_contributes():
    rf = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    # Multi-turn: search tokens, then answer — shaping terms must not leak in
    output = _output_with_tokens(
        "Paris",
        [1, 2, 3, 4, 5],
        [1, 1, 0, 1, 1],
        rounds_used=3.0,
        repeated_search_queries=5.0,
        subquestion_coverage_ratio=0.1,
    )
    token_rewards = rf.compute_token_rewards(output, "Paris", _exact_match)
    assert token_rewards[4] == pytest.approx(1.0)
    assert all(r == 0.0 for r in token_rewards[:4])


def test_compute_terminal_reward_ignores_shaping_even_in_shaped_mode():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            reward_mode="shaped",
            correctness_weight=1.0,
            citation_support_weight=0.3,
            subquestion_coverage_weight=0.2,
            search_quality_weight=0.15,
            unnecessary_search_penalty=-0.05,
            duplicate_query_penalty=-0.1,
        )
    )
    output = _output_with_tokens(
        "Paris",
        [1, 2, 3, 4, 5],
        [1, 1, 0, 1, 1],
        rounds_used=4.0,
        repeated_search_queries=5.0,
        subquestion_coverage_ratio=0.1,
    )

    assert rf.compute(output, "Paris", _exact_match) != pytest.approx(1.0)
    assert rf.compute_terminal_reward(output, "Paris", _exact_match) == pytest.approx(
        1.0
    )


def test_compute_sparse_token_rewards_uses_only_terminal_reward():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            reward_mode="shaped",
            correctness_weight=1.0,
            citation_support_weight=0.3,
            subquestion_coverage_weight=0.2,
            search_quality_weight=0.15,
            unnecessary_search_penalty=-0.05,
            duplicate_query_penalty=-0.1,
        )
    )
    output = _output_with_tokens(
        "Paris",
        [1, 2, 3, 4, 5],
        [1, 1, 0, 1, 1],
        rounds_used=4.0,
        repeated_search_queries=5.0,
        subquestion_coverage_ratio=0.1,
    )

    token_rewards = rf.compute_sparse_token_rewards(output, "Paris", _exact_match)
    assert token_rewards[4] == pytest.approx(1.0)
    assert all(r == 0.0 for r in token_rewards[:4])


def test_advantages_normalised_within_group():
    rf = SearchRewardFunction()
    rewards = [1.0, 3.0, 2.0]
    group_ids = ["g1", "g1", "g1"]
    outcome_adv = rf.compute_grpo_outcome_advantages(rewards, group_ids)
    adv = rf.compute_batch_advantages(rewards, group_ids)
    assert outcome_adv == pytest.approx([-1.0, 1.0, 0.0], abs=1e-9)
    # Within-group: mean=2, std=sqrt(2/3)≈0.816
    assert pytest.approx(adv[0], abs=0.01) == (1.0 - 2.0) / (0.8165 + 1e-8)
    assert pytest.approx(adv[2], abs=0.01) == (2.0 - 2.0) / (0.8165 + 1e-8)


def test_outcome_advantages_are_computed_without_value_model():
    rf = SearchRewardFunction()
    rewards = [1.0, 0.7, 0.1]
    group_ids = ["g1", "g1", "g1"]
    adv = rf.compute_grpo_outcome_advantages(rewards, group_ids)
    assert adv[0] == pytest.approx(0.4)
    assert adv[1] == pytest.approx(0.1)
    assert adv[2] == pytest.approx(-0.5)


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
    outcome_adv = rf.compute_grpo_outcome_advantages(rewards, group_ids)
    adv = rf.compute_batch_advantages(rewards, group_ids)
    # g2 has one sample → advantage 0
    assert outcome_adv[1] == 0.0
    assert adv[1] == 0.0


def test_advantages_length_mismatch_raises():
    rf = SearchRewardFunction()
    with pytest.raises(ValueError, match="same length"):
        rf.compute_batch_advantages([1.0, 2.0], ["g1"])


# ---------------------------------------------------------------------------
# assign_grpo_outcome_token_advantages — end-to-end GRPO token vectors
# ---------------------------------------------------------------------------


def test_grpo_token_advantages_group_normalised_at_last_action_token():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    # Three rollouts from the same prompt; rewards: 1.0, 0.7, 0.1 → mean=0.6
    outputs = [
        _output_with_tokens("Paris", [1, 2, 3], [1, 1, 1]),  # reward 1.0
        _output_with_tokens("Paris", [1, 2, 3], [1, 1, 1]),  # reward 1.0
        _output_with_tokens("Wrong", [1, 2, 3], [1, 1, 1]),  # reward 0.0
    ]
    group_ids = ["g1", "g1", "g1"]
    ground_truths = ["Paris", "Paris", "Paris"]
    result = rf.assign_grpo_outcome_token_advantages(
        outputs, ground_truths, _exact_match, group_ids
    )

    assert len(result) == 3
    # All non-last positions must be 0
    for vec in result:
        assert len(vec) == 3
        assert vec[0] == 0.0
        assert vec[1] == 0.0
    # Rewards: 1.0, 1.0, 0.0 → mean=2/3, std>0 → first two positive, last negative
    assert result[0][2] > 0.0
    assert result[1][2] > 0.0
    assert result[2][2] < 0.0


def test_grpo_token_advantages_mask_determines_placement():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    # mask: search(1,1), obs(0,0), answer(1) — advantage goes to index 4
    o1 = _output_with_tokens("Paris", [10, 11, 0, 0, 20], [1, 1, 0, 0, 1])
    o2 = _output_with_tokens("Wrong", [10, 11, 0, 0, 20], [1, 1, 0, 0, 1])
    result = rf.assign_grpo_outcome_token_advantages(
        [o1, o2], ["Paris", "Paris"], _exact_match, ["g1", "g1"]
    )
    assert result[0][4] != 0.0
    assert result[1][4] != 0.0
    assert all(v == 0.0 for v in result[0][:4])
    assert all(v == 0.0 for v in result[1][:4])


def test_grpo_token_advantages_groups_normalised_independently():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    # g1: both wrong → std=0 → advantages 0.  g2: one right, one wrong → non-zero.
    outputs = [
        _output_with_tokens("Wrong", [1, 2], [1, 1]),
        _output_with_tokens("Wrong", [1, 2], [1, 1]),
        _output_with_tokens("Paris", [1, 2], [1, 1]),
        _output_with_tokens("Wrong", [1, 2], [1, 1]),
    ]
    group_ids = ["g1", "g1", "g2", "g2"]
    ground_truths = ["Paris"] * 4
    result = rf.assign_grpo_outcome_token_advantages(
        outputs, ground_truths, _exact_match, group_ids
    )

    # g1: identical rewards → std=0 → both advantages 0
    assert result[0][1] == pytest.approx(0.0, abs=1e-6)
    assert result[1][1] == pytest.approx(0.0, abs=1e-6)
    # g2: one correct, one wrong → non-zero advantages with opposite signs
    assert result[2][1] > 0.0
    assert result[3][1] < 0.0


def test_grpo_token_advantages_single_sample_group_gives_zero_vector():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    output = _output_with_tokens("Paris", [1, 2, 3], [1, 1, 1])
    result = rf.assign_grpo_outcome_token_advantages(
        [output], ["Paris"], _exact_match, ["g1"]
    )
    assert result[0] == [0.0, 0.0, 0.0]


def test_grpo_token_advantages_empty_response_gives_empty_vector():
    rf = SearchRewardFunction()
    o1 = _output_with_tokens("Paris", [], [])
    o2 = _output_with_tokens("Wrong", [], [])
    result = rf.assign_grpo_outcome_token_advantages(
        [o1, o2], ["Paris", "Paris"], _exact_match, ["g1", "g1"]
    )
    assert result[0] == []
    assert result[1] == []


def test_grpo_token_advantages_length_mismatch_raises():
    rf = SearchRewardFunction()
    o = _output_with_tokens("Paris", [1], [1])
    with pytest.raises(ValueError, match="same length"):
        rf.assign_grpo_outcome_token_advantages(
            [o, o], ["Paris"], _exact_match, ["g1", "g1"]
        )


# ---------------------------------------------------------------------------
# compute_batch_sparse_token_rewards — batch API for the training loop
# ---------------------------------------------------------------------------


def test_compute_batch_sparse_token_rewards_matches_individual_calls():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    outputs = [
        _output_with_tokens("Paris", [1, 2, 3, 4], [1, 1, 0, 1]),
        _output_with_tokens("Wrong", [1, 2, 3], [1, 1, 1]),
        _output_with_tokens("Paris", [], []),
    ]
    ground_truths = ["Paris", "Paris", "Paris"]

    batch = rf.compute_batch_sparse_token_rewards(outputs, ground_truths, _exact_match)
    individual = [
        rf.compute_sparse_token_rewards(o, g, _exact_match)
        for o, g in zip(outputs, ground_truths)
    ]

    assert batch == individual


def test_compute_batch_sparse_token_rewards_length_mismatch_raises():
    rf = SearchRewardFunction()
    o = _output_with_tokens("Paris", [1], [1])
    with pytest.raises(ValueError, match="same length"):
        rf.compute_batch_sparse_token_rewards([o], ["Paris", "extra"], _exact_match)


def test_compute_batch_sparse_token_rewards_uses_batch_judge_fn():
    """batch_judge_fn should be called once, not once per rollout."""
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    call_count = {"n": 0}

    def batch_judge(answers: list[str], gts: list[str]) -> list[float]:
        call_count["n"] += 1
        return [1.0 if a.strip() == g.strip() else 0.0 for a, g in zip(answers, gts)]

    outputs = [
        _output_with_tokens("Paris", [1, 2], [1, 1]),
        _output_with_tokens("Wrong", [1, 2], [1, 1]),
        _output_with_tokens("Paris", [1, 2], [1, 1]),
    ]
    result = rf.compute_batch_sparse_token_rewards(
        outputs, ["Paris", "Paris", "Paris"], _exact_match, batch_judge_fn=batch_judge
    )

    assert call_count["n"] == 1, "batch_judge_fn must be called exactly once"
    assert result[0][1] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(0.0)
    assert result[2][1] == pytest.approx(1.0)


def test_grpo_advantages_use_terminal_reward_not_shaped():
    """assign_grpo_outcome_token_advantages must use terminal reward only.

    When the reward config includes shaping terms, the GRPO advantage
    computation must still use only the terminal (judge) score for group
    normalisation — not the full shaped reward.
    """
    rf = SearchRewardFunction(
        SearchRewardConfig(
            reward_mode="shaped",
            correctness_weight=1.0,
            citation_support_weight=0.3,
            subquestion_coverage_weight=0.2,
            search_quality_weight=0.15,
            unnecessary_search_penalty=-0.05,
            duplicate_query_penalty=-0.1,
        )
    )
    # One correct, one wrong — heavy penalties on the correct rollout.
    # If shaping were included, the rewards would be inverted.
    outputs = [
        _output_with_tokens(
            "Paris", [1, 2], [1, 1], rounds_used=4.0, repeated_search_queries=5.0
        ),
        _output_with_tokens("Wrong", [1, 2], [1, 1]),
    ]
    result = rf.assign_grpo_outcome_token_advantages(
        outputs, ["Paris", "Paris"], _exact_match, ["g1", "g1"]
    )

    # Terminal reward for "Paris" = 1.0, for "Wrong" = 0.0 → "Paris" has higher advantage
    assert result[0][1] > result[1][1]


def test_grpo_advantages_batch_judge_fn_called_once():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=1.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            unnecessary_search_penalty=0.0,
        )
    )
    call_count = {"n": 0}

    def batch_judge(answers: list[str], gts: list[str]) -> list[float]:
        call_count["n"] += 1
        return [1.0 if a.strip() == g.strip() else 0.0 for a, g in zip(answers, gts)]

    outputs = [
        _output_with_tokens("Paris", [1, 2], [1, 1]),
        _output_with_tokens("Wrong", [1, 2], [1, 1]),
    ]
    rf.assign_grpo_outcome_token_advantages(
        outputs,
        ["Paris", "Paris"],
        _exact_match,
        ["g1", "g1"],
        batch_judge_fn=batch_judge,
    )

    assert call_count["n"] == 1


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


# ---------------------------------------------------------------------------
# Phase C — retriever_cost / rerank_cost / evidence_gain (T-C.1)
# ---------------------------------------------------------------------------


def test_new_action_terms_default_to_zero():
    """With default config the new terms contribute nothing, even if metrics exist."""
    rf = SearchRewardFunction(SearchRewardConfig())
    output = _output_with_answer(
        "ans [R1Q1D1]",
        web_searches=2.0,
        vdb_searches=3.0,
        rerank_calls=4.0,
        evidence_gain_total=0.9,
    )
    components = rf.reward_components(output, "anything", lambda a, g: 1.0)

    assert components["retriever_cost"] == 0.0
    assert components["rerank_cost"] == 0.0
    assert components["evidence_gain"] == 0.0


def test_retriever_cost_prices_web_higher_than_vdb():
    rf = SearchRewardFunction(
        SearchRewardConfig(retriever_cost_vdb=-0.01, retriever_cost_web=-0.05)
    )
    output = _output_with_answer("ans", web_searches=1.0, vdb_searches=2.0)
    components = rf.reward_components(output, "anything", lambda a, g: 0.0)

    assert components["retriever_cost"] == pytest.approx(-0.05 * 1 + -0.01 * 2)


def test_rerank_cost_is_flat_per_call():
    rf = SearchRewardFunction(SearchRewardConfig(rerank_cost=-0.02))
    output = _output_with_answer("ans", rerank_calls=3.0)
    components = rf.reward_components(output, "anything", lambda a, g: 0.0)

    assert components["rerank_cost"] == pytest.approx(-0.06)


def test_evidence_gain_rewards_improvement():
    rf = SearchRewardFunction(SearchRewardConfig(evidence_gain_weight=0.5))
    output = _output_with_answer("ans", evidence_gain_total=0.4)
    components = rf.reward_components(output, "anything", lambda a, g: 0.0)

    assert components["evidence_gain"] == pytest.approx(0.2)


def test_existing_presets_unaffected_by_new_terms():
    """Presets leave the new action terms at zero (byte-stable shaping)."""
    output = _output_with_answer(
        "ans [R1Q1D1]",
        web_searches=1.0,
        vdb_searches=1.0,
        rerank_calls=1.0,
        evidence_gain_total=0.5,
    )
    for cfg in (
        SearchRewardConfig.sparse_final_only(),
        SearchRewardConfig.second_pass(),
        SearchRewardConfig.third_pass_with_format(),
    ):
        components = SearchRewardFunction(cfg).reward_components(
            output, "anything", lambda a, g: 1.0
        )
        assert components["retriever_cost"] == 0.0
        assert components["rerank_cost"] == 0.0
        assert components["evidence_gain"] == 0.0


def test_retriever_aware_preset_enables_action_terms():
    """Phase C preset prices web 5× the VDB and enables rerank/evidence-gain terms."""
    cfg = SearchRewardConfig.retriever_aware()

    assert cfg.retriever_cost_web == pytest.approx(5 * cfg.retriever_cost_vdb)
    assert cfg.retriever_cost_web < 0.0
    assert cfg.rerank_cost < 0.0
    assert cfg.evidence_gain_weight > 0.0
    # Inherits second-pass shaping (search efficiency + citation signals).
    assert cfg.per_search_penalty != 0.0
    assert cfg.citation_support_weight != 0.0


def test_early_stop_bonus_zero_by_default_is_inert():
    """Default config leaves early_stop_bonus at zero even if the metric exists."""
    rf = SearchRewardFunction(SearchRewardConfig())
    output = _output_with_answer("ans", early_stops=3.0)
    components = rf.reward_components(output, "anything", lambda a, g: 1.0)

    assert SearchRewardConfig().early_stop_bonus == 0.0
    assert components["early_stop_bonus"] == 0.0


def test_early_stop_bonus_rewards_early_stops_metric():
    rf = SearchRewardFunction(SearchRewardConfig(early_stop_bonus=0.1))
    output = _output_with_answer("ans", early_stops=2.0)
    components = rf.reward_components(output, "anything", lambda a, g: 0.0)

    assert components["early_stop_bonus"] == pytest.approx(0.2)


def test_retriever_aware_surfaces_early_stop_bonus():
    cfg = SearchRewardConfig.retriever_aware(early_stop_bonus=0.05)
    assert cfg.early_stop_bonus == 0.05
    # Default retriever_aware keeps it inert so existing runs are unchanged.
    assert SearchRewardConfig.retriever_aware().early_stop_bonus == 0.0


def test_existing_presets_leave_early_stop_bonus_zero():
    output = _output_with_answer("ans [R1Q1D1]", early_stops=2.0)
    for cfg in (
        SearchRewardConfig.sparse_final_only(),
        SearchRewardConfig.second_pass(),
        SearchRewardConfig.third_pass_with_format(),
    ):
        components = SearchRewardFunction(cfg).reward_components(
            output, "anything", lambda a, g: 1.0
        )
        assert components["early_stop_bonus"] == 0.0


# ---------------------------------------------------------------------------
# Task 9 — forced_final_answer_penalty
# ---------------------------------------------------------------------------


def test_forced_final_answer_penalty_applied():
    cfg = SearchRewardConfig()  # serving defaults
    assert cfg.forced_final_answer_penalty == -0.05
    pen = cfg.forced_final_answer_penalty * 1.0
    assert pen == -0.05


def test_zeroed_preset_zeroes_forced_penalty():
    cfg = SearchRewardConfig._zeroed()
    assert cfg.forced_final_answer_penalty == 0.0


def test_forced_final_answer_penalty_fires_when_metric_set():
    rf = SearchRewardFunction(
        SearchRewardConfig(
            correctness_weight=0.0,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            unnecessary_search_penalty=0.0,
            budget_penalty=0.0,
            fetch_usefulness_reward=0.0,
            forced_final_answer_penalty=-0.05,
        )
    )
    output = _output_with_answer("x", forced_final_answer=1.0)
    reward = rf.compute(output, ground_truth="x", judge_fn=_exact_match)
    assert reward == pytest.approx(-0.05, abs=0.001)


def test_forced_final_answer_penalty_zero_when_metric_absent():
    """Existing rollouts that omit forced_final_answer are unaffected."""
    rf = SearchRewardFunction(SearchRewardConfig())
    output = _output_with_answer("x")
    components = rf.reward_components(output, "x", lambda a, g: 1.0)
    assert components["forced_final_answer_penalty"] == pytest.approx(0.0)
