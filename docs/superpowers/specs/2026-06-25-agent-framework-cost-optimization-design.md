# Agent Framework Cost Optimization Design

## Objective

Optimize the existing Agent Framework around Planner, Search Tool, Reranker Tool,
Evidence Judge, and Answer Generator by reducing low-value search/rerank work
without changing the evidence safety rails or broad loop architecture.

This is a follow-on to the implemented modular agent framework. The current code
already supports retriever selection, per-search reranking, evidence scoring, and
GRPO reward metrics. This pass keeps that architecture and adds conservative
runtime controls plus clearer metrics so training and serving can distinguish
useful actions from skipped or redundant ones.

## Assumptions

- "Optimize" means latency/cost first, with only small architecture hardening
  where it directly supports the cost-saving behavior.
- The policy remains tag-based; no new model head or dependency is introduced.
- Evidence sufficiency gating remains unchanged. The agent should still be
  blocked from unsupported answers under the existing rules.
- Reranking remains opt-in via the existing per-search `rerank="true"` flag.
- Existing reward defaults stay backward compatible.

## Proposed Approach

Use a cost-aware loop optimization rather than a broad refactor.

1. Add conservative rerank gating.
   - If rerank is requested but there are no results, skip it.
   - If rerank is requested but every query in the round returns fewer than two
     results, skip it because reordering cannot materially help.
   - Count requested, executed, and skipped rerank actions separately.

2. Strengthen repeated-query prevention.
   - Keep the existing exact query tracking.
   - Add normalized-query tracking for repeated-search detection, using a small
     local normalizer that trims whitespace and folds repeated internal spaces.
   - Avoid semantic expansion or fuzzy matching in this pass.

3. Preserve answer and evidence behavior.
   - Do not change `SearchResultEvaluator` thresholds.
   - Do not change answer rejection behavior.
   - Do not change citation formatting or Answer Generator behavior.

4. Add focused tests.
   - Rerank requested with no docs: skipped, no reranker call.
   - Rerank requested with too few docs: skipped, no reranker call.
   - Rerank requested with enough docs: reranker runs and metrics update.
   - Repeated query with normalized whitespace: blocked as repeated.

## Component Impact

### Planner

No behavior change. The existing parser continues to recognize search actions,
retriever attributes, answer tags, and rerank requests.

### Search Tool

No public API change. The loop may use a normalized key before dispatching
searches, but backend selection and degradation behavior stay as they are.

### Reranker Tool

Reranking remains an injected callable. The loop should decide whether the
callable is worth invoking for a given result set before calling it. This keeps
the tool simple and avoids hiding cost policy inside the reranker.

### Evidence Judge

No behavior change. Existing evidence score and sufficiency metrics remain the
quality signal used by reward shaping.

### Answer Generator

No behavior change. Citation labels must continue to match the document order
shown to the model. Rerank gating must happen before labels are added to the
round context, as it does today for executed reranks.

## Metrics

Add or clarify these loop metrics:

- `rerank_requested`: count of search rounds that asked for reranking.
- `rerank_calls`: count of search rounds where reranking actually ran, preserving
  the existing flat per-action reward cost semantics.
- `rerank_skipped`: count of requested reranks skipped by cost-aware gating.
- `repeated_search_queries`: continue to count blocked duplicate queries,
  including normalized duplicates.

Reward code should keep reading `rerank_calls` for actual cost. The skipped
metric is observational, not a penalty by default.

## Testing Strategy

Primary verification:

```bash
pytest tests/unit/test_agent_loop.py -k "rerank or repeated" -v
pytest tests/unit/test_components.py -v
```

Regression verification:

```bash
pytest tests/unit/test_reward.py -k "rerank or retriever_aware" -v
```

No integration stack is required for this pass.

## Out of Scope

- Learned stop classifier.
- Reward-weight tuning from real training runs.
- Semantic duplicate detection.
- Broad rewiring of `SearchAgentLoop` onto all component classes.
- New retriever or reranker dependencies.

## Success Criteria

- Low-value rerank requests are skipped deterministically.
- Executed rerank cost is still measured by `rerank_calls`.
- Skipped reranks are visible in metrics.
- Normalized repeated queries are blocked without changing ordinary distinct
  queries.
- Existing evidence and answer-gating tests continue to pass.
