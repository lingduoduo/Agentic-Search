# Optimize Agentic RAG — Best-in-Class Search & Answer Quality

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve end-to-end answer quality of the Agentic RAG pipeline by adding step-back prompting, seen-query deduplication, structured gap analysis, MMR diversity filtering, a stronger answer synthesis prompt, paragraph-aware semantic chunking, and runtime citation grounding verification.

**Architecture:** Seven targeted changes across the pipeline — `index_builder` (chunking) → `QueryEnhancer` → `AgenticRAGLoop` → `HybridRetriever` → `build_answer_prompt` → `GroundingVerifier` (post-generation) — each independently verifiable.

**Tech Stack:** Python 3.11+, pytest, unittest.mock; no new dependencies required.

---

## File Map

| File | Change |
|------|--------|
| `src/context/query_enhancer.py` | Add `step_back()` method; extend `QueryBundle` with `step_back_query` field |
| `src/agents/agentic_rag.py` | Add seen-query tracking; replace free-form follow-up with structured gap analysis |
| `src/retrieval/hybrid_retriever.py` | Add `mmr_lambda` config + `maximal_marginal_relevance()` post-fusion step |
| `src/context/prompts.py` | Improve `build_answer_prompt` with multi-hop, uncertainty, and grounding rules |
| `src/retrieval/index_builder.py` | Add paragraph/section-aware `_split_paragraphs()`, `_split_sentences_in_paragraph()`, `_split_text_paragraphs()`; update `chunk_document()` to set `section_continuation` |
| `src/context/grounding.py` | New: `CitationVerdict`, `GroundingReport`, `GroundingVerifier` |
| `src/context/models.py` | Add `CitationVerdict`, `GroundingReport` dataclasses; add `grounding_report` field to `AnswerGenerationResult`; add `verify_grounding` flag to `AnswerGenerationRequest` |
| `src/context/pipeline.py` | Wire `GroundingVerifier` into `generate_answer()` when `verify_grounding=True` |
| `tests/unit/test_query_enhancer.py` | Tests for `step_back()` and updated `QueryBundle.all_queries()` |
| `tests/unit/test_agentic_rag.py` | Tests for seen-query dedup and structured gap analysis |
| `tests/unit/test_hybrid_retriever.py` | Tests for `maximal_marginal_relevance()` and MMR integration |

---

## Task 1: Step-Back Prompting in QueryEnhancer

Step-back prompting generates a more abstract version of the query (e.g. "What is vector search?" → "What are approximate nearest-neighbour algorithms?") to broaden retrieval coverage for deep or narrow questions. The broader query runs alongside sub-queries and HyDE text.

**Files:**
- Modify: `src/context/query_enhancer.py`
- Modify: `tests/unit/test_query_enhancer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_query_enhancer.py`:

```python
# ---------------------------------------------------------------------------
# step_back
# ---------------------------------------------------------------------------

def test_step_back_returns_broader_query():
    broader = "What are approximate nearest-neighbour search algorithms?"
    enhancer = QueryEnhancer(_llm(broader))
    result = enhancer.step_back("How does FAISS handle GPU indexing for billion-scale datasets?")
    assert result == broader


def test_step_back_returns_none_when_llm_none():
    enhancer = QueryEnhancer(llm=None)
    assert enhancer.step_back("any query") is None


def test_step_back_returns_none_on_llm_failure():
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("timeout")
    enhancer = QueryEnhancer(llm)
    assert enhancer.step_back("any query") is None


def test_step_back_returns_none_on_empty_response():
    enhancer = QueryEnhancer(_llm(""))
    assert enhancer.step_back("any query") is None


def test_query_bundle_includes_step_back_in_all_queries():
    b = QueryBundle(
        original="How does FAISS handle GPU indexing?",
        sub_queries=["FAISS GPU index", "FAISS IVF flat GPU"],
        hyde_text="FAISS supports GPU indexes via the faiss-gpu package.",
        step_back_query="What are approximate nearest-neighbour search algorithms?",
    )
    queries = b.all_queries()
    assert "What are approximate nearest-neighbour search algorithms?" in queries
    # step_back should not duplicate original or sub_queries
    assert queries.count("What are approximate nearest-neighbour search algorithms?") == 1


def test_enhance_includes_step_back():
    llm = MagicMock()
    llm.complete.side_effect = [
        "FAISS GPU index\nFAISS IVF GPU",   # decompose
        "FAISS is a library by Facebook.",   # hyde
        "What are ANN search algorithms?",   # step_back
    ]
    enhancer = QueryEnhancer(llm)
    bundle = enhancer.enhance("How does FAISS handle GPU indexing?")
    assert bundle.step_back_query == "What are ANN search algorithms?"
    assert "What are ANN search algorithms?" in bundle.all_queries()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_query_enhancer.py -k "step_back or step_back_query or includes_step_back" -v
```

Expected: FAIL with `AttributeError: 'QueryBundle' object has no attribute 'step_back_query'`

- [ ] **Step 3: Implement step-back prompting**

Edit `src/context/query_enhancer.py`. Add the prompt constant after `_HYDE_PROMPT`:

```python
_STEP_BACK_PROMPT = """Rewrite the following question as a broader, more general question that
would help retrieve background knowledge useful for answering the original.
Return only the rewritten question, no explanation.

Question: {query}
Broader question:""".strip()
```

Update `QueryBundle` to add the new field (after `hyde_text`):

```python
@dataclass
class QueryBundle:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    step_back_query: str | None = None

    def all_queries(self) -> list[str]:
        """Return deduplicated list: sub_queries first, then hyde_text, then step_back_query."""
        seen: set[str] = set()
        result: list[str] = []
        for q in self.sub_queries:
            if q not in seen:
                seen.add(q)
                result.append(q)
        if self.hyde_text and self.hyde_text not in seen:
            seen.add(self.hyde_text)
            result.append(self.hyde_text)
        if self.step_back_query and self.step_back_query not in seen:
            result.append(self.step_back_query)
        return result or [self.original]
```

Add `step_back()` method to `QueryEnhancer` (after `hyde()`):

```python
def step_back(self, query: str) -> str | None:
    """Generate a broader, more abstract version of the query for wider retrieval coverage.
    Returns None on failure or when no LLM is configured."""
    if self.llm is None:
        return None
    try:
        raw = _llm_text(
            self.llm.complete(
                [ChatMessage(role="user", content=_STEP_BACK_PROMPT.format(query=query))]
            )
        ).strip()
        return raw or None
    except Exception as exc:
        logger.warning("Step-back generation failed: %s", exc)
        return None
```

Update `enhance()` to call `step_back`:

```python
def enhance(self, query: str) -> QueryBundle:
    """Run decompose + HyDE + step-back and return a QueryBundle."""
    return QueryBundle(
        original=query,
        sub_queries=self.decompose(query),
        hyde_text=self.hyde(query),
        step_back_query=self.step_back(query),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_query_enhancer.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/context/query_enhancer.py tests/unit/test_query_enhancer.py
git commit -m "feat: add step-back prompting to QueryEnhancer for broader retrieval coverage"
```

---

## Task 2: Seen-Query Deduplication in AgenticRAGLoop

Currently each round's follow-up queries replace the previous queries with no memory of what was already tried. A follow-up may repeat an earlier query, wasting a retrieval call. Fix: track a `seen_queries` set across rounds and skip queries already attempted.

**Files:**
- Modify: `src/agents/agentic_rag.py`
- Modify: `tests/unit/test_agentic_rag.py`

- [ ] **Step 1: Write failing test**

Add to `tests/unit/test_agentic_rag.py`:

```python
# ---------------------------------------------------------------------------
# Seen-query deduplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_duplicate_retrieval_queries_across_rounds():
    """Follow-up queries that duplicate earlier queries should not trigger new retrievals."""
    bundle = _make_bundle(["d1"])
    # decompose, hyde, sufficiency round 1 → "no", follow-up repeats sub-query → loop breaks
    llm = _llm_responses(
        "unique sub-query",   # decompose
        "HyDE text",          # hyde
        "broader question",   # step_back
        "no",                 # sufficiency round 1
        "unique sub-query",   # follow-up returns a duplicate → should be skipped → empty → break
        "answer",             # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)
    retrieval_calls: list[str] = []

    async def _track_retrieve(query, **kwargs):
        retrieval_calls.append(query)
        return bundle

    with patch("src.agents.agentic_rag.retrieve_context", side_effect=_track_retrieve):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("what is FAISS?")

    # "unique sub-query" should only be retrieved once even though follow-up returns it again
    assert retrieval_calls.count("unique sub-query") == 1
    assert result.rounds_used >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_agentic_rag.py::test_no_duplicate_retrieval_queries_across_rounds -v
```

Expected: FAIL (duplicate query is currently retrieved twice)

- [ ] **Step 3: Add seen-query tracking to AgenticRAGLoop.run()**

In `src/agents/agentic_rag.py`, update `run()` to track seen queries and filter follow-ups:

```python
async def run(
    self,
    question: str,
    *,
    chat_history: list[ChatMessage] | None = None,
) -> AgenticRAGResult:
    bundle = self._enhancer.enhance(question)
    current_queries = bundle.all_queries()

    accumulated: dict[str, ContextDocument] = {}
    seen_queries: set[str] = set()
    rounds_used = 0

    for round_idx in range(self.config.max_rounds):
        rounds_used += 1
        novel_queries = [q for q in current_queries if q not in seen_queries]
        if not novel_queries:
            break
        seen_queries.update(novel_queries)

        for q in novel_queries:
            try:
                ctx = await retrieve_context(
                    q,
                    search_url=self.config.retrieval_url,
                    top_k=self.config.topk,
                )
                for doc in ctx.documents:
                    if doc.id not in accumulated:
                        accumulated[doc.id] = doc
            except Exception as exc:
                logger.warning("Retrieval failed for query %r: %s", q, exc)

        merged = SearchContextBundle(
            query=question, documents=list(accumulated.values())
        )

        is_last = round_idx == self.config.max_rounds - 1
        if not is_last:
            if self._is_sufficient(question, merged):
                break
            follow_ups = self._generate_followup(question, merged)
            novel_follow_ups = [q for q in follow_ups if q not in seen_queries]
            if not novel_follow_ups:
                break
            current_queries = novel_follow_ups

    gen_result = generate_answer(
        AnswerGenerationRequest(
            question=question,
            context=merged,
            chat_history=chat_history or [],
        ),
        llm=self.llm,
    )
    return AgenticRAGResult(
        answer=gen_result.answer,
        citations=gen_result.citations,
        rounds_used=rounds_used,
        context=merged,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_agentic_rag.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/agentic_rag.py tests/unit/test_agentic_rag.py
git commit -m "feat: add seen-query deduplication to AgenticRAGLoop to prevent redundant retrieval"
```

---

## Task 3: Structured Gap Analysis in AgenticRAGLoop

The current `_FOLLOWUP_PROMPT` asks the LLM to "generate follow-up queries" without reasoning about what information is specifically missing. Replace with a two-stage prompt: (1) explicitly identify information gaps, (2) generate one targeted query per gap. This produces more focused follow-ups.

**Files:**
- Modify: `src/agents/agentic_rag.py`
- Modify: `tests/unit/test_agentic_rag.py`

- [ ] **Step 1: Write failing test**

Add to `tests/unit/test_agentic_rag.py`:

```python
# ---------------------------------------------------------------------------
# Structured gap analysis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_follow_up_queries_do_not_duplicate_question():
    """_generate_followup should never return the original question verbatim."""
    bundle = _make_bundle(["d1"])
    original_question = "what is FAISS?"
    # Simulate LLM returning the original question as a follow-up
    llm = _llm_responses(
        "sub-query",
        "hyde",
        "broader",
        "no",
        f"gap: {original_question}\nquery: {original_question}",  # follow-up returns original
        "answer",
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)

    with patch("src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)):
        loop = AgenticRAGLoop(config, llm=llm)
        # Should not raise; original question is already seen → follow-ups filtered
        result = await loop.run(original_question)

    assert isinstance(result, AgenticRAGResult)
```

- [ ] **Step 2: Run test to verify it fails (or passes — verify behavior)**

```bash
pytest tests/unit/test_agentic_rag.py::test_follow_up_queries_do_not_duplicate_question -v
```

- [ ] **Step 3: Replace `_FOLLOWUP_PROMPT` with structured gap-analysis prompt**

In `src/agents/agentic_rag.py`, replace the `_FOLLOWUP_PROMPT` constant:

```python
_GAP_ANALYSIS_PROMPT = """You are analyzing whether retrieved documents fully answer a question.

Question: {question}

Retrieved context (first 1000 chars):
{context}

Step 1 — List the specific pieces of information the question requires but the context does NOT provide.
         Write each gap as a short phrase (e.g. "training cost of GPT-4").
         If nothing is missing, write "none".

Step 2 — For each gap, write one focused search query that would retrieve the missing information.
         Format: one query per line, no numbering, no extra text.
         Queries only — do not repeat the gap phrases.

Output format:
GAPS:
<gap 1>
<gap 2>

QUERIES:
<query 1>
<query 2>""".strip()
```

Update `_generate_followup()` to parse the structured output:

```python
def _generate_followup(
    self, question: str, context: SearchContextBundle
) -> list[str]:
    if self.llm is None:
        return []
    prompt = _GAP_ANALYSIS_PROMPT.format(
        question=question,
        context=context.to_context_text()[:1000],
    )
    try:
        raw = _llm_text(
            self.llm.complete([ChatMessage(role="user", content=prompt)])
        ).strip()
        return _parse_gap_queries(raw)
    except Exception as exc:
        logger.warning("Gap analysis failed: %s", exc)
        return []
```

Add `_parse_gap_queries()` helper after `_clean_line()`:

```python
def _parse_gap_queries(raw: str) -> list[str]:
    """Extract the QUERIES section from a structured gap-analysis response.

    Falls back to treating every non-empty line as a query when the
    structured format is absent (e.g. legacy LLM response).
    """
    if "QUERIES:" in raw:
        queries_section = raw.split("QUERIES:", 1)[1]
    elif "GAPS:" in raw:
        # Only gaps section, no queries — nothing useful
        return []
    else:
        queries_section = raw

    return [
        _clean_line(line)
        for line in queries_section.splitlines()
        if _clean_line(line)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_agentic_rag.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/agentic_rag.py tests/unit/test_agentic_rag.py
git commit -m "feat: replace free-form follow-up generation with structured gap analysis in AgenticRAGLoop"
```

---

## Task 4: MMR Diversity Filtering in HybridRetriever

After RRF fusion, the top-k documents may cluster around the same aspect of the question (e.g. five chunks from the same source document). Maximal Marginal Relevance (MMR) re-scores documents by balancing relevance against redundancy, promoting a diverse result set. This is applied as a post-processing step on the fused list.

MMR formula: `score(d) = λ · relevance(d) − (1−λ) · max_sim(d, already_selected)`

Since we only have scalar RRF scores (not embedding vectors), we use normalized RRF score as relevance and document-id prefix matching as a cheap proxy for similarity (documents from the same source have matching id prefixes in the demo corpus; this is configurable).

**Files:**
- Modify: `src/retrieval/hybrid_retriever.py`
- Modify: `tests/unit/test_hybrid_retriever.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_hybrid_retriever.py`:

```python
from src.retrieval.hybrid_retriever import maximal_marginal_relevance


# ---------------------------------------------------------------------------
# maximal_marginal_relevance
# ---------------------------------------------------------------------------


def test_mmr_lambda_one_preserves_relevance_order():
    """At lambda=1.0 MMR is pure relevance — order matches input."""
    results = [_result("a", 0.9), _result("b", 0.5), _result("c", 0.1)]
    out = maximal_marginal_relevance(results, topk=3, mmr_lambda=1.0)
    assert [r["document"]["id"] for r in out] == ["a", "b", "c"]


def test_mmr_lambda_zero_maximises_diversity():
    """At lambda=0.0 MMR picks items maximally distant from what's already selected."""
    # "a" and "b" share a source prefix "src-1"; "c" comes from "src-2"
    results = [
        {"document": {"id": "src-1-chunk-0", "contents": "text"}, "score": 0.9},
        {"document": {"id": "src-1-chunk-1", "contents": "text"}, "score": 0.8},
        {"document": {"id": "src-2-chunk-0", "contents": "text"}, "score": 0.3},
    ]
    out = maximal_marginal_relevance(results, topk=2, mmr_lambda=0.0)
    ids = [r["document"]["id"] for r in out]
    # First pick is always the highest-relevance doc
    assert ids[0] == "src-1-chunk-0"
    # Second pick should prefer diversity → src-2-chunk-0 over src-1-chunk-1
    assert ids[1] == "src-2-chunk-0"


def test_mmr_returns_topk_results():
    results = [_result(str(i), 1.0 / (i + 1)) for i in range(10)]
    out = maximal_marginal_relevance(results, topk=3, mmr_lambda=0.5)
    assert len(out) == 3


def test_mmr_returns_all_when_topk_exceeds_input():
    results = [_result("a", 0.9), _result("b", 0.5)]
    out = maximal_marginal_relevance(results, topk=10, mmr_lambda=0.5)
    assert len(out) == 2


def test_mmr_empty_input_returns_empty():
    assert maximal_marginal_relevance([], topk=5, mmr_lambda=0.5) == []


def test_hybrid_retriever_applies_mmr_when_configured():
    retriever = _make_hybrid(0.5)
    retriever._dense.retrieve.return_value = [
        [_result("a", 0.9), _result("b", 0.8), _result("c", 0.3)]
    ]
    retriever._sparse.retrieve.return_value = [
        [_result("b", 8.0), _result("c", 6.0), _result("d", 2.0)]
    ]
    retriever.config = retriever.config.__class__(
        **{**retriever.config.__dict__, "mmr_lambda": 0.7, "mmr_topk": 2}
    )
    results = retriever.retrieve(["q"])
    # MMR topk=2 → exactly 2 results
    assert len(results[0]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_hybrid_retriever.py -k "mmr" -v
```

Expected: FAIL with `ImportError: cannot import name 'maximal_marginal_relevance'`

- [ ] **Step 3: Implement MMR in hybrid_retriever.py**

Add `mmr_lambda` and `mmr_topk` to `HybridRetrieverConfig` (after `rrf_k`):

```python
@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Config for a retriever that fuses dense and sparse search.

    hybrid_alpha controls which backends are active:
        0.0  — pure BM25 (sparse only, no embedding computed)
        1.0  — pure dense (embedding only, BM25 index not loaded)
        0 < alpha < 1 — both run in parallel and results are fused with RRF

    mmr_lambda: 1.0 = pure relevance (no diversity), 0.5 = balanced (default).
    mmr_topk: number of results to return after MMR (None = return all fused results).
    """

    dense: DenseRetrieverConfig
    sparse: SparseRetrieverConfig | None = None
    hybrid_alpha: float = 0.5
    rrf_k: int = _RRF_K
    mmr_lambda: float = 1.0
    mmr_topk: int | None = None

    def validate(self) -> None:
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0.")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0.0 and 1.0.")
        if self.hybrid_alpha < 1.0 and self.sparse is None:
            raise ValueError(
                "sparse config is required when hybrid_alpha < 1.0 "
                "(pure-dense mode requires hybrid_alpha=1.0)."
            )
        self.dense.validate()
        if self.sparse is not None:
            self.sparse.validate()
```

Add `maximal_marginal_relevance()` function after `combine_retrieval_results()`:

```python
def _doc_source_prefix(doc_id: str) -> str:
    """Return the source-level prefix of a document id for similarity estimation.

    Convention used in the demo corpus: ids are 'source-name-chunk-N'.
    Prefix = everything before the last '-' separator.  When the id has no
    separator the full id is used (no similarity penalty applied).
    """
    sep = doc_id.rfind("-")
    return doc_id[:sep] if sep > 0 else doc_id


def maximal_marginal_relevance(
    results: list[dict[str, Any]],
    *,
    topk: int,
    mmr_lambda: float = 0.5,
) -> list[dict[str, Any]]:
    """Re-rank `results` with Maximal Marginal Relevance to balance relevance and diversity.

    Uses the RRF/retrieval score as relevance and source-prefix matching as a cheap
    proxy for inter-document similarity (no embeddings required).

    Args:
        results: Ranked list of {"document": dict, "score": float} items.
        topk: Maximum number of results to return.
        mmr_lambda: 1.0 = pure relevance order; 0.0 = maximum diversity.
    """
    if not results:
        return []
    if mmr_lambda == 1.0:
        return results[:topk]

    max_score = max(r["score"] for r in results) or 1.0
    normalized = [(r, r["score"] / max_score) for r in results]

    selected: list[dict[str, Any]] = []
    selected_prefixes: list[str] = []
    remaining = list(normalized)

    while remaining and len(selected) < topk:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda x: x[1])
        else:
            def mmr_score(item: tuple[dict[str, Any], float]) -> float:
                result, rel = item
                doc_id = str(result["document"].get("id", ""))
                prefix = _doc_source_prefix(doc_id)
                # Similarity = 1 if sharing source prefix with any selected, else 0
                sim = 1.0 if prefix in selected_prefixes else 0.0
                return mmr_lambda * rel - (1.0 - mmr_lambda) * sim

            best = max(remaining, key=mmr_score)

        result, _ = best
        selected.append(result)
        selected_prefixes.append(
            _doc_source_prefix(str(result["document"].get("id", "")))
        )
        remaining.remove(best)

    return selected
```

Update `HybridRetriever.retrieve()` to apply MMR after fusion:

```python
def retrieve(
    self,
    queries: list[str],
    topk: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Return one ranked result list per query, optionally MMR-diversified."""
    # Pure modes — single retriever, no fusion overhead.
    if self._sparse is None:
        assert self._dense is not None
        raw_results = self._dense.retrieve(queries, topk)
    elif self._dense is None:
        raw_results = self._sparse.retrieve(queries, topk)
    else:
        # Hybrid — dispatch both retrievers concurrently then fuse per query.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            dense_fut = pool.submit(self._dense.retrieve, queries, topk)
            sparse_fut = pool.submit(self._sparse.retrieve, queries, topk)
            dense_results = dense_fut.result()
            sparse_results = sparse_fut.result()

        raw_results = [
            combine_retrieval_results(
                [dense_results[i], sparse_results[i]],
                rrf_k=self.config.rrf_k,
            )
            for i in range(len(queries))
        ]

    # Apply MMR when lambda < 1.0 or an explicit mmr_topk is set.
    if self.config.mmr_lambda < 1.0 or self.config.mmr_topk is not None:
        mmr_topk = self.config.mmr_topk or (topk or len(raw_results[0] if raw_results else []))
        return [
            maximal_marginal_relevance(
                result_list,
                topk=mmr_topk,
                mmr_lambda=self.config.mmr_lambda,
            )
            for result_list in raw_results
        ]

    return raw_results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_hybrid_retriever.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/hybrid_retriever.py tests/unit/test_hybrid_retriever.py
git commit -m "feat: add MMR diversity filtering to HybridRetriever post-RRF fusion"
```

---

## Task 5: Improved Answer Synthesis Prompt

The current `build_answer_prompt` in `src/context/prompts.py` is minimal. It doesn't guide the LLM on how to handle multi-hop reasoning, conflicting evidence, or uncertainty. Add structured synthesis instructions that produce grounded, citation-accurate, uncertainty-aware answers.

**Files:**
- Modify: `src/context/prompts.py`
- Modify: `tests/unit/test_context_pipeline.py` (add prompt content assertions)

- [ ] **Step 1: Write failing test**

Read the existing test file first:

```bash
grep -n "build_answer_prompt\|build_chat_prompt" tests/unit/test_context_pipeline.py | head -20
```

Then add to `tests/unit/test_context_pipeline.py`:

```python
from src.context.prompts import build_answer_prompt
from src.context.models import AgentBehaviorConfig, SearchContextBundle, ContextDocument


def _make_bundle_for_prompt() -> SearchContextBundle:
    return SearchContextBundle(
        query="What is FAISS?",
        documents=[
            ContextDocument(
                id="D1",
                title="FAISS Overview",
                content="FAISS is a library for efficient similarity search.",
                score=0.9,
            )
        ],
    )


def test_answer_prompt_instructs_on_uncertainty():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    full_text = prompt.system + prompt.user
    assert "uncertain" in full_text.lower() or "missing" in full_text.lower()


def test_answer_prompt_instructs_on_citation_format():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    full_text = prompt.system + prompt.user
    assert "[D" in full_text


def test_answer_prompt_instructs_on_conflicting_evidence():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    assert "conflict" in prompt.system.lower() or "contradict" in prompt.system.lower() or "disagree" in prompt.system.lower()


def test_answer_prompt_forbids_fabrication():
    prompt = build_answer_prompt("What is FAISS?", _make_bundle_for_prompt())
    full_text = prompt.system + prompt.user
    assert "fabricat" in full_text.lower() or "not in the context" in full_text.lower() or "only" in full_text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_context_pipeline.py -k "answer_prompt" -v
```

Expected: FAIL on the `conflicting_evidence` assertion.

- [ ] **Step 3: Improve build_answer_prompt**

In `src/context/prompts.py`, replace `build_answer_prompt`:

```python
def build_answer_prompt(
    question: str,
    context: SearchContextBundle,
    config: AgentBehaviorConfig | None = None,
) -> PromptBundle:
    config = config or AgentBehaviorConfig()
    system = (
        "You are a retrieval-grounded research assistant.\n"
        f"{build_agent_behavior_prompt(config)}\n\n"
        "Synthesis rules:\n"
        "1. Base every claim on the retrieved context. Do not fabricate facts not present in the context.\n"
        "2. Cite each claim inline using the document label, e.g. [D1] or [D2].\n"
        "3. If the context contains conflicting or contradictory information, note the disagreement "
        "and cite both sides rather than choosing one silently.\n"
        "4. For multi-step questions, reason through each step explicitly before stating the conclusion.\n"
        "5. If the context is insufficient to answer fully, state exactly what information is missing "
        "rather than speculating."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context.to_context_text()}\n\n"
        "Answer using only the retrieved context. "
        "For anything not covered by the context, say: "
        "'The retrieved context does not contain information about [topic].'"
    )
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
    return PromptBundle(system=system, user=user, messages=messages)
```

- [ ] **Step 4: Run all unit tests to verify nothing regressed**

```bash
pytest tests/unit/ -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/context/prompts.py tests/unit/test_context_pipeline.py
git commit -m "feat: improve answer synthesis prompt with multi-hop, conflict, and uncertainty guidance"
```

---

## Task 7: Paragraph-Aware Semantic Chunking

Retrieval and generation operate at the chunk level, so chunk quality directly determines answer quality. The current `_split_sentences()` ([index_builder.py:988](src/retrieval/index_builder.py#L988)) calls `re.sub(r"\s+", " ", text)` which collapses double-newlines and markdown headers into single spaces — paragraph and section structure is completely lost before any chunking happens. A document with two unrelated sections can easily produce a chunk that spans both, making it semantically incoherent.

**Fix:** Add `_split_paragraphs()` that splits on `\n\n+` / `\n#` boundaries first, then sentence-pack within each paragraph using existing logic. Flush the current chunk at paragraph boundaries when it is ≥ 50% full. Also set `section_continuation=True` on every non-first chunk so consumers can distinguish section-start chunks from mid-section continuations.

**Files:**
- Modify: `src/retrieval/index_builder.py`
- Modify: `tests/unit/test_indexing_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_indexing_pipeline.py`:

```python
# ---------------------------------------------------------------------------
# Paragraph-aware chunking
# ---------------------------------------------------------------------------

from src.retrieval.index_builder import _split_paragraphs, _split_sentences_in_paragraph


def test_split_paragraphs_splits_on_double_newline():
    text = "First paragraph with sentences.\n\nSecond paragraph here."
    result = _split_paragraphs(text)
    assert len(result) == 2
    assert result[0] == "First paragraph with sentences."
    assert result[1] == "Second paragraph here."


def test_split_paragraphs_splits_on_markdown_header():
    text = "Intro text here.\n## Section Two\nSection two content."
    result = _split_paragraphs(text)
    assert len(result) == 2
    assert "Intro text" in result[0]
    assert "Section Two" in result[1] or "Section two" in result[1]


def test_split_paragraphs_single_paragraph_returns_one_item():
    text = "Just one sentence. And another. And a third."
    result = _split_paragraphs(text)
    assert len(result) == 1


def test_split_paragraphs_empty_returns_empty():
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   \n\n  ") == []


def test_split_sentences_in_paragraph_splits_on_punctuation():
    para = "First sentence. Second sentence! Third sentence?"
    result = _split_sentences_in_paragraph(para)
    assert len(result) == 3
    assert result[0] == "First sentence."


def test_split_sentences_in_paragraph_does_not_collapse_paragraphs():
    # This function should only be called on a single paragraph already
    para = "One sentence. Two sentence."
    result = _split_sentences_in_paragraph(para)
    assert len(result) == 2


def test_chunk_document_does_not_span_section_boundary():
    """Chunks should respect paragraph boundaries — no chunk should span two unrelated sections."""
    section_a = " ".join(["word"] * 60)   # 60 tokens — well past chunk_size=50
    section_b = " ".join(["term"] * 60)
    document = Document(
        id="doc-sections",
        title="Test",
        contents=f"{section_a}\n\n{section_b}",
        metadata={},
        permissions={},
    )
    chunks = chunk_document(
        document,
        ChunkingConfig(chunk_size=50, chunk_overlap=5, include_title=False, include_metadata=False),
    )
    for chunk in chunks:
        # A well-formed chunk should be dominated by one vocabulary (either "word" or "term")
        word_count = chunk.text.count("word")
        term_count = chunk.text.count("term")
        # Cross-section chunks have roughly equal counts; reject those
        if word_count > 0 and term_count > 0:
            # Allow a small overlap carry-over (up to overlap tokens = 5)
            assert min(word_count, term_count) <= 5, (
                f"Chunk spans sections: {word_count} 'word' tokens and {term_count} 'term' tokens"
            )


def test_chunk_document_sets_section_continuation_on_non_first_chunks():
    """section_continuation should be True for every chunk after the first."""
    document = Document(
        id="doc-cont",
        title="Title",
        contents=" ".join(["sentence."] * 30),
        metadata={},
        permissions={},
    )
    chunks = chunk_document(
        document,
        ChunkingConfig(chunk_size=10, chunk_overlap=2, include_title=False, include_metadata=False),
    )
    assert len(chunks) >= 2, "Need at least 2 chunks to test continuation flag"
    assert chunks[0].section_continuation is False
    for chunk in chunks[1:]:
        assert chunk.section_continuation is True, (
            f"chunk_id={chunk.chunk_id} should have section_continuation=True"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_indexing_pipeline.py -k "split_paragraphs or split_sentences_in_paragraph or section_boundary or section_continuation" -v
```

Expected: FAIL with `ImportError: cannot import name '_split_paragraphs'`

- [ ] **Step 3: Add paragraph-splitting helpers to index_builder.py**

In `src/retrieval/index_builder.py`, add three new functions after `_split_sentences()` (around line 996):

```python
def _split_paragraphs(text: str) -> list[str]:
    """Split text on paragraph and section boundaries without destroying internal whitespace.

    Splits on:
    - Two or more consecutive newlines (blank-line paragraph breaks)
    - A newline immediately followed by a markdown heading marker (#)

    Each returned string is one paragraph/section, stripped of leading/trailing
    whitespace, with internal whitespace preserved for sentence splitting.
    """
    if not text or not text.strip():
        return []
    parts = re.split(r'\n{2,}|\n(?=#)', text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences_in_paragraph(para: str) -> list[str]:
    """Sentence-split within a single paragraph.

    Unlike _split_sentences(), this does NOT collapse whitespace across
    the whole document — it only normalises spaces/tabs within the
    paragraph already handed to it.
    """
    normalized = re.sub(r'[ \t]+', ' ', para).strip()
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(r'(?<=[.!?。！？])\s+', normalized)
        if part.strip()
    ]


def _split_text_paragraphs(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Paragraph-aware chunking: flush at section boundaries when chunk is ≥ 50% full.

    Algorithm:
    1. Split document into paragraph/section units.
    2. Within each unit, sentence-split and pack sentences up to chunk_size.
    3. At a paragraph boundary, if the running chunk is already ≥ 50% full,
       flush it before starting the next paragraph.  This keeps chunks
       semantically coherent without introducing overly short chunks for
       documents with many short paragraphs.
    4. Falls back to token-window splitting for sentences that exceed chunk_size.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para_idx, para in enumerate(paragraphs):
        para_sentences = _split_sentences_in_paragraph(para)
        if not para_sentences:
            continue

        # Flush at paragraph boundary when chunk is ≥ 50% full.
        if current and current_tokens >= chunk_size // 2 and para_idx > 0:
            chunks.append(" ".join(current).strip())
            current = _overlap_tail(current, chunk_overlap)
            current_tokens = _token_count(" ".join(current))

        for sentence in para_sentences:
            sentence_tokens = _token_count(sentence)

            if sentence_tokens > chunk_size:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(_split_token_window(sentence, chunk_size, chunk_overlap))
                continue

            would_exceed = current and current_tokens + sentence_tokens > chunk_size
            if would_exceed:
                chunks.append(" ".join(current).strip())
                current = _overlap_tail(current, chunk_overlap)
                current_tokens = _token_count(" ".join(current))
                if current and current_tokens + sentence_tokens > chunk_size:
                    current = []
                    current_tokens = 0

            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]
```

- [ ] **Step 4: Update `_split_text()` to use the paragraph-aware path**

Replace `_split_text()` in `src/retrieval/index_builder.py`:

```python
def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks, respecting paragraph and section boundaries."""
    return _split_text_paragraphs(text, chunk_size, chunk_overlap)
```

- [ ] **Step 5: Set `section_continuation` in `chunk_document()`**

In `src/retrieval/index_builder.py`, update the `IndexChunk(...)` construction inside `chunk_document()` to pass `section_continuation`:

```python
chunks.append(
    IndexChunk(
        id=f"{document.id}::chunk-{chunk_id}",
        document_id=document.id,
        chunk_id=chunk_id,
        text=index_text,
        title=document.title,
        url=document.url,
        metadata=metadata,
        blurb=_extract_blurb(chunk_text, config.blurb_size),
        metadata_suffix_semantic=metadata_suffix_semantic,
        metadata_suffix_keyword=metadata_suffix_keyword,
        mini_chunk_texts=mini_chunk_texts,
        section_continuation=chunk_id > 0,
    )
)
```

- [ ] **Step 6: Export the new helpers from `__all__`**

In `src/retrieval/index_builder.py`, add to `__all__`:

```python
__all__ = [
    "ChunkingConfig",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "IndexChunk",
    "IndexingPipelineConfig",
    "IndexingPipelineResult",
    "IndexWriterConfig",
    "_split_paragraphs",
    "_split_sentences_in_paragraph",
    "chunk_document",
    "chunk_documents",
    "deterministic_embedding_fn",
    "embed_chunks",
    "embed_chunks_with_failure_handling",
    "filter_indexable_documents",
    "generate_large_chunks",
    "prepare_texts",
    "run_indexing_pipeline",
    "write_faiss_index",
]
```

- [ ] **Step 7: Run all tests to verify nothing regressed**

```bash
pytest tests/unit/test_indexing_pipeline.py -v
```

Expected: ALL PASS. The existing `test_chunk_document_splits_with_overlap_and_title` test must still pass — the input text there has no `\n\n` so `_split_text_paragraphs` falls through to identical single-paragraph packing behaviour.

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/index_builder.py tests/unit/test_indexing_pipeline.py
git commit -m "feat: add paragraph-aware semantic chunking to preserve section boundaries"
```

---

## Task 8: Runtime Citation Grounding Verifier

At inference time there is no check that a `[D1]` citation in the answer actually corresponds to a document in the context, or that the sentence making the claim has any lexical overlap with that document. The training-time `_unsupported_claim_penalty` in `reward.py` only signals during RL training — it has no effect at serving time.

This task adds a lightweight **post-generation grounding verifier** that:
1. Parses every `[Dx]` citation in the answer.
2. Checks whether the cited document exists in the `SearchContextBundle` (dangling citation detection).
3. For citations that do resolve, computes stopword-filtered lexical overlap between the citing sentence and the cited document content.
4. Returns a `GroundingReport` with per-citation verdicts and a clean answer with dangling citations stripped.
5. Integrates into `generate_answer()` as an opt-in flag (`verify_grounding=True`) — zero overhead when disabled.

No new dependencies. The lexical overlap check reuses the same stopword-filtered tokeniser already in `pipeline.py`.

**Files:**
- Create: `src/context/grounding.py`
- Modify: `src/context/models.py`
- Modify: `src/context/pipeline.py`
- Create: `tests/unit/test_grounding.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_grounding.py`:

```python
"""Unit tests for the citation grounding verifier."""
from __future__ import annotations

import pytest

from src.context.grounding import GroundingVerifier
from src.context.models import (
    CitationVerdict,
    ContextDocument,
    GroundingReport,
    SearchContextBundle,
)


def _bundle(*doc_contents: str) -> SearchContextBundle:
    docs = [
        ContextDocument(
            id=f"D{i}",
            title=f"Doc {i}",
            content=content,
            score=0.9,
        )
        for i, content in enumerate(doc_contents, 1)
    ]
    return SearchContextBundle(query="test", documents=docs)


# ---------------------------------------------------------------------------
# GroundingVerifier.verify — grounded citations
# ---------------------------------------------------------------------------


def test_verify_grounded_citation():
    bundle = _bundle("FAISS is a vector similarity search library developed by Facebook.")
    answer = "FAISS enables fast similarity search. [D1]"
    report = GroundingVerifier().verify(answer, bundle)
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.citation == "D1"
    assert v.document_found is True
    assert v.overlap_score > 0.0
    assert v.is_grounded is True


def test_verify_dangling_citation_not_in_context():
    bundle = _bundle("Some content about FAISS.")
    answer = "The model was trained on 1B parameters. [D9]"
    report = GroundingVerifier().verify(answer, bundle)
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.citation == "D9"
    assert v.document_found is False
    assert v.overlap_score == 0.0
    assert v.is_grounded is False


def test_verify_dangling_citation_stripped_from_answer_clean():
    bundle = _bundle("FAISS content.")
    answer = "The answer is 42. [D99] More text here."
    report = GroundingVerifier().verify(answer, bundle)
    assert "[D99]" not in report.answer_clean
    assert "The answer is 42." in report.answer_clean


def test_verify_valid_citation_not_stripped():
    bundle = _bundle("FAISS is a similarity search library.")
    answer = "FAISS enables fast search. [D1]"
    report = GroundingVerifier().verify(answer, bundle)
    assert "[D1]" in report.answer_clean


def test_verify_multiple_citations_mixed():
    bundle = _bundle(
        "Dense retrieval uses embeddings.",
        "Sparse retrieval uses BM25 term matching.",
    )
    answer = (
        "Dense retrieval uses embeddings for search. [D1] "
        "Sparse retrieval relies on term frequency. [D2] "
        "Magic happens at night. [D99]"
    )
    report = GroundingVerifier().verify(answer, bundle)
    citations = {v.citation: v for v in report.verdicts}
    assert citations["D1"].document_found is True
    assert citations["D2"].document_found is True
    assert citations["D99"].document_found is False
    assert "[D99]" not in report.answer_clean
    assert "[D1]" in report.answer_clean
    assert "[D2]" in report.answer_clean


def test_grounding_rate_all_grounded():
    bundle = _bundle("FAISS is a similarity search library developed by Facebook AI.")
    answer = "FAISS is a similarity search library. [D1]"
    report = GroundingVerifier().verify(answer, bundle)
    assert report.grounding_rate == 1.0


def test_grounding_rate_none_grounded():
    bundle = _bundle("Some unrelated content here.")
    answer = "The sky is blue today. [D99]"
    report = GroundingVerifier().verify(answer, bundle)
    assert report.grounding_rate == 0.0


def test_grounding_rate_empty_answer():
    bundle = _bundle("content")
    report = GroundingVerifier().verify("", bundle)
    assert report.verdicts == []
    assert report.grounding_rate == 1.0
    assert report.answer_clean == ""


def test_dangling_citations_property():
    bundle = _bundle("content")
    answer = "Fact one [D1]. Fact two [D7]. Fact three [D8]."
    report = GroundingVerifier().verify(answer, bundle)
    assert set(report.dangling_citations) == {"D7", "D8"}


def test_ungrounded_citations_includes_dangling_and_low_overlap():
    bundle = _bundle("Completely unrelated document about cooking.")
    answer = "FAISS is used for vector search. [D1]"
    report = GroundingVerifier(overlap_threshold=0.5).verify(answer, bundle)
    assert "D1" in report.ungrounded_citations


# ---------------------------------------------------------------------------
# Integration: AnswerGenerationRequest.verify_grounding → generate_answer
# ---------------------------------------------------------------------------


def test_generate_answer_attaches_grounding_report_when_requested():
    from src.context.pipeline import generate_answer
    from src.context.models import AnswerGenerationRequest

    bundle = _bundle("FAISS is a vector similarity search library.")
    req = AnswerGenerationRequest(
        question="What is FAISS?",
        context=bundle,
        verify_grounding=True,
    )
    result = generate_answer(req, llm=None)
    assert result.grounding_report is not None
    assert isinstance(result.grounding_report, GroundingReport)


def test_generate_answer_no_grounding_report_by_default():
    from src.context.pipeline import generate_answer
    from src.context.models import AnswerGenerationRequest

    bundle = _bundle("FAISS content.")
    req = AnswerGenerationRequest(question="What is FAISS?", context=bundle)
    result = generate_answer(req, llm=None)
    assert result.grounding_report is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_grounding.py -v
```

Expected: FAIL with `ImportError: cannot import name 'GroundingVerifier' from 'src.context.grounding'`

- [ ] **Step 3: Add `CitationVerdict` and `GroundingReport` to models.py**

In `src/context/models.py`, add these two dataclasses after `EvidenceSnippet` (around line 153):

```python
@dataclass(frozen=True)
class CitationVerdict:
    """Grounding verdict for a single [Dx] citation in a generated answer."""

    citation: str
    document_found: bool
    overlap_score: float
    is_grounded: bool
    sentence: str


@dataclass(frozen=True)
class GroundingReport:
    """Per-answer report produced by GroundingVerifier."""

    verdicts: list[CitationVerdict]
    answer_clean: str

    @property
    def dangling_citations(self) -> list[str]:
        return [v.citation for v in self.verdicts if not v.document_found]

    @property
    def ungrounded_citations(self) -> list[str]:
        return [v.citation for v in self.verdicts if not v.is_grounded]

    @property
    def grounding_rate(self) -> float:
        if not self.verdicts:
            return 1.0
        return sum(1 for v in self.verdicts if v.is_grounded) / len(self.verdicts)
```

Add `verify_grounding: bool = False` to `AnswerGenerationRequest`:

```python
@dataclass(frozen=True)
class AnswerGenerationRequest:
    question: str
    context: SearchContextBundle
    chat_history: list[ChatMessage] = field(default_factory=list)
    behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    verify_grounding: bool = False
```

Add `grounding_report: GroundingReport | None = None` to `AnswerGenerationResult`:

```python
@dataclass(frozen=True)
class AnswerGenerationResult:
    answer: str
    citations: list[str]
    context: SearchContextBundle
    prompt: PromptBundle
    grounding_report: GroundingReport | None = None
```

- [ ] **Step 4: Create `src/context/grounding.py`**

```python
"""Runtime citation grounding verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import CitationVerdict, GroundingReport, SearchContextBundle

_CITATION_RE = re.compile(r"\[(D\d+)\]")
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "to of in on at for by with from about into through during before "
    "after above below between among and or but nor so yet both either "
    "neither not only also just more most some any such other each every "
    "both few more most other some such no nor not only own same so than "
    "too very i me my we our you your he she it its they them their "
    "what which who whom this that these those i s t don doesn won couldn "
    "how when where why".split()
)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _overlap(sentence_tokens: set[str], doc_tokens: set[str]) -> float:
    if not sentence_tokens or not doc_tokens:
        return 0.0
    return len(sentence_tokens & doc_tokens) / len(sentence_tokens)


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


class GroundingVerifier:
    """Verifies that each [Dx] citation in an answer is supported by the cited document.

    Uses stopword-filtered lexical overlap as a cheap entailment proxy — no NLI
    model required, runs in < 1 ms per answer.  Dangling citations (referencing
    documents not present in the context) are always flagged regardless of threshold.

    Args:
        overlap_threshold: Minimum fraction of sentence tokens that must appear in
            the cited document for the citation to be considered grounded.
            Default 0.15 is intentionally lenient to avoid false positives on
            paraphrase-style citations.
    """

    def __init__(self, *, overlap_threshold: float = 0.15) -> None:
        self.overlap_threshold = overlap_threshold

    def verify(self, answer: str, context: SearchContextBundle) -> GroundingReport:
        doc_map = {doc.id: doc for doc in context.documents}
        sentences = _split_sentences(answer)

        verdicts: list[CitationVerdict] = []
        for sentence in sentences:
            for citation in _CITATION_RE.findall(sentence):
                doc = doc_map.get(citation)
                if doc is None:
                    verdicts.append(CitationVerdict(
                        citation=citation,
                        document_found=False,
                        overlap_score=0.0,
                        is_grounded=False,
                        sentence=sentence,
                    ))
                    continue
                score = _overlap(_tokenize(sentence), _tokenize(doc.content))
                verdicts.append(CitationVerdict(
                    citation=citation,
                    document_found=True,
                    overlap_score=score,
                    is_grounded=score >= self.overlap_threshold,
                    sentence=sentence,
                ))

        dangling = {v.citation for v in verdicts if not v.document_found}
        answer_clean = answer
        for cit in sorted(dangling):
            answer_clean = re.sub(rf"\[{re.escape(cit)}\]", "", answer_clean)
        answer_clean = re.sub(r" {2,}", " ", answer_clean).strip()

        return GroundingReport(verdicts=verdicts, answer_clean=answer_clean)
```

- [ ] **Step 5: Wire verifier into `generate_answer()` in pipeline.py**

In `src/context/pipeline.py`, update `generate_answer()`:

```python
def generate_answer(
    request: AnswerGenerationRequest,
    *,
    llm: LLMClient | None = None,
) -> AnswerGenerationResult:
    prompt = build_chat_prompt(
        request.question,
        request.context,
        history=request.chat_history,
        config=request.behavior,
    )
    if llm is None:
        answer = synthesize_answer_from_context(request.question, request.context)
    else:
        raw = llm.complete(prompt.messages)
        answer = raw.text if isinstance(raw, LLMResponse) else str(raw)

    grounding_report = None
    if request.verify_grounding:
        from .grounding import GroundingVerifier
        report = GroundingVerifier().verify(answer, request.context)
        answer = report.answer_clean
        grounding_report = report

    return AnswerGenerationResult(
        answer=answer,
        citations=extract_citations(answer),
        context=request.context,
        prompt=prompt,
        grounding_report=grounding_report,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/test_grounding.py -v
```

Expected: ALL PASS

- [ ] **Step 7: Run full unit suite to verify no regressions**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: ALL PASS. Existing callers of `generate_answer()` pass no `verify_grounding`, so `grounding_report` is `None` and behaviour is identical.

- [ ] **Step 8: Commit**

```bash
git add src/context/grounding.py src/context/models.py src/context/pipeline.py tests/unit/test_grounding.py
git commit -m "feat: add runtime citation grounding verifier with dangling-citation stripping"
```

---

## Task 6: Full Regression Run + PR

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 2: Lint**

```bash
ruff check . --fix && ruff format .
```

Expected: No errors.

- [ ] **Step 3: Verify agentic RAG end-to-end (smoke test)**

```bash
python3 -m examples.run_agentic_search \
  --mode search \
  --question "What is the difference between dense and sparse retrieval?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local --device cpu
```

Expected: Answer with citations, no tracebacks.

- [ ] **Step 4: Push branch and open PR**

```bash
git push -u origin add-agent-trigger-check-script
```

Then open PR targeting `main` with title:
`feat: optimize Agentic RAG — chunking, step-back, dedup, gap analysis, MMR, grounding`

---

## Self-Review

**Spec coverage:**
- ✅ Paragraph-aware chunking (Task 7) — sections stay intact; `section_continuation` flag set
- ✅ Step-back prompting (Task 1) — broader query generation for coverage
- ✅ Seen-query deduplication (Task 2) — no wasted retrieval calls
- ✅ Structured gap analysis (Task 3) — targeted follow-ups instead of vague ones
- ✅ MMR diversity (Task 4) — reduces result cluster redundancy post-fusion
- ✅ Answer synthesis prompt (Task 5) — grounding, conflict handling, uncertainty
- ✅ Runtime citation grounding verifier (Task 8) — dangling citation stripping + per-citation overlap verdicts

**Placeholder scan:** No TBDs, todos, or vague instructions — all steps have complete code.

**Type consistency:**
- `QueryBundle.step_back_query: str | None` — used consistently in `enhance()` and tests
- `maximal_marginal_relevance()` returns `list[dict[str, Any]]` — same type as `combine_retrieval_results()`
- `HybridRetrieverConfig.mmr_lambda: float`, `mmr_topk: int | None` — used in `retrieve()` guard and `maximal_marginal_relevance()` call
- `_parse_gap_queries()` returns `list[str]` — same as old `_generate_followup()` return type
- `CitationVerdict` and `GroundingReport` defined in `models.py`, imported by `grounding.py` — no circular dependency (`grounding.py` imports from `models.py`; `pipeline.py` imports `GroundingVerifier` lazily inside `generate_answer()` to avoid any import cycles)
- `AnswerGenerationResult.grounding_report: GroundingReport | None = None` — backward compatible; all existing callers receive `None` unless they opt in via `verify_grounding=True`
