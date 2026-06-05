# Deep Research Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `DeepResearchAgent` that turns a broad question into a multi-section, cited research report via LLM-planned outline → per-section iterative retrieval → per-section synthesis → final compiled report.

**Architecture:** Three phases: (1) `_plan_research()` — one LLM call produces a `ResearchPlan` (title + list of `SectionSpec`); (2) `_research_section()` — one `retrieve_context` call + one LLM synthesis call per section; (3) `_compile_report()` — assembles sections into a markdown report. Exposed to the web API as a new `"deep_report"` mode alongside the existing `chat_once` / `chat_loop` / `hybrid_search` modes.

**Tech Stack:** Python 3.11+, pytest, unittest.mock, asyncio; no new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `src/agents/deep_research.py` | Create: all agent code — models, prompts, parsers, `DeepResearchAgent` |
| `src/backend/servers/web/app.py` | Add `"deep_report"` to `_VALID_AGENT_MODES`; handle in `run_agent()` |
| `tests/unit/test_deep_research.py` | Create: full unit test suite |

---

## Task 1: Core Models + Research Plan Parser

The plan parser is the only pure-function core — no IO, easy to test exhaustively. Write it first so Tasks 2-3 can build on a verified foundation.

**Files:**
- Create: `src/agents/deep_research.py`
- Create: `tests/unit/test_deep_research.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_deep_research.py`:

```python
"""Unit tests for DeepResearchAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.deep_research import (
    DeepResearchAgent,
    DeepResearchConfig,
    DeepResearchResult,
    ResearchPlan,
    SectionSpec,
    _parse_research_plan,
)
from src.context.models import ContextDocument, SearchContextBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(contents: list[str], query: str = "q") -> SearchContextBundle:
    docs = [
        ContextDocument(id=f"D{i}", title=f"T{i}", content=c, score=0.9)
        for i, c in enumerate(contents, 1)
    ]
    return SearchContextBundle(query=query, documents=docs)


def _llm(*responses: str) -> MagicMock:
    llm = MagicMock()
    llm.complete.side_effect = list(responses)
    return llm


# ---------------------------------------------------------------------------
# _parse_research_plan
# ---------------------------------------------------------------------------


def test_parse_plan_extracts_title_and_sections():
    raw = (
        "TITLE: Dense vs Sparse Retrieval: A Comparison\n\n"
        "SECTIONS:\n"
        "1. Dense Retrieval | How do embedding-based methods work?\n"
        "2. Sparse Retrieval | How does BM25 score documents?\n"
        "3. Hybrid Approaches | When should dense and sparse be combined?\n"
    )
    plan = _parse_research_plan(raw, question="Compare dense and sparse retrieval", max_sections=4)
    assert plan.title == "Dense vs Sparse Retrieval: A Comparison"
    assert len(plan.sections) == 3
    assert plan.sections[0].title == "Dense Retrieval"
    assert plan.sections[0].research_question == "How do embedding-based methods work?"
    assert plan.sections[2].title == "Hybrid Approaches"


def test_parse_plan_missing_title_falls_back_to_question():
    raw = "SECTIONS:\n1. Overview | What is FAISS?\n"
    plan = _parse_research_plan(raw, question="Explain FAISS", max_sections=4)
    assert plan.title == "Explain FAISS"
    assert len(plan.sections) == 1


def test_parse_plan_respects_max_sections_cap():
    raw = (
        "TITLE: Report\n\n"
        "SECTIONS:\n"
        "1. A | Q1\n"
        "2. B | Q2\n"
        "3. C | Q3\n"
        "4. D | Q4\n"
        "5. E | Q5\n"
    )
    plan = _parse_research_plan(raw, question="q", max_sections=3)
    assert len(plan.sections) == 3


def test_parse_plan_no_pipe_uses_line_as_both_title_and_question():
    raw = "TITLE: T\n\nSECTIONS:\n1. Background and Context\n"
    plan = _parse_research_plan(raw, question="q", max_sections=4)
    assert plan.sections[0].title == "Background and Context"
    assert plan.sections[0].research_question == "Background and Context"


def test_parse_plan_empty_raw_returns_single_overview_section():
    plan = _parse_research_plan("", question="What is RAG?", max_sections=4)
    assert len(plan.sections) == 1
    assert plan.sections[0].title == "Overview"
    assert plan.sections[0].research_question == "What is RAG?"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_deep_research.py -k "parse_plan" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.deep_research'`

- [ ] **Step 3: Create `src/agents/deep_research.py` with models and parser**

```python
"""Deep research agent: plan an outline → research each section → compile a report."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.context.models import (
    ChatMessage,
    ContextDocument,
    LLMClient,
    SearchContextBundle,
)
from src.context.pipeline import retrieve_context, synthesize_answer_from_context
from src.context.utils import extract_citations

logger = logging.getLogger(__name__)

_LIST_MARKER_RE = re.compile(r"^\d+[.)]\s*|^[-*•]\s*")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionSpec:
    """A single planned section: its display title and the focused research question."""

    title: str
    research_question: str


@dataclass(frozen=True)
class ResearchPlan:
    """The LLM-generated report outline."""

    title: str
    sections: list[SectionSpec]


@dataclass
class ReportSection:
    """Synthesized content for one section of the report."""

    title: str
    content: str
    citations: list[str]
    documents: list[ContextDocument]


@dataclass
class DeepResearchResult:
    """Final output of DeepResearchAgent.run()."""

    full_report: str
    plan: ResearchPlan
    sections: list[ReportSection]
    documents: list[ContextDocument]


@dataclass(frozen=True)
class DeepResearchConfig:
    max_sections: int = 4
    topk_per_section: int = 5
    retrieval_url: str = "http://localhost:8000/retrieve"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_research_plan(raw: str, *, question: str, max_sections: int) -> ResearchPlan:
    """Parse a structured LLM plan response into a ResearchPlan.

    Expected format:
        TITLE: <title>

        SECTIONS:
        1. <section title> | <research question>
        2. <section title> | <research question>

    Falls back gracefully when sections or title are missing.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    title = question
    section_lines: list[str] = []
    in_sections = False

    for line in lines:
        upper = line.upper()
        if upper.startswith("TITLE:"):
            candidate = line[6:].strip()
            if candidate:
                title = candidate
        elif upper.startswith("SECTIONS:"):
            in_sections = True
        elif in_sections:
            section_lines.append(line)

    sections: list[SectionSpec] = []
    for line in section_lines:
        line = _LIST_MARKER_RE.sub("", line).strip()
        if not line:
            continue
        if "|" in line:
            title_part, _, question_part = line.partition("|")
            sections.append(
                SectionSpec(
                    title=title_part.strip(),
                    research_question=question_part.strip() or title_part.strip(),
                )
            )
        else:
            sections.append(SectionSpec(title=line, research_question=line))
        if len(sections) >= max_sections:
            break

    if not sections:
        sections = [SectionSpec(title="Overview", research_question=question)]

    return ResearchPlan(title=title, sections=sections)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_deep_research.py -k "parse_plan" -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/deep_research.py tests/unit/test_deep_research.py
git commit -m "feat: add DeepResearchAgent models and research plan parser"
```

---

## Task 2: Prompts + `_plan_research()` + `_research_section()`

Add the LLM prompts and the two main worker methods. Both are independently testable: `_plan_research` only calls `self.llm.complete`; `_research_section` calls `retrieve_context` (mockable) + `self.llm.complete`.

**Files:**
- Modify: `src/agents/deep_research.py`
- Modify: `tests/unit/test_deep_research.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_deep_research.py`:

```python
# ---------------------------------------------------------------------------
# DeepResearchAgent._plan_research
# ---------------------------------------------------------------------------


def test_plan_research_parses_llm_output():
    llm = _llm(
        "TITLE: FAISS In Depth\n\n"
        "SECTIONS:\n"
        "1. What Is FAISS | What is FAISS and what problem does it solve?\n"
        "2. GPU Support | How does FAISS handle GPU indexing?\n"
    )
    agent = DeepResearchAgent(DeepResearchConfig(max_sections=4), llm=llm)
    plan = agent._plan_research("Explain FAISS deeply")
    assert plan.title == "FAISS In Depth"
    assert len(plan.sections) == 2
    assert plan.sections[1].research_question == "How does FAISS handle GPU indexing?"


def test_plan_research_returns_fallback_when_llm_is_none():
    agent = DeepResearchAgent(DeepResearchConfig(), llm=None)
    plan = agent._plan_research("Explain RAG")
    assert plan.title == "Explain RAG"
    assert len(plan.sections) == 1
    assert plan.sections[0].title == "Overview"


def test_plan_research_returns_fallback_on_llm_failure():
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("timeout")
    agent = DeepResearchAgent(DeepResearchConfig(), llm=llm)
    plan = agent._plan_research("Explain RAG")
    assert len(plan.sections) == 1
    assert plan.sections[0].title == "Overview"


# ---------------------------------------------------------------------------
# DeepResearchAgent._research_section
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_section_returns_report_section():
    bundle = _make_bundle(["FAISS is a vector similarity search library by Facebook AI."])
    llm = _llm("FAISS enables efficient similarity search. [D1]")
    agent = DeepResearchAgent(DeepResearchConfig(), llm=llm)
    spec = SectionSpec(title="What Is FAISS", research_question="What is FAISS?")

    with patch("src.agents.deep_research.retrieve_context", AsyncMock(return_value=bundle)):
        section = await agent._research_section(spec, report_title="FAISS Report")

    assert section.title == "What Is FAISS"
    assert "FAISS" in section.content
    assert section.documents


@pytest.mark.asyncio
async def test_research_section_handles_retrieval_failure_gracefully():
    llm = _llm("fallback answer")
    agent = DeepResearchAgent(DeepResearchConfig(), llm=llm)
    spec = SectionSpec(title="Fail Section", research_question="unreachable?")

    with patch(
        "src.agents.deep_research.retrieve_context",
        AsyncMock(side_effect=ConnectionError("server down")),
    ):
        section = await agent._research_section(spec, report_title="Report")

    assert section.title == "Fail Section"
    assert "retrieval failed" in section.content.lower() or section.content
    assert section.documents == []


@pytest.mark.asyncio
async def test_research_section_uses_extractive_fallback_when_llm_is_none():
    bundle = _make_bundle(["FAISS is a fast similarity search library."])
    agent = DeepResearchAgent(DeepResearchConfig(), llm=None)
    spec = SectionSpec(title="FAISS", research_question="What is FAISS?")

    with patch("src.agents.deep_research.retrieve_context", AsyncMock(return_value=bundle)):
        section = await agent._research_section(spec, report_title="Report")

    assert section.title == "FAISS"
    assert section.content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_deep_research.py -k "plan_research or research_section" -v
```

Expected: FAIL with `AttributeError: 'DeepResearchAgent' object has no attribute '_plan_research'`

- [ ] **Step 3: Add prompts and methods to `deep_research.py`**

Append the prompt constants after `_LIST_MARKER_RE`:

```python
_RESEARCH_PLAN_PROMPT = """You are planning a research report.

Question: {question}

Create a report outline with up to {max_sections} sections, each covering a distinct aspect.

Output only this format — no extra text:
TITLE: <report title>

SECTIONS:
1. <section title> | <focused research question for this section>
2. <section title> | <focused research question>

Keep titles short (3–6 words). Make each research question specific and answerable.""".strip()

_SECTION_SYNTHESIS_PROMPT = """You are writing one section of a research report.

Report title: {report_title}
Section title: {section_title}
Section research question: {research_question}

Retrieved evidence:
{context}

Write 2–3 focused paragraphs for this section using ONLY the retrieved evidence.
Cite sources inline using [Dx] notation (e.g. [D1], [D2]).
Do not fabricate information absent from the evidence.
If the evidence is insufficient, state what is missing rather than speculating.""".strip()
```

Add `_llm_text` helper and the `DeepResearchAgent` class at the bottom of the module:

```python
def _llm_text(response: object) -> str:
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


class DeepResearchAgent:
    """Produces a multi-section research report via LLM-planned outline + per-section retrieval.

    Flow:
      1. _plan_research(question)  →  ResearchPlan (1 LLM call)
      2. _research_section(spec)   →  ReportSection per section (1 retrieve + 1 LLM call each)
      3. _compile_report(plan, sections)  →  markdown string
    """

    def __init__(
        self,
        config: DeepResearchConfig,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.llm = llm

    def _plan_research(self, question: str) -> ResearchPlan:
        """Call the LLM to generate a structured report outline."""
        if self.llm is None:
            return ResearchPlan(
                title=question,
                sections=[SectionSpec(title="Overview", research_question=question)],
            )
        prompt = _RESEARCH_PLAN_PROMPT.format(
            question=question,
            max_sections=self.config.max_sections,
        )
        try:
            raw = _llm_text(
                self.llm.complete([ChatMessage(role="user", content=prompt)])
            ).strip()
            return _parse_research_plan(
                raw, question=question, max_sections=self.config.max_sections
            )
        except Exception as exc:
            logger.warning("Research planning failed: %s", exc)
            return ResearchPlan(
                title=question,
                sections=[SectionSpec(title="Overview", research_question=question)],
            )

    async def _research_section(
        self,
        spec: SectionSpec,
        *,
        report_title: str,
    ) -> ReportSection:
        """Retrieve evidence for one section and synthesize its content."""
        try:
            ctx = await retrieve_context(
                spec.research_question,
                search_url=self.config.retrieval_url,
                top_k=self.config.topk_per_section,
            )
        except Exception as exc:
            logger.warning("Retrieval failed for section %r: %s", spec.title, exc)
            return ReportSection(
                title=spec.title,
                content=f"*Retrieval failed for this section: {exc}*",
                citations=[],
                documents=[],
            )

        if self.llm is None:
            content = synthesize_answer_from_context(spec.research_question, ctx)
        else:
            prompt = _SECTION_SYNTHESIS_PROMPT.format(
                report_title=report_title,
                section_title=spec.title,
                research_question=spec.research_question,
                context=ctx.to_context_text(),
            )
            raw = self.llm.complete([ChatMessage(role="user", content=prompt)])
            content = _llm_text(raw).strip()

        return ReportSection(
            title=spec.title,
            content=content,
            citations=extract_citations(content),
            documents=list(ctx.documents),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_deep_research.py -k "plan_research or research_section" -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/deep_research.py tests/unit/test_deep_research.py
git commit -m "feat: add _plan_research and _research_section to DeepResearchAgent"
```

---

## Task 3: `run()` + `_compile_report()`

Wire the two worker methods into a full async `run()` that processes all sections and assembles the final report. Cross-section document deduplication happens here.

**Files:**
- Modify: `src/agents/deep_research.py`
- Modify: `tests/unit/test_deep_research.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_deep_research.py`:

```python
# ---------------------------------------------------------------------------
# DeepResearchAgent.run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_deep_research_result():
    bundle = _make_bundle(["Dense retrieval uses embeddings.", "BM25 uses TF-IDF scoring."])
    llm = _llm(
        # plan
        "TITLE: Retrieval Methods\n\nSECTIONS:\n1. Dense | How does dense retrieval work?\n2. Sparse | How does BM25 work?\n",
        # section 1 synthesis
        "Dense retrieval maps queries to vectors. [D1]",
        # section 2 synthesis
        "BM25 ranks by term frequency. [D2]",
    )
    config = DeepResearchConfig(max_sections=2, topk_per_section=5)

    with patch("src.agents.deep_research.retrieve_context", AsyncMock(return_value=bundle)):
        agent = DeepResearchAgent(config, llm=llm)
        result = await agent.run("Compare dense and sparse retrieval")

    assert isinstance(result, DeepResearchResult)
    assert result.plan.title == "Retrieval Methods"
    assert len(result.sections) == 2
    assert "## Dense" in result.full_report
    assert "## Sparse" in result.full_report
    assert result.documents


@pytest.mark.asyncio
async def test_run_report_starts_with_markdown_h1_title():
    bundle = _make_bundle(["content about X."])
    llm = _llm(
        "TITLE: My Report\n\nSECTIONS:\n1. Overview | What is X?\n",
        "X is a thing. [D1]",
    )
    with patch("src.agents.deep_research.retrieve_context", AsyncMock(return_value=bundle)):
        agent = DeepResearchAgent(DeepResearchConfig(max_sections=2), llm=llm)
        result = await agent.run("What is X?")

    assert result.full_report.startswith("# My Report")


@pytest.mark.asyncio
async def test_run_deduplicates_documents_across_sections():
    shared_content = "Shared document content."
    bundle = _make_bundle([shared_content])
    llm = _llm(
        "TITLE: T\n\nSECTIONS:\n1. A | Q1\n2. B | Q2\n",
        "Section A content. [D1]",
        "Section B content. [D1]",
    )
    with patch("src.agents.deep_research.retrieve_context", AsyncMock(return_value=bundle)):
        agent = DeepResearchAgent(DeepResearchConfig(max_sections=2), llm=llm)
        result = await agent.run("q")

    # Both sections retrieved the same doc; it should appear only once globally.
    assert len(result.documents) == 1


@pytest.mark.asyncio
async def test_run_calls_retrieve_once_per_section():
    bundle = _make_bundle(["content"])
    llm = _llm(
        "TITLE: T\n\nSECTIONS:\n1. A | Q1\n2. B | Q2\n3. C | Q3\n",
        "s1", "s2", "s3",
    )
    retrieve_mock = AsyncMock(return_value=bundle)
    with patch("src.agents.deep_research.retrieve_context", retrieve_mock):
        agent = DeepResearchAgent(DeepResearchConfig(max_sections=3), llm=llm)
        await agent.run("q")

    assert retrieve_mock.call_count == 3


@pytest.mark.asyncio
async def test_run_continues_after_one_section_retrieval_failure():
    good_bundle = _make_bundle(["good content"])
    llm = _llm(
        "TITLE: T\n\nSECTIONS:\n1. Good | Q1\n2. Bad | Q2\n",
        "good answer [D1]",
        # no third response needed — failed section uses error text, skips LLM synthesis
    )

    call_count = 0

    async def _mixed_retrieve(question: str, **kwargs: object) -> SearchContextBundle:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ConnectionError("server down")
        return good_bundle

    with patch("src.agents.deep_research.retrieve_context", side_effect=_mixed_retrieve):
        agent = DeepResearchAgent(DeepResearchConfig(max_sections=2), llm=llm)
        result = await agent.run("q")

    assert len(result.sections) == 2
    assert result.sections[0].documents  # good section has docs
    assert result.sections[1].documents == []  # failed section has no docs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_deep_research.py -k "test_run" -v
```

Expected: FAIL with `AttributeError: 'DeepResearchAgent' object has no attribute 'run'`

- [ ] **Step 3: Add `run()` and `_compile_report()` to `deep_research.py`**

Append to the `DeepResearchAgent` class (inside the class body, after `_research_section`):

```python
    async def run(self, question: str) -> DeepResearchResult:
        """Research the question and return a fully compiled DeepResearchResult."""
        plan = self._plan_research(question)

        sections: list[ReportSection] = []
        for spec in plan.sections:
            section = await self._research_section(spec, report_title=plan.title)
            sections.append(section)

        full_report = _compile_report(plan, sections)

        # Deduplicate across sections by (url, content-prefix) fingerprint.
        all_docs: list[ContextDocument] = []
        seen: set[tuple[str | None, str]] = set()
        for section in sections:
            for doc in section.documents:
                key = (doc.url, doc.content[:120])
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        return DeepResearchResult(
            full_report=full_report,
            plan=plan,
            sections=sections,
            documents=all_docs,
        )
```

Add the `_compile_report` module-level function after `_llm_text`:

```python
def _compile_report(plan: ResearchPlan, sections: list[ReportSection]) -> str:
    """Assemble section content into a markdown report."""
    parts = [f"# {plan.title}\n"]
    for section in sections:
        parts.append(f"## {section.title}\n\n{section.content}")
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run all deep-research tests**

```bash
pytest tests/unit/test_deep_research.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/deep_research.py tests/unit/test_deep_research.py
git commit -m "feat: add DeepResearchAgent.run() with section assembly and cross-section doc deduplication"
```

---

## Task 4: Wire `"deep_report"` Mode into the Web App

Add `"deep_report"` as a valid agent mode in `app.py`. The handler calls `DeepResearchAgent.run()` and maps the result into the existing `AgentExperienceResponse` shape — no new endpoint or response type needed.

**Files:**
- Modify: `src/backend/servers/web/app.py`
- Modify: `tests/unit/servers/test_web_app_agent.py` (if it exists — check first; add a new test file if not)

- [ ] **Step 1: Check whether a web-app agent test file exists**

```bash
ls tests/unit/servers/ 2>/dev/null || echo "directory not found"
grep -rn "run_agent\|AgentExperienceRequest\|/api/agent" tests/ --include="*.py" -l
```

- [ ] **Step 2: Write a failing test for the `deep_report` mode**

Identify the correct test file from the grep output above. If none exists, create `tests/unit/test_web_app_agent.py`. Add:

```python
"""Tests for the deep_report agent mode in the web app."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agents.deep_research import DeepResearchConfig, DeepResearchResult, ResearchPlan, ReportSection, SectionSpec
from src.context.models import ContextDocument, SearchContextBundle
from src.backend.servers.web.app import create_web_app


def _make_research_result() -> DeepResearchResult:
    docs = [ContextDocument(id="D1", title="T", content="C", score=0.9)]
    plan = ResearchPlan(
        title="Test Report",
        sections=[SectionSpec(title="Overview", research_question="What is X?")],
    )
    sections = [ReportSection(title="Overview", content="X is a thing. [D1]", citations=["D1"], documents=docs)]
    return DeepResearchResult(
        full_report="# Test Report\n\n## Overview\n\nX is a thing. [D1]",
        plan=plan,
        sections=sections,
        documents=docs,
    )


def test_deep_report_mode_returns_full_report():
    app = create_web_app()
    client = TestClient(app)

    mock_result = _make_research_result()

    with patch(
        "src.backend.servers.web.app.DeepResearchAgent"
    ) as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_instance

        resp = client.post(
            "/api/agent",
            json={"query": "Compare dense and sparse retrieval", "mode": "deep_report"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "# Test Report" in body["answer"]
    assert body["documents"]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/test_web_app_agent.py::test_deep_report_mode_returns_full_report -v
```

Expected: FAIL — `422 Unprocessable Entity` because `"deep_report"` is not a valid mode yet.

- [ ] **Step 4: Add the import and mode to `app.py`**

At the top of `src/backend/servers/web/app.py`, add the import after the existing agent imports:

```python
from src.agents.deep_research import DeepResearchAgent, DeepResearchConfig
```

In `_VALID_AGENT_MODES`, add the new mode:

```python
_VALID_AGENT_MODES = {
    "search_tool",
    "hybrid_search",
    "chat_once",
    "chat_loop",
    "deep_report",
}
```

- [ ] **Step 5: Add the `deep_report` handler block in `run_agent()`**

In `src/backend/servers/web/app.py`, insert the following block inside `run_agent()`, immediately after the `if mode == "chat_loop":` block (around line 461 in the current file, before the `result = await answer_with_retrieval(...)` call):

```python
            if mode == "deep_report":
                researcher = DeepResearchAgent(
                    DeepResearchConfig(
                        max_sections=4,
                        topk_per_section=top_k,
                        retrieval_url=search_url,
                    ),
                    llm=llm,
                )
                research = await researcher.run(query)
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=research.full_report,
                    metadata={
                        "citations": [doc.citation for doc in research.documents],
                        "document_ids": [doc.id for doc in research.documents],
                        "hooks": hook_metadata,
                        "mode": mode,
                        "sections": [s.title for s in research.sections],
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=research.full_report,
                    citations=[doc.citation for doc in research.documents],
                    documents=[_document_view(doc) for doc in research.documents],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/test_web_app_agent.py -v
```

Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents/deep_research.py src/backend/servers/web/app.py tests/unit/test_web_app_agent.py
git commit -m "feat: wire deep_report agent mode into web API"
```

---

## Task 5: Regression Run + PR

- [ ] **Step 1: Run full unit test suite**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: ALL PASS. No existing test should change behaviour — `_VALID_AGENT_MODES` gained one member; all other modes are unaffected.

- [ ] **Step 2: Lint**

```bash
ruff check . --fix && ruff format .
```

Expected: No errors.

- [ ] **Step 3: Smoke-test the deep_report mode end-to-end**

Start the demo retrieval server in a separate terminal:

```bash
python3 -m src.backend.servers.retrieval.demo --corpus_path data/corpus.jsonl
```

Then in the project root:

```bash
curl -s -X POST http://127.0.0.1:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval methods", "mode": "deep_report"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['answer'][:500])"
```

Expected: A markdown-formatted report beginning with `# ` and containing at least one `## ` section header.

- [ ] **Step 4: Push branch and open PR**

```bash
git push -u origin add-agent-trigger-check-script
```

Open PR targeting `main` with title:
`feat: add DeepResearchAgent — multi-section reports via LLM-planned outline + per-section retrieval`

---

## Self-Review

**Spec coverage:**
- ✅ Multi-step research flow — `_plan_research()` decomposes broad question into section specs, each researched independently
- ✅ In-depth reports — `_compile_report()` assembles sections into a structured markdown document
- ✅ LLM-planned outline — `_RESEARCH_PLAN_PROMPT` + `_parse_research_plan()` produce a `ResearchPlan`
- ✅ Per-section retrieval + synthesis — `_research_section()` calls `retrieve_context` + LLM once per section
- ✅ Graceful degradation — retrieval failures produce an error-note section; `llm=None` uses extractive fallback
- ✅ Web API integration — `"deep_report"` mode in `run_agent()` returns full report as `answer` field

**Placeholder scan:** No TBDs, no vague steps — all steps contain complete code or exact commands.

**Type consistency:**
- `_parse_research_plan` returns `ResearchPlan` → used directly by `_plan_research` → passed to `_compile_report` ✓
- `_research_section` returns `ReportSection` (not frozen — `documents` list is mutable intentionally) ✓
- `DeepResearchResult.documents: list[ContextDocument]` — same type as `SearchContextBundle.documents`, compatible with `_document_view()` in `app.py` ✓
- `DeepResearchAgent` import added to `app.py` alongside existing agent imports ✓

**Known limitation:** Citations in the report text (e.g. `[D1]`) are section-local — `[D1]` in "Section A" refers to Section A's first retrieved document, not a globally-indexed document. A future improvement would re-number citations globally before synthesis.
