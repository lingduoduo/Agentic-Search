"""Prompt builders for retrieval-grounded chat and agent behavior."""

from __future__ import annotations

from .enums import AgentBehavior
from .enums import AnswerStyle
from .models import AgentBehaviorConfig
from .models import ChatMessage
from .models import EvidenceSource
from .models import PromptBundle
from .models import SearchContextBundle


def build_agent_behavior_prompt(config: AgentBehaviorConfig) -> str:
    citation_rule = (
        "Cite retrieved evidence using document labels like [D1]."
        if config.require_citations
        else "Use retrieved evidence when helpful."
    )
    style_rule = {
        AnswerStyle.CONCISE: "Answer concisely.",
        AnswerStyle.DETAILED: "Answer with enough detail to explain the reasoning.",
        AnswerStyle.BULLETS: "Prefer compact bullet points.",
    }[config.answer_style]
    behavior_rule = {
        AgentBehavior.DIRECT: "Answer directly when the provided context is sufficient.",
        AgentBehavior.SEARCH_FIRST: "Prefer searching before answering unless the answer is obvious.",
        AgentBehavior.RESEARCH: "Decompose unclear questions, use evidence, and avoid overclaiming.",
    }[config.behavior]
    return "\n".join([behavior_rule, style_rule, citation_rule])


def build_retrieval_prompt(question: str, *, max_queries: int = 3) -> str:
    return (
        "Generate focused retrieval queries for the user question.\n"
        f"Return at most {max_queries} queries, one per line.\n"
        "Prefer entity names, dates, and the specific fact needed.\n\n"
        f"Question: {question}"
    )


def build_answer_prompt(
    question: str,
    context: SearchContextBundle,
    config: AgentBehaviorConfig | None = None,
    *,
    user_memory: str | None = None,
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
    ) + (user_memory or "")
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


def build_chat_prompt(
    question: str,
    context: SearchContextBundle,
    history: list[ChatMessage] | None = None,
    config: AgentBehaviorConfig | None = None,
    *,
    user_memory: str | None = None,
) -> PromptBundle:
    prompt = build_answer_prompt(question, context, config, user_memory=user_memory)
    messages = [prompt.messages[0], *(history or []), prompt.messages[1]]
    return PromptBundle(system=prompt.system, user=prompt.user, messages=messages)


def build_structured_answer_prompt(
    question: str,
    context: SearchContextBundle,
    config: AgentBehaviorConfig | None = None,
    *,
    history: list[ChatMessage] | None = None,
    evidence: list[EvidenceSource] | None = None,
    user_memory: str | None = None,
) -> PromptBundle:
    """Build the strict internal AnswerDraft prompt used by the safety guard."""
    config = config or AgentBehaviorConfig()
    system = (
        "You are a retrieval-grounded research assistant.\n"
        f"{build_agent_behavior_prompt(config)}\n\n"
        "Return only one JSON object with exactly these keys, in this order: "
        "abstain, missing_information, claims. Each claim must contain exactly "
        "text and evidence_ids. evidence_ids must be a non-empty array of IDs present in "
        "the supplied evidence. Do not emit markdown or extra keys. Every factual "
        "claim must be supported by every evidence ID it cites. If evidence is "
        "missing or you are uncertain, record the gap in missing_information or "
        "set abstain to true; never guess."
    ) + (user_memory or "")
    evidence_text = (
        _format_evidence(evidence)
        if evidence is not None
        else context.to_context_text()
    )
    user = f"Question:\n{question}\n\nEvidence:\n{evidence_text}\n\nReturn the AnswerDraft JSON."
    messages = [
        ChatMessage(role="system", content=system),
        *(history or []),
        ChatMessage(role="user", content=user),
    ]
    return PromptBundle(system=system, user=user, messages=messages)


def build_corrective_answer_prompt(
    question: str,
    context: SearchContextBundle,
    *,
    original_draft: str,
    verifier_feedback: str,
    config: AgentBehaviorConfig | None = None,
    history: list[ChatMessage] | None = None,
    evidence: list[EvidenceSource] | None = None,
    user_memory: str | None = None,
) -> PromptBundle:
    """Build one bounded correction request without changing the evidence."""
    prompt = build_structured_answer_prompt(
        question,
        context,
        config,
        history=history,
        evidence=evidence,
        user_memory=user_memory,
    )
    user = (
        f"{prompt.user}\n\nOriginal structured draft:\n{original_draft}\n\n"
        f"Verifier feedback:\n{verifier_feedback}\n\n"
        "Correct the JSON. Remove unsupported material rather than inventing new "
        "evidence. You may explicitly abstain or report uncertainty."
    )
    messages = [*prompt.messages[:-1], ChatMessage(role="user", content=user)]
    return PromptBundle(system=prompt.system, user=user, messages=messages)


def _format_evidence(evidence: list[EvidenceSource]) -> str:
    if not evidence:
        return "No available evidence."
    return "\n\n".join(f"[{item.id}] {item.title}\n{item.text}" for item in evidence)
