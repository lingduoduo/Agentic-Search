"""Search-context, prompt, chat, and answer-generation helpers."""

from .enums import AgentBehavior
from .enums import AnswerStyle
from .enums import QueryType
from .enums import SearchType
from .models import AgentBehaviorConfig
from .models import AnswerGenerationRequest
from .models import AnswerGenerationResult
from .models import ChatMessage
from .models import ContextDocument
from .models import ContextSection
from .models import EvidenceSnippet
from .models import LLMClient
from .models import LLMResponse
from .models import PromptBundle
from .models import SearchContextBundle
from .models import SearchFilters
from .models import SearchRequest
from .pipeline import answer_with_retrieval
from .pipeline import generate_answer
from .pipeline import rank_evidence_snippets
from .pipeline import retrieve_context
from .pipeline import synthesize_answer_from_context
from .prompts import build_agent_behavior_prompt
from .prompts import build_answer_prompt
from .prompts import build_chat_prompt
from .prompts import build_retrieval_prompt
from .utils import build_context_bundle
from .utils import documents_from_search_results
from .utils import extract_citations
from .utils import merge_adjacent_documents

__all__ = [
    "AgentBehavior",
    "AgentBehaviorConfig",
    "AnswerGenerationRequest",
    "AnswerGenerationResult",
    "AnswerStyle",
    "ChatMessage",
    "ContextDocument",
    "ContextSection",
    "EvidenceSnippet",
    "LLMClient",
    "LLMResponse",
    "PromptBundle",
    "QueryType",
    "SearchContextBundle",
    "SearchFilters",
    "SearchRequest",
    "SearchType",
    "answer_with_retrieval",
    "build_agent_behavior_prompt",
    "build_answer_prompt",
    "build_chat_prompt",
    "build_context_bundle",
    "build_retrieval_prompt",
    "documents_from_search_results",
    "extract_citations",
    "generate_answer",
    "merge_adjacent_documents",
    "rank_evidence_snippets",
    "retrieve_context",
    "synthesize_answer_from_context",
]
