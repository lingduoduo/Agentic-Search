"""Runner for the ToolAgentLoop, shared by the /api/agent path and /tool/* router.

Relocated out of app.py so query_and_chat routers can reuse it without importing
app.py (which imports those routers — that would be circular).
"""

from __future__ import annotations

import json as _json

from pydantic import BaseModel

from src.context.models import ContextDocument
from .intent_routing import _infer_intent_from_output

NO_LOCAL_MODEL_MESSAGE = (
    "tool_agent mode requires a local model. "
    "Set SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL in .env and restart."
)

TOOL_AGENT_SYSTEM_PROMPT = (
    "You are a research assistant with tools. When a question asks about facts, "
    "documents, or anything you cannot verify yourself, call the tool that fits "
    "and answer only from what it returns — do not answer from memory. Call one "
    "tool at a time, then use its result. Answer directly only when no tool applies."
)


# Answers were being cut mid-word at 512 tokens. Two caps govern that, and they
# are not the same budget:
#   max_tokens       — caps ONE generation, so it bounds the answer itself.
#   response_length  — caps the whole rollout AND truncates each response. Tool
#                      results are fed back in and counted here too (they enter
#                      response_mask as zeros), so sizing it to max_tokens alone
#                      lets a few large tool results exhaust the budget before
#                      the model ever writes an answer — the run then ends on
#                      the tool-calling turn and "answers" with <tool_call>
#                      markup. Leave room for the tool traffic.
TOOL_AGENT_MAX_TOKENS = 1024
_ROLLOUT_BUDGET_MULTIPLIER = 4

# The corpus search the agent is given, and the trace name its results carry.
_CORPUS_SEARCH_NAME = "search"
_CORPUS_SEARCH_TOP_K = 5


class ToolCallView(BaseModel):
    tool_name: str
    status: str
    arguments: dict[str, object]
    result_summary: str
    latency_ms: int
    error: str | None = None


def _extract_tool_calls_and_docs(output) -> tuple[list, list]:
    """Parse a ToolAgentLoop action_trace into ToolCallView + ContextDocument lists."""
    tool_calls: list[ToolCallView] = []
    documents: list = []
    if not output.action_trace:
        return tool_calls, documents
    for line in output.action_trace.split("\n"):
        if not line.strip():
            continue
        try:
            rec = _json.loads(line)
            tool_name = rec.get("tool_name", "")
            perf = rec.get("performance", {})
            latency_ms = round(perf.get("execution_time", 0.0) * 1000)
            status_raw = str(rec.get("status", "failed")).lower()
            is_completed = "completed" in status_raw
            result = rec.get("result")
            decoded_result = result
            if isinstance(result, str):
                try:
                    decoded_result = _json.loads(result)
                except Exception:
                    pass
            if isinstance(decoded_result, list):
                result_summary = f"{len(decoded_result)} items"
            elif result is not None:
                result_summary = str(result)[:200]
            else:
                result_summary = ""
            args = rec.get("arguments") or {}
            args = args if isinstance(args, dict) else {}
            tool_calls.append(
                ToolCallView(
                    tool_name=tool_name,
                    status="completed" if is_completed else "failed",
                    arguments=args,
                    result_summary=result_summary,
                    latency_ms=latency_ms,
                    error=rec.get("error_message"),
                )
            )
            if tool_name == _CORPUS_SEARCH_NAME and result:
                raw = _json.loads(result) if isinstance(result, str) else result
                if isinstance(raw, list):
                    for i, item in enumerate(raw, 1):
                        documents.append(
                            ContextDocument(
                                id=f"D{i}",
                                title=item.get("title", ""),
                                content=item.get("content", ""),
                                url=item.get("url"),
                                score=0.0,
                                metadata={"source": _CORPUS_SEARCH_NAME},
                            )
                        )
        except Exception:
            pass
    return tool_calls, documents


async def _run_tool_agent(
    query: str,
    *,
    manager,
    tokenizer,
    search_url: str,
    history: list,
    resolved,
    on_turn=None,
    on_approval=None,
    with_search_tool: bool,
) -> tuple:
    """Run the ToolAgentLoop. Assumes a local model is configured.

    ``answer`` is ``output.final_answer or ""`` with no fallback applied; the
    last assistant message is exposed in ``extra["_assistant_fallback"]`` so the
    auto-route (degrade on empty) and explicit mode (fall back to it) can each
    apply their own policy. Callers must pop ``_assistant_fallback`` before it
    reaches the response/metadata.
    """
    from src.agents.tool import ToolAgentLoop, ToolAgentLoopConfig
    from src.internal.tools import build_search_routing_tool, tool_registry

    max_tokens = (
        getattr(resolved, "tool_agent_max_tokens", None) or TOOL_AGENT_MAX_TOKENS
    )

    # agent_tools() excludes anything registered as not agent-callable: tools
    # that answer instead of returning evidence, and remote tools that re-enter
    # an agent. The decision travels with registration rather than being matched
    # by name here, where a rename would silently disable it.
    tools = list(tool_registry.agent_tools())
    if with_search_tool:
        # Bind the corpus search to this request's retrieval URL rather than the
        # one the registry was seeded with.
        corpus_search = build_search_routing_tool(
            search_url=search_url, top_k=_CORPUS_SEARCH_TOP_K, name=_CORPUS_SEARCH_NAME
        )
        tools = [corpus_search] + [t for t in tools if t.name != corpus_search.name]
    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=manager,
        tools=tools,
        config=ToolAgentLoopConfig(
            response_length=max_tokens * _ROLLOUT_BUDGET_MULTIPLIER,
            tool_parser_format=resolved.tool_agent_parser,
            approval_timeout_seconds=getattr(
                resolved, "tool_approval_timeout_seconds", 60.0
            ),
        ),
    )
    messages = [{"role": m.role, "content": m.content} for m in history] + [
        {"role": "user", "content": query}
    ]
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT})
    output = await loop.run(
        messages,
        sampling_params={"temperature": 0.0, "max_tokens": max_tokens},
        on_turn=on_turn,
        on_approval=on_approval,
    )
    tool_calls, documents = _extract_tool_calls_and_docs(output)
    fallback = next(
        (
            m["content"]
            for m in reversed(output.trajectory_messages)
            if m.get("role") == "assistant"
        ),
        "",
    )
    extra = {
        "tool_calls": tool_calls,
        "num_turns": output.num_turns,
        # The wall clock can cut a long local answer mid-word. Say so rather
        # than handing back a fragment that looks complete.
        "truncated": bool(getattr(output, "truncated", False)),
        "_assistant_fallback": fallback,
    }
    return (
        output.final_answer or "",
        [d.citation for d in documents],
        documents,
        _infer_intent_from_output(output),
        extra,
    )
