# src/cli/_client.py
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class AgentResult:
    session_id: str
    answer: str
    citations: list[str]
    documents: list[dict]


async def query_agent(
    base_url: str,
    query: str,
    token: str,
    *,
    top_k: int = 5,
    session_id: str | None = None,
) -> AgentResult:
    """POST /api/agent and return a typed result.

    Raises ``httpx.HTTPStatusError`` on 4xx/5xx.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/api/agent",
            json={"query": query, "top_k": top_k, "session_id": session_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()

    return AgentResult(
        session_id=data["session_id"],
        answer=data["answer"],
        citations=data.get("citations", []),
        documents=data.get("documents", []),
    )
