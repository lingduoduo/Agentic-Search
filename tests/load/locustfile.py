"""Locust load test for the retrieval service at 50 QPS.

Usage:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 10 --run-time 60s --headless

To run against the web backend instead:
    locust -f tests/load/locustfile.py --host http://localhost:7860 \
        --users 50 --spawn-rate 10 --run-time 60s --headless \
        -t LoadTestAgent --tags agent

P99 success criteria (M4 gate): no regression vs. M3 baseline when running at 50 QPS.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, tag, task

_QUERIES = [
    "What is retrieval augmented generation?",
    "Compare BM25 and dense retrieval",
    "How does hybrid search work?",
    "What are the best practices for chunking documents?",
    "Explain reciprocal rank fusion",
    "What is the difference between sparse and dense vectors?",
    "How do I set up an OpenSearch cluster?",
    "Weaviate vs Pinecone for vector search",
    "What is FAISS and how does it work?",
    "How to evaluate retrieval quality with NDCG?",
]


class RetrievalUser(HttpUser):
    """Targets the retrieval service /search endpoint (port 8000 by default)."""

    wait_time = between(0.01, 0.05)  # ~20-100 QPS per user at 50 users → ~50 QPS total

    @tag("search")
    @task(10)
    def search(self) -> None:
        query = random.choice(_QUERIES)
        self.client.post(
            "/search",
            json={"query": query, "top_k": 10},
            name="/search",
        )

    @tag("health")
    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")


class AgentUser(HttpUser):
    """Targets the web backend /api/agent endpoint (port 7860 by default)."""

    wait_time = between(0.1, 0.5)

    @tag("agent")
    @task
    def agent(self) -> None:
        query = random.choice(_QUERIES)
        self.client.post(
            "/api/agent",
            json={"query": query, "top_k": 5},
            name="/api/agent",
        )
