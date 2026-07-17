"""A small synthetic search environment for the classical-RL (Q-learning) demo.

Replaces the treasure-hunt world the sampled ``rl_agent.py`` was written for.
Topics each hold one fact; each question needs a subset of facts. The agent
retrieves facts and must answer once it holds every required fact.
"""

from __future__ import annotations

import random


TOPICS: dict[str, str] = {
    "faiss": "faiss is a dense vector index",
    "bm25": "bm25 is a sparse lexical ranker",
    "rrf": "rrf fuses ranked lists",
    "e5": "e5 is a dense embedding model",
    "chunking": "chunking splits documents",
    "rerank": "a cross-encoder reranks candidates",
}

QUESTIONS: list[dict] = [
    {"id": "what_is_faiss", "required_facts": {"faiss"}, "answer": "faiss"},
    {"id": "dense_vs_sparse", "required_facts": {"faiss", "bm25"}, "answer": "hybrid"},
    {"id": "hybrid_pipeline", "required_facts": {"bm25", "e5", "rrf"}, "answer": "rrf"},
    {
        "id": "rerank_stage",
        "required_facts": {"chunking", "rerank"},
        "answer": "rerank",
    },
]


class SearchEnvironment:
    """Synthetic retrieve-until-sufficient environment.

    Actions: ``retrieve:<topic>`` for each topic, plus ``answer`` and ``stop``.
    """

    def __init__(
        self,
        stochastic: bool = False,
        max_steps: int = 20,
        seed: int | None = None,
        fail_prob: float = 0.2,
    ):
        self.stochastic = stochastic
        self.max_steps = max_steps
        self.fail_prob = fail_prob
        self._rng = random.Random(seed)

        self.topics = dict(TOPICS)
        self.questions = [
            {
                "id": q["id"],
                "required_facts": set(q["required_facts"]),
                "answer": q["answer"],
            }
            for q in QUESTIONS
        ]

        self._question_idx = 0
        self.current_question = self.questions[0]["id"]
        self.gathered: set[str] = set()
        self.steps = 0
        self.game_over = False
        self.victory = False

    def _current(self) -> dict:
        return next(q for q in self.questions if q["id"] == self.current_question)

    def reset(self) -> None:
        self.current_question = self.questions[self._question_idx]["id"]
        self._question_idx = (self._question_idx + 1) % len(self.questions)
        self.gathered = set()
        self.steps = 0
        self.game_over = False
        self.victory = False

    def get_available_actions(self) -> list[str]:
        return [f"retrieve:{t}" for t in self.topics] + ["answer", "stop"]
