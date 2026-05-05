"""Intent classification utilities for routing search behavior.

Public surface
--------------
INTENT_LABELS       — canonical list of supported intent strings
IntentPrediction    — (intent, confidence) result
IntentPipeline      — train, predict, and resolve search settings
load_training_data  — load a JSON examples file into (tokens, label) pairs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

INTENT_LABELS: list[str] = ["purchase", "navigate", "qa", "recommendation"]


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float


# ---------------------------------------------------------------------------
# Neural model (private — users go through IntentPipeline)
# ---------------------------------------------------------------------------


class _IntentClassifier:
    """Small feedforward classifier over averaged token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.fc1 = nn.Linear(embedding_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
                self.drop = nn.Dropout(0.3)

            def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
                import torch.nn.functional as F

                x = self.embedding(ids).mean(dim=1)
                x = self.drop(F.relu(self.fc1(x)))
                x = self.drop(F.relu(self.fc2(x)))
                return self.fc3(x)

        self._net = _Net().to(self._device)

    def train_batched(
        self,
        encoded: list[list[int]],
        labels: list[int],
        *,
        epochs: int,
        lr: float,
    ) -> None:
        import torch
        import torch.nn as nn

        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self._net.train()

        # Pad sequences in a single batch tensor for efficiency.
        max_len = max(len(ids) for ids in encoded) or 1
        ids_tensor = torch.zeros(
            len(encoded), max_len, dtype=torch.long, device=self._device
        )
        for i, ids in enumerate(encoded):
            ids_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self._device)

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self._net(ids_tensor), labels_tensor)
            loss.backward()
            optimizer.step()

    def predict_batch(self, encoded: list[list[int]]) -> list[IntentPrediction]:
        import torch

        self._net.eval()
        max_len = max(len(ids) for ids in encoded) or 1
        ids_tensor = torch.zeros(
            len(encoded), max_len, dtype=torch.long, device=self._device
        )
        for i, ids in enumerate(encoded):
            ids_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        with torch.no_grad():
            logits = self._net(ids_tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idx = probs.max(dim=1)

        return [
            IntentPrediction(intent=INTENT_LABELS[idx.item()], confidence=prob.item())
            for idx, prob in zip(top_idx, top_probs)
        ]


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


class IntentPipeline:
    """Trainable intent classification pipeline.

    Typical usage::

        pipeline = IntentPipeline()
        data = load_training_data("data/intent_examples.json")
        pipeline.train(data)
        pred = pipeline.predict_text("What is FAISS?")
        # pred.intent == "qa", pred.confidence ≈ 0.85
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        from src.search.vocabulary import Vocabulary

        self._vocab = Vocabulary()
        self._model = _IntentClassifier(
            vocab_size, embedding_dim, hidden_dim, len(INTENT_LABELS)
        )
        self._label_to_id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.is_trained = False

    def train(
        self,
        data: list[tuple[list[str], str]],
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        min_freq: int = 2,
    ) -> None:
        """Train on (token_list, intent_label) pairs."""
        self._vocab.build([tokens for tokens, _ in data], min_freq=min_freq)
        encoded = [self._vocab.encode(tokens) or [0] for tokens, _ in data]
        labels = [self._label_to_id[label] for _, label in data]
        self._model.train_batched(encoded, labels, epochs=epochs, lr=lr)
        self.is_trained = True

    def predict(self, tokens: Sequence[str]) -> IntentPrediction:
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        encoded = self._vocab.encode(list(tokens)) or [0]
        return self._model.predict_batch([encoded])[0]

    def predict_text(self, text: str) -> IntentPrediction:
        from src.search.vocabulary import tokenize_text

        return self.predict(tokenize_text(text))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained pipeline to *path* (.pt file).

        Stores the vocabulary mappings, model weights, and the hyperparameters
        needed to reconstruct the architecture on load.
        """
        import torch

        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before saving.")
        checkpoint = {
            "version": 1,
            "intent_labels": INTENT_LABELS,
            "vocab": {
                "token2idx": self._vocab.token2idx,
                "token2cnt": self._vocab.token2cnt,
                "idx2token": self._vocab.idx2token,
            },
            "model_state": self._model._net.state_dict(),
            "config": {
                "vocab_size": self._model._net.embedding.num_embeddings,
                "embedding_dim": self._model._net.embedding.embedding_dim,
                "hidden_dim": self._model._net.fc1.out_features,
                "num_classes": len(INTENT_LABELS),
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "IntentPipeline":
        """Load a previously saved pipeline from *path*."""
        import torch
        from src.search.vocabulary import Vocabulary

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError(
                f"Unsupported checkpoint version: {checkpoint.get('version')}"
            )

        cfg = checkpoint["config"]
        pipeline = cls(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
        )
        # Restore vocabulary
        vocab = Vocabulary()
        vocab.token2idx = checkpoint["vocab"]["token2idx"]
        vocab.token2cnt = checkpoint["vocab"]["token2cnt"]
        vocab.idx2token = checkpoint["vocab"]["idx2token"]
        pipeline._vocab = vocab
        # Restore model weights
        pipeline._model._net.load_state_dict(checkpoint["model_state"])
        pipeline._model._net.eval()
        pipeline.is_trained = True
        return pipeline

    def resolve_search_settings(
        self,
        question: str,
        *,
        topk: int,
        max_search_limit: int,
        require_evidence: bool,
        allow_internal_knowledge: bool,
        min_confidence: float = 0.6,
    ) -> tuple[int, int, bool, bool, dict[str, Any]]:
        """Predict the intent for *question* and return adjusted search settings."""
        pred = self.predict_text(question)
        return resolve_search_settings(
            pred,
            topk=topk,
            max_search_limit=max_search_limit,
            require_evidence=require_evidence,
            allow_internal_knowledge=allow_internal_knowledge,
            min_confidence=min_confidence,
        )


# ---------------------------------------------------------------------------
# Module-level routing helper (testable without a trained pipeline)
# ---------------------------------------------------------------------------


def resolve_search_settings(
    prediction: IntentPrediction,
    *,
    topk: int,
    max_search_limit: int,
    require_evidence: bool,
    allow_internal_knowledge: bool,
    min_confidence: float = 0.6,
) -> tuple[int, int, bool, bool, dict[str, Any]]:
    """Map an IntentPrediction to adjusted search settings.

    Returns (topk, max_search_limit, require_evidence, allow_internal, metadata).
    """
    meta: dict[str, Any] = {
        "intent_routing_used": True,
        "predicted_intent": prediction.intent,
        "intent_confidence": prediction.confidence,
    }
    if prediction.confidence < min_confidence:
        meta["intent_policy_applied"] = False
        return topk, max_search_limit, require_evidence, allow_internal_knowledge, meta

    meta["intent_policy_applied"] = True
    policy: dict[str, tuple[int, int, bool, bool]] = {
        "qa": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
        "navigate": (max(topk, 5), max(max_search_limit, 2), True, False),
        "purchase": (max(topk, 8), max(max_search_limit, 2), True, False),
        "recommendation": (max(topk, 8), max(max_search_limit, 3), True, False),
    }
    t, s, r, a = policy.get(
        prediction.intent,
        (topk, max_search_limit, require_evidence, allow_internal_knowledge),
    )
    return t, s, r, a, meta


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

IntentionClassificationPipeline = IntentPipeline


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(path: str) -> list[tuple[list[str], str]]:
    """Load a JSON examples file into (token_list, intent_label) pairs.

    Accepts items with either a ``text`` / ``question`` field and a
    ``label`` / ``intent`` field.
    """
    from src.search.vocabulary import tokenize_text

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    examples: list[tuple[list[str], str]] = []
    for item in raw:
        text = item.get("text") or item.get("question") or ""
        label = item.get("label") or item.get("intent")
        if text and label:
            examples.append((tokenize_text(text), str(label)))
    return examples
