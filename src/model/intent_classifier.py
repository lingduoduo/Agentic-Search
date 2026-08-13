"""Intent classification utilities for query routing."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .intent_pretrained import PretrainedBundle
from .wordpiece import PAD_ID, UNK_ID, WordPieceVocabulary

INTENT_LABELS: list[str] = ["chat", "search", "tool"]

# BERT's own layout, because the ids index a pretrained matrix: padding stays 0
# so masked-mean pooling keeps ignoring it, and unknown words take 100.
PADDING_ID = PAD_ID
UNKNOWN_ID = UNK_ID


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float


class _IntentClassifier:
    def __init__(
        self,
        embedding_matrix: "np.ndarray",
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        embedding_dim = weights.shape[1]

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Frozen: with a few hundred training examples, fine-tuning
                # these would overwrite the pretrained semantics that make
                # unseen words readable in the first place.
                self.embedding = nn.Embedding.from_pretrained(
                    weights, freeze=True, padding_idx=PADDING_ID
                )
                self.fc1 = nn.Linear(embedding_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
                self.drop = nn.Dropout(0.3)

            def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
                import torch.nn.functional as F

                mask = ids.ne(PADDING_ID).unsqueeze(-1)
                embedded = self.embedding(ids)
                x = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                x = self.drop(F.relu(self.fc1(x)))
                x = self.drop(F.relu(self.fc2(x)))
                return self.fc3(x)

        self._net = _Net().to(self._device)

    def _pad_sequences(self, encoded: list[list[int]]):
        max_len = max(len(ids) for ids in encoded) or 1
        ids_tensor = self._torch.zeros(
            len(encoded), max_len, dtype=self._torch.long, device=self._device
        )
        for i, ids in enumerate(encoded):
            ids_tensor[i, : len(ids)] = self._torch.tensor(ids, dtype=self._torch.long)
        return ids_tensor

    def train_batched(
        self,
        encoded: list[list[int]],
        labels: list[int],
        *,
        epochs: int,
        lr: float,
    ) -> None:
        import torch.nn as nn

        optimizer = self._torch.optim.Adam(
            (p for p in self._net.parameters() if p.requires_grad), lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        self._net.train()

        ids_tensor = self._pad_sequences(encoded)
        labels_tensor = self._torch.tensor(
            labels, dtype=self._torch.long, device=self._device
        )

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self._net(ids_tensor), labels_tensor)
            loss.backward()
            optimizer.step()

    def predict_batch(self, encoded: list[list[int]]) -> list[IntentPrediction]:
        self._net.eval()
        ids_tensor = self._pad_sequences(encoded)

        with self._torch.no_grad():
            logits = self._net(ids_tensor)
            probs = self._torch.softmax(logits, dim=1)
            top_probs, top_idx = probs.max(dim=1)

        return [
            IntentPrediction(intent=INTENT_LABELS[idx.item()], confidence=prob.item())
            for idx, prob in zip(top_idx, top_probs)
        ]


class IntentPipeline:
    def __init__(self, bundle: "PretrainedBundle", *, hidden_dim: int = 256) -> None:
        self._bundle = bundle
        self._hidden_dim = hidden_dim
        self._model = self._new_model()
        self._label_to_id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.is_trained = False

    def _encode_text(self, text: str) -> list[int]:
        """Encode one request as wordpiece ids.

        Reading no tokens is a fact about the input, not padding, so an empty
        result becomes a single [UNK] rather than an empty sequence.
        """
        return self._bundle.vocabulary.encode(text) or [UNKNOWN_ID]

    def train(
        self,
        data: list[tuple[list[str], str]],
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        seed: int = 17,
    ) -> None:
        """Train the head on (token_list, intent_label) pairs."""
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._model = self._new_model()
        encoded = [self._encode_text(" ".join(tokens)) for tokens, _ in data]
        labels = [self._label_to_id[label] for _, label in data]
        self._model.train_batched(encoded, labels, epochs=epochs, lr=lr)
        self.is_trained = True

    def predict(self, tokens: Sequence[str]) -> IntentPrediction:
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        return self._model.predict_batch([self._encode_text(" ".join(tokens))])[0]

    def predict_text(self, text: str) -> IntentPrediction:
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        return self._model.predict_batch([self._encode_text(text)])[0]

    def save(
        self,
        path: str,
        *,
        dataset_fingerprint: str,
        promoted_min_confidence: float | None = None,
    ) -> None:
        import torch

        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before saving.")
        if promoted_min_confidence is not None and (
            not math.isfinite(promoted_min_confidence)
            or not 0.0 <= promoted_min_confidence <= 1.0
        ):
            raise ValueError(
                "promoted_min_confidence must be null or a finite probability"
            )
        checkpoint = {
            "version": 4,
            "intent_labels": list(INTENT_LABELS),
            "preprocessing": {
                "tokenizer": "wordpiece",
                "padding_id": PADDING_ID,
                "unknown_id": UNKNOWN_ID,
                "pooling": "masked_mean",
                "embeddings": "frozen_pretrained",
            },
            "dataset_fingerprint": dataset_fingerprint,
            "promoted_min_confidence": promoted_min_confidence,
            "vocab_tokens": self._bundle.vocabulary.tokens,
            "embeddings": torch.tensor(self._bundle.embeddings),
            "model_state": self._model._net.state_dict(),
            "config": {
                "vocab_size": self._bundle.size,
                "embedding_dim": self._bundle.dim,
                "hidden_dim": self._model._net.fc1.out_features,
                "num_classes": len(INTENT_LABELS),
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "IntentPipeline":
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        version = checkpoint.get("version")
        if version in (1, 2, 3):
            raise ValueError(
                f"Checkpoint version {version} predates pretrained wordpiece "
                "embeddings; retrain the intent model before loading it."
            )
        if version != 4:
            raise ValueError(f"Unsupported checkpoint version: {version}")
        if checkpoint.get("intent_labels") != INTENT_LABELS:
            raise ValueError(
                "Checkpoint intent_labels do not match the supported label order."
            )
        expected_preprocessing = {
            "tokenizer": "wordpiece",
            "padding_id": PADDING_ID,
            "unknown_id": UNKNOWN_ID,
            "pooling": "masked_mean",
            "embeddings": "frozen_pretrained",
        }
        if checkpoint.get("preprocessing") != expected_preprocessing:
            raise ValueError("Checkpoint preprocessing contract is unsupported.")
        if not isinstance(checkpoint.get("dataset_fingerprint"), str):
            raise ValueError("Checkpoint dataset_fingerprint must be a string.")
        promoted_min_confidence = checkpoint.get("promoted_min_confidence")
        if promoted_min_confidence is not None and (
            isinstance(promoted_min_confidence, bool)
            or not isinstance(promoted_min_confidence, (int, float))
            or not math.isfinite(float(promoted_min_confidence))
            or not 0.0 <= float(promoted_min_confidence) <= 1.0
        ):
            raise ValueError(
                "Checkpoint promoted_min_confidence must be null or a finite "
                "probability."
            )

        cfg = checkpoint["config"]
        cls._validate_checkpoint_dimensions(cfg, checkpoint["model_state"])
        bundle = PretrainedBundle(
            vocabulary=WordPieceVocabulary.from_tokens(checkpoint["vocab_tokens"]),
            embeddings=checkpoint["embeddings"].numpy().astype(np.float16),
        )
        pipeline = cls(bundle, hidden_dim=cfg["hidden_dim"])
        pipeline._model._net.load_state_dict(checkpoint["model_state"])
        pipeline._model._net.eval()
        pipeline.is_trained = True
        pipeline.promoted_min_confidence = (
            float(promoted_min_confidence)
            if promoted_min_confidence is not None
            else None
        )
        return pipeline

    def _new_model(self) -> _IntentClassifier:
        return _IntentClassifier(
            self._bundle.embeddings, self._hidden_dim, len(INTENT_LABELS)
        )

    @staticmethod
    def _validate_checkpoint_dimensions(
        config: dict[str, Any], model_state: dict[str, Any]
    ) -> None:
        required_config = ("vocab_size", "embedding_dim", "hidden_dim", "num_classes")
        if (
            not isinstance(config, dict)
            or any(
                not isinstance(config.get(key), int) or config[key] <= 0
                for key in required_config
            )
            or config["num_classes"] != len(INTENT_LABELS)
        ):
            raise ValueError(
                "Checkpoint config is invalid for the intent architecture."
            )

        hidden_half = config["hidden_dim"] // 2
        expected_shapes = {
            "embedding.weight": (config["vocab_size"], config["embedding_dim"]),
            "fc1.weight": (config["hidden_dim"], config["embedding_dim"]),
            "fc1.bias": (config["hidden_dim"],),
            "fc2.weight": (hidden_half, config["hidden_dim"]),
            "fc2.bias": (hidden_half,),
            "fc3.weight": (config["num_classes"], hidden_half),
            "fc3.bias": (config["num_classes"],),
        }
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint config/model state dimensions are invalid.")
        for name, expected_shape in expected_shapes.items():
            tensor = model_state.get(name)
            if tensor is None or tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    "Checkpoint config/model state dimensions do not match."
                )


def resolve_search_settings(
    prediction: IntentPrediction,
    *,
    topk: int,
    max_search_limit: int,
    require_evidence: bool,
    allow_internal_knowledge: bool,
    min_confidence: float = 0.6,
) -> tuple[int, int, bool, bool, dict[str, Any]]:
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
        "chat": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
        "search": (max(topk, 8), max(max_search_limit, 3), True, False),
        "tool": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
    }
    t, s, r, a = policy.get(
        prediction.intent,
        (topk, max_search_limit, require_evidence, allow_internal_knowledge),
    )
    return t, s, r, a, meta


IntentionClassificationPipeline = IntentPipeline


def load_training_data(path: str) -> list[tuple[list[str], str]]:
    from src.internal.document_index.text import tokenize_text

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training data file not found: {path!r}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training data file is not valid JSON: {path!r}") from exc

    examples: list[tuple[list[str], str]] = []
    for item in raw:
        text = item.get("text") or item.get("question") or ""
        label = item.get("label") or item.get("intent")
        if text and label:
            examples.append((tokenize_text(text), str(label)))
    return examples
