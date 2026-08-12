"""Intent classification utilities for query routing."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Sequence

INTENT_LABELS: list[str] = ["chat", "search", "tool"]


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float


class _IntentClassifier:
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

                mask = ids.ne(0).unsqueeze(-1)
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

        optimizer = self._torch.optim.Adam(self._net.parameters(), lr=lr)
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
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        from src.internal.document_index.text import Vocabulary

        self._vocab = Vocabulary()
        self._model_config = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
        }
        self._model = self._new_model()
        self._label_to_id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.is_trained = False

    def train(
        self,
        data: list[tuple[list[str], str]],
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        min_freq: int = 2,
        seed: int = 17,
    ) -> None:
        """Train on (token_list, intent_label) pairs."""
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._model = self._new_model()
        self._vocab.build([tokens for tokens, _ in data], min_freq=min_freq)
        max_vocab_index = max(self._vocab.token2idx.values(), default=0)
        if max_vocab_index >= self._model._net.embedding.num_embeddings:
            raise ValueError(
                "Built vocabulary exceeds configured vocab_size "
                f"{self._model._net.embedding.num_embeddings}: "
                f"maximum token index is {max_vocab_index}"
            )
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
        from src.internal.document_index.text import tokenize_text

        return self.predict(tokenize_text(text))

    def save(self, path: str, *, dataset_fingerprint: str) -> None:
        import torch

        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before saving.")
        checkpoint = {
            "version": 2,
            "intent_labels": list(INTENT_LABELS),
            "preprocessing": {
                "tokenizer": "document_index.tokenize_text",
                "padding_id": 0,
                "pooling": "masked_mean",
            },
            "dataset_fingerprint": dataset_fingerprint,
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
        import torch
        from src.internal.document_index.text import Vocabulary

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") == 1:
            raise ValueError(
                "Checkpoint version 1 uses unsafe legacy class indices; retrain "
                "the intent model before loading it."
            )
        if checkpoint.get("version") != 2:
            raise ValueError(
                f"Unsupported checkpoint version: {checkpoint.get('version')}"
            )
        if checkpoint.get("intent_labels") != INTENT_LABELS:
            raise ValueError(
                "Checkpoint intent_labels do not match the supported label order."
            )
        expected_preprocessing = {
            "tokenizer": "document_index.tokenize_text",
            "padding_id": 0,
            "pooling": "masked_mean",
        }
        if checkpoint.get("preprocessing") != expected_preprocessing:
            raise ValueError("Checkpoint preprocessing contract is unsupported.")
        if not isinstance(checkpoint.get("dataset_fingerprint"), str):
            raise ValueError("Checkpoint dataset_fingerprint must be a string.")

        cfg = checkpoint["config"]
        cls._validate_checkpoint_dimensions(cfg, checkpoint["model_state"])
        pipeline = cls(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
        )
        vocab = Vocabulary()
        vocab.token2idx = checkpoint["vocab"]["token2idx"]
        vocab.token2cnt = checkpoint["vocab"]["token2cnt"]
        vocab.idx2token = checkpoint["vocab"]["idx2token"]
        pipeline._vocab = vocab
        pipeline._model._net.load_state_dict(checkpoint["model_state"])
        pipeline._model._net.eval()
        pipeline.is_trained = True
        return pipeline

    def _new_model(self) -> _IntentClassifier:
        return _IntentClassifier(
            self._model_config["vocab_size"],
            self._model_config["embedding_dim"],
            self._model_config["hidden_dim"],
            len(INTENT_LABELS),
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
