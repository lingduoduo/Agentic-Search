"""Intent classification utilities for routing search behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from src.search.vocabulary import Vocabulary, tokenize_text

INTENT_LABELS = ["purchase", "navigate", "qa", "recommendation"]


class IntentionClassifier(nn.Module):
    """Small feedforward intent classifier over averaged token embeddings."""

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        pooled = embedded.mean(dim=1)
        hidden = self.dropout(self.relu(self.fc1(pooled)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        return self.fc3(hidden)

    def predict(self, token_ids: torch.Tensor) -> tuple[list[str], list[float]]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(token_ids)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        predicted_intents = [INTENT_LABELS[pred.item()] for pred in predictions]
        predicted_probs = [prob.item() for prob in probs.max(dim=1).values]
        return predicted_intents, predicted_probs


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float


class IntentionClassificationPipeline:
    """Trainable intent classification pipeline with shared tokenization."""

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        self.vocab = Vocabulary()
        self.model = IntentionClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=len(INTENT_LABELS))
        self.is_trained = False
        self.vocab_size = vocab_size

    def train_model(
        self,
        train_data: list[tuple[list[str], str]],
        *,
        epochs: int = 10,
        lr: float = 0.001,
        min_freq: int = 2,
    ) -> None:
        self.vocab.build([tokens for tokens, _ in train_data], min_freq=min_freq)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        label_to_id = {label: idx for idx, label in enumerate(INTENT_LABELS)}

        self.model.train()
        for _ in range(epochs):
            for tokens, label in train_data:
                token_ids = self.vocab.encode(tokens)
                if not token_ids:
                    token_ids = [0]
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.model.device)
                label_tensor = torch.tensor([label_to_id[label]], dtype=torch.long).to(self.model.device)

                optimizer.zero_grad()
                logits = self.model(token_tensor)
                loss = criterion(logits, label_tensor)
                loss.backward()
                optimizer.step()

        self.is_trained = True

    def predict(self, tokens: Sequence[str]) -> IntentPrediction:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_model first.")

        token_ids = self.vocab.encode(tokens)
        if not token_ids:
            token_ids = [0]
        token_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.model.device)
        predicted_intents, predicted_probs = self.model.predict(token_tensor)
        return IntentPrediction(intent=predicted_intents[0], confidence=predicted_probs[0])

    def predict_text(self, text: str) -> IntentPrediction:
        return self.predict(tokenize_text(text))
