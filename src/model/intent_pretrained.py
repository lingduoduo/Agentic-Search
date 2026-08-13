"""The frozen pretrained embedding bundle the intent model reads.

Extraction needs sentence-transformers; loading needs only numpy. The split
matters: extraction runs once offline, while loading runs wherever the model is
trained or served.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .wordpiece import WordPieceVocabulary

VOCAB_FILENAME = "vocab.txt"
EMBEDDINGS_FILENAME = "embeddings.fp16.npy"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class PretrainedBundle:
    """A wordpiece vocabulary and the frozen matrix its ids index into."""

    vocabulary: WordPieceVocabulary
    embeddings: np.ndarray

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


def write_pretrained_bundle(
    directory: Path, *, tokens: Sequence[str], embeddings: np.ndarray
) -> None:
    """Write vocab.txt and the fp16 matrix, creating the directory if needed."""
    if embeddings.dtype != np.float16:
        raise ValueError(
            f"Pretrained embeddings must be float16, got {embeddings.dtype}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / VOCAB_FILENAME).write_text("\n".join(tokens) + "\n", encoding="utf-8")
    np.save(directory / EMBEDDINGS_FILENAME, embeddings)


def load_pretrained_bundle(directory: Path) -> PretrainedBundle:
    """Load and validate a bundle written by ``write_pretrained_bundle``."""
    vocabulary_path = directory / VOCAB_FILENAME
    embeddings_path = directory / EMBEDDINGS_FILENAME
    for path in (embeddings_path, vocabulary_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained bundle is missing {path.name}: {directory}. Run "
                "`python -m src.model.intent_training embeddings` to create it."
            )

    vocabulary = WordPieceVocabulary.from_file(vocabulary_path)
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[0] != vocabulary.size:
        raise ValueError(
            "Pretrained matrix rows must equal the vocabulary size: "
            f"{embeddings.shape} rows against {vocabulary.size} tokens"
        )
    return PretrainedBundle(vocabulary=vocabulary, embeddings=embeddings)


def extract_pretrained_bundle(
    model_name: str = DEFAULT_MODEL, directory: Path = Path("data/intent_pretrained")
) -> None:
    """Pull the tokenizer vocabulary and input embedding matrix from a model.

    Only the embedding table is taken. The transformer itself never runs, at
    training time or at serving time.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    vocabulary = model.tokenizer.get_vocab()
    tokens = [token for token, _ in sorted(vocabulary.items(), key=lambda kv: kv[1])]
    weights = model[0].auto_model.embeddings.word_embeddings.weight
    embeddings = weights.detach().cpu().numpy().astype(np.float16)
    if embeddings.shape[0] != len(tokens):
        raise ValueError(
            "Model vocabulary and embedding matrix disagree: "
            f"{len(tokens)} tokens against {embeddings.shape[0]} rows"
        )
    write_pretrained_bundle(directory, tokens=tokens, embeddings=embeddings)
