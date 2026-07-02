"""Command-line index builder: builds dense (FAISS) or BM25 (pyserini) indexes
from a JSONL corpus.

This is the ``python -m`` build tool; it is intentionally kept apart from the
library indexing path (chunking/embedding/pipeline) used by the background
workers. Invocation via ``src.internal.document_index.index_builder`` still works
through the facade in that module.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from src.internal.document_index._common import (
    _require_torch,
    _require_transformers,
    _require_tqdm,
)
from src.internal.document_index.embedding import (
    _encode_batch,
    load_model,
    prepare_texts,
    resolve_pooling_method,
)
from src.internal.document_index.faiss_io import (
    load_corpus,
    write_dense_faiss_index,
)
from src.internal.document_index.text import (
    MAX_LENGTH as DEFAULT_VOCAB_MAX_LENGTH,
    Vocabulary,
    normalize_document,
    tokenize_text,
)


@dataclass(frozen=True)
class IndexBuilderConfig:
    retrieval_method: str
    corpus_path: str
    save_dir: str
    model_path: str | None = None
    max_length: int = 180
    batch_size: int = 512
    use_fp16: bool = False
    pooling_method: str = "mean"
    faiss_type: str = "Flat"
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    embedding_path: str | None = None
    save_embedding: bool = False
    faiss_gpu: bool = False
    save_vocabulary: bool = True
    keyword_limit: int = 10
    vocab_max_length: int = DEFAULT_VOCAB_MAX_LENGTH
    # 0 = auto-detect (uses all available CPUs)
    bm25_threads: int = 0

    def validate(self) -> None:
        retrieval_method = self.retrieval_method.strip().lower()
        if not retrieval_method:
            raise ValueError("retrieval_method is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if (
            retrieval_method != "bm25"
            and not self.model_path
            and not self.embedding_path
        ):
            raise ValueError(
                "model_path or embedding_path is required for dense indexing."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1.")
        if self.keyword_limit < 1:
            raise ValueError("keyword_limit must be at least 1.")
        if self.vocab_max_length < 1:
            raise ValueError("vocab_max_length must be at least 1.")
        if self.hnsw_ef_construction is not None and self.hnsw_ef_construction < 1:
            raise ValueError("hnsw_ef_construction must be at least 1.")
        if self.hnsw_ef_search is not None and self.hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be at least 1.")


class IndexBuilder:
    """Builds dense or BM25 indexes for retrieval."""

    def __init__(self, config: IndexBuilderConfig):
        config.validate()
        self.config = config
        self.retrieval_method = config.retrieval_method.lower()
        self.model_path = config.model_path
        self.corpus_path = config.corpus_path
        self.save_dir = Path(config.save_dir)
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.use_fp16 = config.use_fp16
        self.pooling_method = config.pooling_method
        self.faiss_type = config.faiss_type
        self.hnsw_ef_construction = config.hnsw_ef_construction
        self.hnsw_ef_search = config.hnsw_ef_search
        self.embedding_path = config.embedding_path
        self.save_embedding = config.save_embedding
        self.faiss_gpu = config.faiss_gpu
        self.save_vocabulary = config.save_vocabulary
        self.keyword_limit = config.keyword_limit
        self.vocab_max_length = config.vocab_max_length
        self.bm25_threads = config.bm25_threads or cpu_count()

        torch = _require_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self._prepare_save_dir()
        self.index_save_path = (
            self.save_dir / f"{self.retrieval_method}_{self.faiss_type}.index"
        )
        self.embedding_save_path = self.save_dir / f"emb_{self.retrieval_method}.memmap"
        self.vocab_save_path = self.save_dir / "vocabulary_corpus.json"
        self.corpus = load_corpus(self.corpus_path)

    def _prepare_save_dir(self) -> None:
        if self.save_dir.exists():
            if any(self.save_dir.iterdir()):
                warnings.warn(
                    f"Some files already exist in {self.save_dir} and may be overwritten.",
                    UserWarning,
                )
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self) -> None:
        if self.save_vocabulary:
            self.save_vocabulary_metadata()
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def save_vocabulary_metadata(self) -> None:
        vocabulary, corpus_entries = self._build_vocabulary_metadata()

        with self.vocab_save_path.open("w", encoding="utf-8") as vocab_file:
            json.dump(
                {
                    "corpus_path": self.corpus_path,
                    "retrieval_method": self.retrieval_method,
                    "keyword_limit": self.keyword_limit,
                    "vocab_max_length": self.vocab_max_length,
                    "vocabulary": {
                        "num_token": vocabulary.num_token,
                        "token2idx": vocabulary.token2idx,
                        "token2cnt": vocabulary.token2cnt,
                        "idx2token": {
                            str(key): value
                            for key, value in vocabulary.idx2token.items()
                        },
                    },
                    "corpus": corpus_entries,
                },
                vocab_file,
                ensure_ascii=False,
                indent=2,
            )

    def _build_vocabulary_metadata(self) -> tuple["Vocabulary", list[dict[str, Any]]]:
        vocabulary = Vocabulary()
        corpus_entries: list[dict[str, Any]] = []

        for index in range(len(self.corpus)):
            item = self.corpus[index]
            text = normalize_document(
                item,
                text_fields=("title", "contents"),
            )
            tokens = tokenize_text(
                text,
                max_length=self.vocab_max_length,
            )
            if tokens:
                vocabulary.add_token_sequence(tokens)
            token_counts = Counter(tokens)
            corpus_entries.append(
                {
                    "doc_id": index,
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "contents": text,
                    "tokens": tokens,
                    "keywords": [
                        token
                        for token, _ in token_counts.most_common(self.keyword_limit)
                    ],
                    "token_count": len(tokens),
                }
            )

        return vocabulary, corpus_entries

    def build_bm25_index(self) -> None:
        bm25_dir = self.save_dir / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(dir=bm25_dir) as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_file_path = temp_dir_path / "temp.jsonl"
            shutil.copyfile(self.corpus_path, temp_file_path)

            pyserini_args = [
                "--collection",
                "JsonCollection",
                "--input",
                str(temp_dir_path),
                "--index",
                str(bm25_dir),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(self.bm25_threads),
            ]
            subprocess.run(
                [sys.executable, "-m", "pyserini.index.lucene", *pyserini_args],
                check=True,
            )

    def _load_embedding(
        self, embedding_path: str, corpus_size: int, hidden_size: int | None = None
    ) -> np.memmap:
        if corpus_size < 1:
            raise ValueError("Cannot infer embedding dimensions for an empty corpus.")

        if hidden_size is None:
            embedding_bytes = Path(embedding_path).stat().st_size
            row_bytes = np.dtype(np.float32).itemsize * corpus_size
            if embedding_bytes % row_bytes != 0:
                raise ValueError(
                    "Embedding file size is not divisible by corpus size and float32 size."
                )
            hidden_size = embedding_bytes // row_bytes

        if hidden_size < 1:
            raise ValueError("Embedding file does not contain any float32 vectors.")

        return np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(
            corpus_size, hidden_size
        )

    def _iter_encoded_batches(
        self, encoder: Any, tokenizer: Any, batch_size: int | None = None
    ):
        tqdm = _require_tqdm()

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        for start_idx in tqdm(
            range(0, len(self.corpus), effective_batch_size),
            desc="Inference embeddings",
        ):
            yield (
                start_idx,
                _encode_batch(
                    encoder,
                    tokenizer,
                    prepare_texts(
                        self.corpus[start_idx : start_idx + effective_batch_size][
                            "contents"
                        ],
                        self.retrieval_method,
                        is_query=False,
                    ),
                    self.retrieval_method,
                    self.max_length,
                    self.pooling_method,
                    self.device,
                ),
            )

    def encode_all(self, encoder: Any, tokenizer: Any) -> np.ndarray:
        torch = _require_torch()
        batch_size = self.batch_size
        if self.gpu_num > 1 and self.device.startswith("cuda"):
            encoder = torch.nn.DataParallel(encoder)
            batch_size *= self.gpu_num

        all_embeddings: np.ndarray | None = None
        write_index = 0
        for _, batch_embeddings in self._iter_encoded_batches(
            encoder, tokenizer, batch_size=batch_size
        ):
            if all_embeddings is None:
                all_embeddings = np.empty(
                    (len(self.corpus), batch_embeddings.shape[1]),
                    dtype=batch_embeddings.dtype,
                )
            stop_index = write_index + batch_embeddings.shape[0]
            all_embeddings[write_index:stop_index] = batch_embeddings
            write_index = stop_index

        if all_embeddings is None:
            return np.empty((0, 0), dtype=np.float32)
        return all_embeddings

    def encode_all_to_memmap(self, encoder: Any, tokenizer: Any) -> np.memmap:
        torch = _require_torch()
        batch_size = self.batch_size
        if self.gpu_num > 1 and self.device.startswith("cuda"):
            encoder = torch.nn.DataParallel(encoder)
            batch_size *= self.gpu_num

        memmap: np.memmap | None = None
        write_index = 0
        for _, batch_embeddings in self._iter_encoded_batches(
            encoder, tokenizer, batch_size=batch_size
        ):
            if memmap is None:
                memmap = np.memmap(
                    self.embedding_save_path,
                    shape=(len(self.corpus), batch_embeddings.shape[1]),
                    mode="w+",
                    dtype=batch_embeddings.dtype,
                )
            stop_index = write_index + batch_embeddings.shape[0]
            memmap[write_index:stop_index] = batch_embeddings
            write_index = stop_index

        if memmap is None:
            raise ValueError("Cannot build a dense index for an empty corpus.")
        memmap.flush()
        return memmap

    def build_dense_index(self) -> None:
        is_hnsw = "HNSW" in self.faiss_type.upper()
        if self.faiss_gpu and is_hnsw:
            raise ValueError(
                "faiss_gpu=True is not compatible with HNSW indexes. "
                "FAISS cannot GPU-shard HNSW indexes. "
                "Use faiss_type='Flat' or an IVF variant for GPU indexing."
            )

        torch = _require_torch()

        if self.index_save_path.exists():
            warnings.warn(
                f"{self.index_save_path} already exists and will be overwritten.",
                UserWarning,
            )

        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            hidden_size: int | None = None
            if self.model_path:
                auto_config, _, _ = _require_transformers()
                hidden_size = auto_config.from_pretrained(
                    self.model_path, trust_remote_code=True
                ).hidden_size
            all_embeddings = self._load_embedding(
                self.embedding_path, corpus_size, hidden_size
            )
        else:
            encoder, tokenizer = load_model(
                model_path=self.model_path or "",
                use_fp16=self.use_fp16,
                device=self.device,
            )
            with torch.no_grad():
                if self.save_embedding:
                    all_embeddings = self.encode_all_to_memmap(encoder, tokenizer)
                else:
                    all_embeddings = self.encode_all(encoder, tokenizer)
            del self.corpus

        write_dense_faiss_index(
            all_embeddings,
            self.index_save_path,
            faiss_type=self.faiss_type,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
            faiss_gpu=self.faiss_gpu,
            gpu_num=self.gpu_num,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a retrieval index.")
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="indexes/")
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_type", type=str, default="Flat")
    parser.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=None,
        help=(
            "HNSW efConstruction: controls index build quality. "
            "Higher values improve recall but slow down indexing. "
            "Use with --faiss_type HNSW64."
        ),
    )
    parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=None,
        help=(
            "HNSW efSearch: controls recall vs. latency at query time. "
            "Higher values improve recall but slow down search. "
            "Stored in the index file and used by the retrieval server."
        ),
    )
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument(
        "--save_vocabulary", dest="save_vocabulary", action="store_true", default=True
    )
    parser.add_argument(
        "--no_save_vocabulary", dest="save_vocabulary", action="store_false"
    )
    parser.add_argument("--keyword_limit", type=int, default=10)
    parser.add_argument(
        "--vocab_max_length", type=int, default=DEFAULT_VOCAB_MAX_LENGTH
    )
    parser.add_argument(
        "--bm25_threads",
        type=int,
        default=0,
        help="BM25 indexing threads (0 = auto-detect CPUs)",
    )
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    args = parse_args()
    config = IndexBuilderConfig(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=resolve_pooling_method(
            args.retrieval_method, args.pooling_method
        ),
        faiss_type=args.faiss_type,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        save_vocabulary=args.save_vocabulary,
        keyword_limit=args.keyword_limit,
        vocab_max_length=args.vocab_max_length,
        bm25_threads=args.bm25_threads,
    )
    IndexBuilder(config).build_index()


if __name__ == "__main__":
    main()
