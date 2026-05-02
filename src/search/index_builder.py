"""Utilities for building dense or BM25 retrieval indexes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import datasets
    import faiss
    import torch
    from transformers import AutoModel, AutoTokenizer

MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
}


def _require_torch():
    import torch

    return torch


def _require_datasets():
    import datasets

    return datasets


def _require_faiss():
    import faiss

    return faiss


def _require_transformers():
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    return AutoConfig, AutoModel, AutoTokenizer


def _require_tqdm():
    from tqdm import tqdm

    return tqdm


def prepare_texts(texts: list[str], retrieval_method: str, *, is_query: bool) -> list[str]:
    normalized_method = retrieval_method.lower()
    prepared = list(texts)

    if "e5" in normalized_method:
        prefix = "query" if is_query else "passage"
        prepared = [f"{prefix}: {text}" for text in prepared]

    if "bge" in normalized_method and is_query:
        prepared = [
            f"Represent this sentence for searching relevant passages: {text}"
            for text in prepared
        ]

    return prepared


def load_model(model_path: str, use_fp16: bool = False, device: str = "cpu") -> tuple[Any, Any]:
    _, auto_model, auto_tokenizer = _require_transformers()
    model = auto_model.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16 and device.startswith("cuda"):
        model = model.half()
    tokenizer = auto_tokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
    pooler_output: Any,
    last_hidden_state: Any,
    attention_mask: Any | None = None,
    pooling_method: str = "mean",
) -> Any:
    if pooling_method == "mean":
        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling.")
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str):
    datasets = _require_datasets()
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


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
    embedding_path: str | None = None
    save_embedding: bool = False
    faiss_gpu: bool = False

    def validate(self) -> None:
        retrieval_method = self.retrieval_method.strip().lower()
        if not retrieval_method:
            raise ValueError("retrieval_method is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if retrieval_method != "bm25" and not self.model_path and not self.embedding_path:
            raise ValueError("model_path or embedding_path is required for dense indexing.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1.")


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
        self.embedding_path = config.embedding_path
        self.save_embedding = config.save_embedding
        self.faiss_gpu = config.faiss_gpu

        torch = _require_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self._prepare_save_dir()
        self.index_save_path = self.save_dir / f"{self.retrieval_method}_{self.faiss_type}.index"
        self.embedding_save_path = self.save_dir / f"emb_{self.retrieval_method}.memmap"
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
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

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
                "1",
            ]
            subprocess.run(
                ["python", "-m", "pyserini.index.lucene", *pyserini_args],
                check=True,
            )

    def _load_embedding(self, embedding_path: str, corpus_size: int, hidden_size: int) -> np.memmap:
        return np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(corpus_size, hidden_size)

    def _save_embedding(self, all_embeddings: np.ndarray) -> None:
        tqdm = _require_tqdm()
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype,
        )
        save_batch_size = 10000
        for start in tqdm(
            range(0, all_embeddings.shape[0], save_batch_size),
            leave=False,
            desc="Saving embeddings",
        ):
            stop = min(start + save_batch_size, all_embeddings.shape[0])
            memmap[start:stop] = all_embeddings[start:stop]
        memmap.flush()

    def encode_all(self, encoder: Any, tokenizer: Any) -> np.ndarray:
        torch = _require_torch()
        tqdm = _require_tqdm()

        batch_size = self.batch_size
        if self.gpu_num > 1 and self.device.startswith("cuda"):
            encoder = torch.nn.DataParallel(encoder)
            batch_size *= self.gpu_num

        all_embeddings = []
        for start_idx in tqdm(range(0, len(self.corpus), batch_size), desc="Inference embeddings"):
            batch_data = self.corpus[start_idx : start_idx + batch_size]["contents"]
            batch_data = prepare_texts(batch_data, self.retrieval_method, is_query=False)

            inputs = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            if "T5" in type(encoder).__name__:
                decoder_input_ids = torch.zeros(
                    (inputs["input_ids"].shape[0], 1),
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                )
                output = encoder(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
                embeddings = output.last_hidden_state[:, 0, :]
            else:
                output = encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    output.pooler_output,
                    output.last_hidden_state,
                    inputs["attention_mask"],
                    self.pooling_method,
                )
                if "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = cast(Any, embeddings).detach().cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def build_dense_index(self) -> None:
        faiss = _require_faiss()
        torch = _require_torch()

        if self.index_save_path.exists():
            warnings.warn(f"{self.index_save_path} already exists and will be overwritten.", UserWarning)

        encoder, tokenizer = load_model(
            model_path=self.model_path or "",
            use_fp16=self.use_fp16,
            device=self.device,
        )

        if self.embedding_path is not None:
            hidden_size = encoder.config.hidden_size
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            with torch.no_grad():
                all_embeddings = self.encode_all(encoder, tokenizer)
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            if not hasattr(faiss, "GpuMultipleClonerOptions") or self.gpu_num == 0:
                raise RuntimeError("faiss_gpu was requested, but GPU FAISS support is not available.")
            clone_options = faiss.GpuMultipleClonerOptions()
            clone_options.useFloat16 = True
            clone_options.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, clone_options)

        if not faiss_index.is_trained:
            faiss_index.train(all_embeddings)
        faiss_index.add(all_embeddings)

        if self.faiss_gpu:
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)

        faiss.write_index(faiss_index, str(self.index_save_path))


def resolve_pooling_method(retrieval_method: str, pooling_method: str | None) -> str:
    if pooling_method is None:
        for model_key, default_pooling in MODEL2POOLING.items():
            if model_key in retrieval_method.lower():
                return default_pooling
        return "mean"
    if pooling_method not in {"mean", "cls", "pooler"}:
        raise NotImplementedError("pooling_method must be one of: mean, cls, pooler")
    return pooling_method


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
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IndexBuilderConfig(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=resolve_pooling_method(args.retrieval_method, args.pooling_method),
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
    )
    IndexBuilder(config).build_index()


if __name__ == "__main__":
    main()
