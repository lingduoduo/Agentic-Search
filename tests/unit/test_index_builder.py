"""Unit tests for the document-index builder's pure-logic functions."""

import pytest

from src.internal.document_index.index_builder import (
    IndexBuilder,
    IndexBuilderConfig,
    _Corpus,
    prepare_texts,
    resolve_pooling_method,
    set_hnsw_ef_construction,
    set_hnsw_ef_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> IndexBuilderConfig:
    defaults = dict(
        retrieval_method="e5",
        corpus_path="/fake/corpus.jsonl",
        save_dir="/fake/save",
        model_path="/fake/model",
    )
    defaults.update(overrides)
    return IndexBuilderConfig(**defaults)


# ---------------------------------------------------------------------------
# IndexBuilderConfig.validate
# ---------------------------------------------------------------------------


class TestIndexBuilderConfigValidate:
    def test_valid_dense_config(self):
        _config().validate()  # must not raise

    def test_whitespace_retrieval_method_raises(self):
        with pytest.raises(ValueError, match="retrieval_method"):
            _config(retrieval_method="   ").validate()

    def test_empty_retrieval_method_raises(self):
        with pytest.raises(ValueError, match="retrieval_method"):
            _config(retrieval_method="").validate()

    def test_empty_corpus_path_raises(self):
        with pytest.raises(ValueError, match="corpus_path"):
            _config(corpus_path="").validate()

    def test_dense_without_model_or_embedding_raises(self):
        with pytest.raises(ValueError, match="model_path or embedding_path"):
            _config(model_path=None, embedding_path=None).validate()

    def test_bm25_does_not_require_model(self):
        _config(
            retrieval_method="bm25", model_path=None, embedding_path=None
        ).validate()

    def test_embedding_path_satisfies_dense_requirement(self):
        _config(model_path=None, embedding_path="/fake/emb.memmap").validate()

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            _config(batch_size=0).validate()

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            _config(batch_size=-1).validate()

    def test_zero_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            _config(max_length=0).validate()

    def test_zero_keyword_limit_raises(self):
        with pytest.raises(ValueError, match="keyword_limit"):
            _config(keyword_limit=0).validate()

    def test_zero_vocab_max_length_raises(self):
        with pytest.raises(ValueError, match="vocab_max_length"):
            _config(vocab_max_length=0).validate()

    def test_zero_hnsw_ef_construction_raises(self):
        with pytest.raises(ValueError, match="hnsw_ef_construction"):
            _config(faiss_type="HNSW64", hnsw_ef_construction=0).validate()

    def test_config_is_frozen(self):
        config = _config()
        with pytest.raises(
            Exception
        ):  # dataclass(frozen=True) raises FrozenInstanceError
            config.batch_size = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# prepare_texts
# ---------------------------------------------------------------------------


class TestPrepareTexts:
    def test_e5_adds_passage_prefix_for_corpus(self):
        result = prepare_texts(["hello world"], "e5", is_query=False)
        assert result == ["passage: hello world"]

    def test_e5_adds_query_prefix_for_queries(self):
        result = prepare_texts(["what is python"], "e5", is_query=True)
        assert result == ["query: what is python"]

    def test_e5_large_also_gets_prefix(self):
        result = prepare_texts(["text"], "e5-large-v2", is_query=False)
        assert result[0].startswith("passage: ")

    def test_bge_query_gets_instruction_prefix(self):
        result = prepare_texts(["find relevant"], "bge", is_query=True)
        assert result[0].startswith("Represent this sentence")
        assert "find relevant" in result[0]

    def test_bge_passage_gets_no_prefix(self):
        result = prepare_texts(["document text"], "bge", is_query=False)
        assert result == ["document text"]

    def test_contriever_no_prefix_for_query(self):
        result = prepare_texts(["find me"], "contriever", is_query=True)
        assert result == ["find me"]

    def test_contriever_no_prefix_for_passage(self):
        result = prepare_texts(["a doc"], "contriever", is_query=False)
        assert result == ["a doc"]

    def test_dpr_no_prefix(self):
        result = prepare_texts(["query"], "dpr", is_query=True)
        assert result == ["query"]

    def test_empty_texts_returns_empty(self):
        assert prepare_texts([], "e5", is_query=False) == []

    def test_multiple_texts_all_prefixed(self):
        texts = ["a", "b", "c"]
        result = prepare_texts(texts, "e5", is_query=False)
        assert len(result) == 3
        assert all(t.startswith("passage: ") for t in result)

    def test_case_insensitive_model_name(self):
        result = prepare_texts(["text"], "E5-LARGE", is_query=False)
        assert result[0].startswith("passage: ")

    def test_explicit_prefix_overrides_model_default(self):
        result = prepare_texts(
            ["text"],
            "e5",
            is_query=False,
            passage_prefix="Represent document:",
        )
        assert result == ["Represent document: text"]

    def test_explicit_prefix_is_not_duplicated(self):
        result = prepare_texts(
            ["query: already prefixed"],
            "e5",
            is_query=True,
            query_prefix="query:",
        )
        assert result == ["query: already prefixed"]


# ---------------------------------------------------------------------------
# resolve_pooling_method
# ---------------------------------------------------------------------------


class TestResolvePoolingMethod:
    def test_e5_defaults_to_mean(self):
        assert resolve_pooling_method("e5-large", None) == "mean"

    def test_bge_defaults_to_cls(self):
        assert resolve_pooling_method("bge-base-en", None) == "cls"

    def test_contriever_defaults_to_mean(self):
        assert resolve_pooling_method("contriever", None) == "mean"

    def test_jina_defaults_to_mean(self):
        assert resolve_pooling_method("jina-embeddings-v2", None) == "mean"

    def test_unknown_model_defaults_to_mean(self):
        assert resolve_pooling_method("bert-base-uncased", None) == "mean"

    def test_explicit_mean_overrides(self):
        assert resolve_pooling_method("bge-m3", "mean") == "mean"

    def test_explicit_cls_overrides(self):
        assert resolve_pooling_method("e5-large", "cls") == "cls"

    def test_explicit_pooler(self):
        assert resolve_pooling_method("any-model", "pooler") == "pooler"

    def test_invalid_explicit_method_raises(self):
        with pytest.raises(NotImplementedError):
            resolve_pooling_method("any-model", "invalid_method")


# ---------------------------------------------------------------------------
# pooling (requires torch — skipped if not installed)
# ---------------------------------------------------------------------------


class TestPooling:
    torch = pytest.importorskip(
        "torch", reason="torch not installed", exc_type=ImportError
    )

    def test_cls_pooling_returns_first_token(self):
        from src.internal.document_index.index_builder import pooling

        last_hidden = self.torch.arange(12, dtype=self.torch.float).reshape(2, 3, 2)
        result = pooling(None, last_hidden, pooling_method="cls")
        expected = last_hidden[:, 0]
        assert self.torch.allclose(result, expected)

    def test_pooler_pooling_returns_pooler_output(self):
        from src.internal.document_index.index_builder import pooling

        pooler_out = self.torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = pooling(pooler_out, None, pooling_method="pooler")
        assert self.torch.allclose(result, pooler_out)

    def test_mean_pooling_requires_attention_mask(self):
        from src.internal.document_index.index_builder import pooling

        last_hidden = self.torch.ones(2, 3, 4)
        with pytest.raises(ValueError, match="attention_mask"):
            pooling(None, last_hidden, attention_mask=None, pooling_method="mean")

    def test_mean_pooling_basic(self):
        from src.internal.document_index.index_builder import pooling

        # 1 sequence, 3 tokens, 2 dims; all tokens attended
        last_hidden = self.torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = self.torch.ones(1, 3, dtype=self.torch.long)
        result = pooling(None, last_hidden, attention_mask=mask, pooling_method="mean")
        expected = self.torch.tensor([[3.0, 4.0]])  # (1+3+5)/3, (2+4+6)/3
        assert self.torch.allclose(result, expected)

    def test_mean_pooling_with_padding(self):
        from src.internal.document_index.index_builder import pooling

        # 2nd token is padding (mask=0)
        last_hidden = self.torch.tensor([[[1.0, 0.0], [99.0, 99.0], [3.0, 0.0]]])
        mask = self.torch.tensor([[1, 0, 1]], dtype=self.torch.long)
        result = pooling(None, last_hidden, attention_mask=mask, pooling_method="mean")
        expected = self.torch.tensor([[2.0, 0.0]])  # (1+3)/2=2, (0+0)/2=0
        assert self.torch.allclose(result, expected)

    def test_invalid_pooling_method_raises(self):
        from src.internal.document_index.index_builder import pooling

        with pytest.raises(NotImplementedError):
            pooling(None, None, pooling_method="unknown")


class TestIndexBuilderInternals:
    def test_set_hnsw_ef_construction_updates_hnsw_index(self):
        class _FakeHnsw:
            efConstruction = 40

        class _FakeIndex:
            hnsw = _FakeHnsw()

        index = _FakeIndex()

        set_hnsw_ef_construction(index, 200)

        assert index.hnsw.efConstruction == 200

    def test_set_hnsw_ef_construction_rejects_flat_index(self):
        with pytest.raises(ValueError, match="HNSW"):
            set_hnsw_ef_construction(object(), 200)

    def test_set_hnsw_ef_search_updates_hnsw_index(self):
        class _FakeHnsw:
            efSearch = 16

        class _FakeIndex:
            hnsw = _FakeHnsw()

        index = _FakeIndex()
        set_hnsw_ef_search(index, 128)
        assert index.hnsw.efSearch == 128

    def test_set_hnsw_ef_search_rejects_flat_index(self):
        with pytest.raises(ValueError, match="HNSW"):
            set_hnsw_ef_search(object(), 128)

    def test_hnsw_ef_search_zero_fails_validation(self):
        with pytest.raises(ValueError, match="hnsw_ef_search"):
            _config(faiss_type="HNSW64", hnsw_ef_search=0).validate()

    def test_hnsw_ef_search_positive_passes_validation(self):
        _config(faiss_type="HNSW64", hnsw_ef_search=64).validate()

    def test_faiss_gpu_with_hnsw_raises_in_build_dense_index(self, monkeypatch):
        monkeypatch.setattr(
            "src.internal.document_index.index_builder._require_faiss", lambda: object()
        )
        monkeypatch.setattr(
            "src.internal.document_index.index_builder._require_torch", lambda: object()
        )
        builder = IndexBuilder.__new__(IndexBuilder)
        builder.faiss_type = "HNSW64"
        builder.faiss_gpu = True
        builder.gpu_num = 1
        builder.embedding_path = None
        builder.index_save_path = type("P", (), {"exists": lambda self: False})()

        with pytest.raises(ValueError, match="HNSW"):
            builder.build_dense_index()

    def test_build_vocabulary_metadata_uses_single_normalized_pass(self):
        builder = IndexBuilder.__new__(IndexBuilder)
        builder.corpus = _Corpus(
            [
                {"id": "d1", "title": "Alpha", "contents": "Alpha beta beta"},
                {"id": "d2", "title": "Gamma", "contents": "Gamma delta"},
            ]
        )
        builder.keyword_limit = 2
        builder.vocab_max_length = 4

        vocabulary, corpus_entries = builder._build_vocabulary_metadata()

        assert [entry["doc_id"] for entry in corpus_entries] == [0, 1]
        assert corpus_entries[0]["tokens"] == ["alpha", "alpha", "beta", "beta"]
        assert corpus_entries[0]["keywords"] == ["alpha", "beta"]
        assert corpus_entries[0]["token_count"] == 4
        assert vocabulary.token2cnt["alpha"] == 2
        assert vocabulary.token2cnt["beta"] == 2
        assert vocabulary.token2cnt["gamma"] == 2

    def test_encode_all_preallocates_and_fills_in_order(self, monkeypatch):
        builder = IndexBuilder.__new__(IndexBuilder)
        builder.batch_size = 2
        builder.gpu_num = 0
        builder.device = "cpu"
        builder.max_length = 32
        builder.pooling_method = "mean"
        builder.retrieval_method = "contriever"
        builder.corpus = _Corpus(
            [
                {"contents": "doc one"},
                {"contents": "doc two"},
                {"contents": "doc three"},
            ]
        )

        monkeypatch.setattr(
            "src.internal.document_index.index_builder._require_torch", lambda: object()
        )
        monkeypatch.setattr(
            "src.internal.document_index.index_builder._require_tqdm",
            lambda: lambda seq, **_: seq,
        )

        calls: list[list[str]] = []

        def fake_encode_batch(*args, **kwargs):
            texts = args[2]
            calls.append(list(texts))
            base = len(calls) * 10
            return pytest.importorskip("numpy").array(
                [[base + index, base + index + 0.5] for index in range(len(texts))],
                dtype="float32",
            )

        monkeypatch.setattr(
            "src.internal.document_index.index_builder._encode_batch", fake_encode_batch
        )

        embeddings = builder.encode_all(encoder=object(), tokenizer=object())

        assert calls == [["doc one", "doc two"], ["doc three"]]
        assert embeddings.tolist() == [
            [10.0, 10.5],
            [11.0, 11.5],
            [20.0, 20.5],
        ]

    def test_encode_all_to_memmap_streams_batches_to_disk(self, monkeypatch, tmp_path):
        np = pytest.importorskip("numpy")
        builder = IndexBuilder.__new__(IndexBuilder)
        builder.batch_size = 2
        builder.gpu_num = 0
        builder.device = "cpu"
        builder.max_length = 32
        builder.pooling_method = "mean"
        builder.retrieval_method = "contriever"
        builder.embedding_save_path = tmp_path / "embeddings.memmap"
        builder.corpus = _Corpus(
            [
                {"contents": "doc one"},
                {"contents": "doc two"},
                {"contents": "doc three"},
            ]
        )

        monkeypatch.setattr(
            "src.internal.document_index.index_builder._require_torch", lambda: object()
        )
        monkeypatch.setattr(
            "src.internal.document_index.index_builder._require_tqdm",
            lambda: lambda seq, **_: seq,
        )

        calls: list[list[str]] = []

        def fake_encode_batch(*args, **kwargs):
            texts = args[2]
            calls.append(list(texts))
            base = len(calls) * 10
            return np.array(
                [[base + index, base + index + 0.5] for index in range(len(texts))],
                dtype="float32",
            )

        monkeypatch.setattr(
            "src.internal.document_index.index_builder._encode_batch", fake_encode_batch
        )

        embeddings = builder.encode_all_to_memmap(encoder=object(), tokenizer=object())

        assert calls == [["doc one", "doc two"], ["doc three"]]
        assert isinstance(embeddings, np.memmap)
        assert embeddings.tolist() == [
            [10.0, 10.5],
            [11.0, 11.5],
            [20.0, 20.5],
        ]
        assert builder.embedding_save_path.stat().st_size == 3 * 2 * 4

    def test_load_embedding_infers_hidden_size_from_file(self, tmp_path):
        np = pytest.importorskip("numpy")
        embedding_path = tmp_path / "precomputed.memmap"
        expected = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        expected.tofile(embedding_path)

        builder = IndexBuilder.__new__(IndexBuilder)
        loaded = builder._load_embedding(str(embedding_path), corpus_size=2)

        assert isinstance(loaded, np.memmap)
        assert loaded.shape == (2, 3)
        assert loaded.tolist() == expected.tolist()

    def test_load_embedding_rejects_misaligned_file_size(self, tmp_path):
        embedding_path = tmp_path / "broken.memmap"
        embedding_path.write_bytes(b"12345")

        builder = IndexBuilder.__new__(IndexBuilder)

        with pytest.raises(ValueError, match="not divisible"):
            builder._load_embedding(str(embedding_path), corpus_size=2)
