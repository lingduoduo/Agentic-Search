from datetime import datetime

from src.internal.document_index.models import (
    QueryType,
    Embedding,
    InferenceChunk,
    InferenceChunkUncleaned,
    IndexFilters,
    MultipassConfig,
    ExternalAccess,
)


def test_query_type_values():
    assert QueryType.SEMANTIC == "semantic"
    assert QueryType.KEYWORD == "keyword"
    assert QueryType.HYBRID == "hybrid"


def test_embedding_is_list_of_float():
    emb: Embedding = [0.1, 0.2, 0.3]
    assert isinstance(emb, list)


def test_inference_chunk_defaults():
    chunk = InferenceChunk(document_id="doc1", chunk_ind=0)
    assert chunk.blurb == ""
    assert chunk.content == ""
    assert chunk.score is None
    assert chunk.match_highlights == []
    assert chunk.document_sets == set()


def test_inference_chunk_full():
    chunk = InferenceChunk(
        document_id="doc1",
        chunk_ind=2,
        blurb="blurb text",
        content="full content",
        score=0.95,
        source_type="web",
        match_highlights=["highlight"],
    )
    assert chunk.content == "full content"
    assert chunk.score == 0.95


def test_inference_chunk_uncleaned_to_inference_chunk():
    raw = InferenceChunkUncleaned(
        document_id="doc1",
        chunk_ind=0,
        content="Title\n\nReal content",
        title="My Title",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    cleaned = raw.to_inference_chunk()
    assert isinstance(cleaned, InferenceChunk)
    assert cleaned.document_id == "doc1"


def test_index_filters_defaults():
    f = IndexFilters()
    assert f.access_control_list is None
    assert f.document_set is None
    assert f.source_type is None
    assert f.time_cutoff is None


def test_index_filters_with_values():
    f = IndexFilters(
        access_control_list=["user1", "PUBLIC"],
        document_set=["set1"],
        source_type=["web"],
        time_cutoff=datetime(2024, 1, 1),
    )
    assert "PUBLIC" in f.access_control_list
    assert f.document_set == ["set1"]


def test_multipass_config_defaults():
    cfg = MultipassConfig()
    assert cfg.multipass_indexing is False
    assert cfg.enable_large_chunks is False


def test_external_access_defaults():
    ea = ExternalAccess()
    assert ea.external_user_emails == []
    assert ea.external_user_group_ids == []
    assert ea.is_public is False


def test_constants_importable():
    from src.internal.configs.constants import (
        PUBLIC_DOC_PAT,
        RETURN_SEPARATOR,
        INDEX_SEPARATOR,
        SOURCE_TYPE,
    )

    assert PUBLIC_DOC_PAT == "PUBLIC"
    assert isinstance(RETURN_SEPARATOR, str)
    assert isinstance(INDEX_SEPARATOR, str)
    assert SOURCE_TYPE == "source_type"


def test_vector_db_settings_defaults():
    from src.internal.configs.app_configs import VectorDbSettings

    s = VectorDbSettings()
    assert s.disable_vector_db is False
    assert s.enable_opensearch is False
    assert "localhost" in s.opensearch_host


def test_vector_db_settings_custom():
    from src.internal.configs.app_configs import VectorDbSettings

    s = VectorDbSettings(disable_vector_db=True)
    assert s.disable_vector_db is True


def test_utils_importable():
    from src.internal.document_index.utils import (
        setup_logger,
        batch_generator,
        remove_invalid_unicode_chars,
        convert_metadata_list_of_strings_to_dict,
        get_experts_stores_representations,
        split_relationship_id,
    )

    assert callable(setup_logger)
    assert callable(batch_generator)
    assert callable(remove_invalid_unicode_chars)
    assert callable(convert_metadata_list_of_strings_to_dict)
    assert callable(get_experts_stores_representations)
    assert callable(split_relationship_id)


def test_batch_generator():
    from src.internal.document_index.utils import batch_generator

    items = list(range(10))
    batches = list(batch_generator(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_batch_generator_exact():
    from src.internal.document_index.utils import batch_generator

    batches = list(batch_generator([1, 2, 3], 3))
    assert batches == [[1, 2, 3]]


def test_remove_invalid_unicode_chars():
    from src.internal.document_index.utils import remove_invalid_unicode_chars

    assert remove_invalid_unicode_chars("hello\x00world") == "helloworld"
    assert remove_invalid_unicode_chars("normal text") == "normal text"


def test_convert_metadata_list_of_strings_to_dict():
    from src.internal.document_index.utils import (
        convert_metadata_list_of_strings_to_dict,
    )

    result = convert_metadata_list_of_strings_to_dict(["key1:val1", "key2:val2"])
    assert result == {"key1": "val1", "key2": "val2"}


def test_convert_metadata_dict_passthrough():
    from src.internal.document_index.utils import (
        convert_metadata_list_of_strings_to_dict,
    )

    result = convert_metadata_list_of_strings_to_dict({"key": "val"})
    assert result == {"key": "val"}


def test_split_relationship_id():
    from src.internal.document_index.utils import split_relationship_id

    source, rel, target = split_relationship_id("doc1:RELATED:doc2")
    assert source == "doc1"
    assert rel == "RELATED"
    assert target == "doc2"


def test_redis_shared_lock_no_raise():
    from src.internal.document_index.utils import redis_shared_lock

    with redis_shared_lock("test"):
        pass  # must not raise


def test_get_shared_kv_store_interface():
    from src.internal.document_index.utils import get_shared_kv_store

    kv = get_shared_kv_store()
    assert kv.get("k") is None
    kv.set("k", "v")
    kv.delete("k")


def test_log_function_time_logs_timing(caplog):
    import logging
    from src.internal.document_index.utils import log_function_time

    @log_function_time(debug_only=True)
    def slow_fn():
        return 42

    with caplog.at_level(logging.DEBUG):
        result = slow_fn()

    assert result == 42
    assert any(
        "slow_fn" in r.message and "took" in r.message and r.levelno == logging.DEBUG
        for r in caplog.records
    )


def test_log_function_time_logs_info_when_not_debug_only(caplog):
    import logging
    from src.internal.document_index.utils import log_function_time

    @log_function_time(debug_only=False)
    def info_fn():
        return 99

    with caplog.at_level(logging.INFO):
        result = info_fn()

    assert result == 99
    assert any(
        "info_fn" in r.message and "took" in r.message and r.levelno == logging.INFO
        for r in caplog.records
    )
