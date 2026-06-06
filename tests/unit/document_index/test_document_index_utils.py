import pytest
from uuid import UUID

from src.backend.document_index.document_index_utils import (
    translate_boost_count_to_multiplier,
    get_uuid_from_chunk_info,
    get_uuid_from_chunk_info_old,
    should_use_multipass,
    DEFAULT_BATCH_SIZE,
)


def test_boost_zero_is_one():
    assert translate_boost_count_to_multiplier(0) == pytest.approx(1.0, abs=0.01)


def test_boost_negative_below_one():
    result = translate_boost_count_to_multiplier(-10)
    assert 0.5 < result < 1.0


def test_boost_positive_above_one():
    result = translate_boost_count_to_multiplier(10)
    assert 1.0 < result <= 2.0


def test_get_uuid_from_chunk_info_returns_uuid():
    uid = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="tenant1")
    assert isinstance(uid, UUID)


def test_get_uuid_from_chunk_info_deterministic():
    uid1 = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="t1")
    uid2 = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="t1")
    assert uid1 == uid2


def test_get_uuid_from_chunk_info_trailing_slash():
    uid1 = get_uuid_from_chunk_info(document_id="doc1/", chunk_id=0, tenant_id="t1")
    uid2 = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="t1")
    assert uid1 == uid2


def test_get_uuid_from_chunk_info_old():
    uid = get_uuid_from_chunk_info_old(document_id="doc1", chunk_id=0)
    assert isinstance(uid, UUID)


def test_should_use_multipass_none_uses_default():
    result = should_use_multipass(None)
    assert isinstance(result, bool)


def test_default_batch_size():
    assert DEFAULT_BATCH_SIZE == 30
