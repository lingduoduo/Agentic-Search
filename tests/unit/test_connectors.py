"""Unit tests for connector interfaces and built-in connectors."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from src.backend.connectors import (
    BaseConnector,
    CheckpointedConnectorWithPermSync,
    ConnectorCheckpoint,
    ConnectorFailure,
    Document,
    HierarchyNode,
    InMemoryConnector,
    LocalFileConnector,
    LoadConnector,
    OAuthConnector,
    PollConnector,
    SearchConnector,
    SlimConnector,
    SlimConnectorWithPermSync,
    SlimDocument,
    StaticCredentialsProvider,
    batched,
)
from src.retrieval.index_builder import (
    dump_connector_to_jsonl,
    load_corpus_from_connector,
)
from src.tools.search import SearchPage


def _collect_checkpoint_output(generator):
    items = []
    while True:
        try:
            items.append(next(generator))
        except StopIteration as exc:
            return items, exc.value


def test_base_connector_metadata_and_credentials_helpers():
    assert BaseConnector.parse_metadata(
        {"author": "Ada", "tags": ["ml", "search"]}
    ) == [
        "author: Ada",
        "tags: ml, search",
    ]
    with pytest.raises(RuntimeError):
        BaseConnector.parse_metadata({"bad": {"nested": True}})

    provider = StaticCredentialsProvider({"token": "secret"}, provider_key="demo")
    with provider as active:
        assert active.get_provider_key() == "demo"
        assert active.get_credentials() == {"token": "secret"}
        active.set_credentials({"token": "rotated"})
    assert provider.get_credentials() == {"token": "rotated"}
    assert provider.is_dynamic() is False


def test_batched_validates_size_and_emits_non_empty_batches():
    docs = [Document(id=str(i), contents="body") for i in range(3)]

    assert list(batched(docs, 2)) == [[docs[0], docs[1]], [docs[2]]]
    with pytest.raises(ValueError):
        list(batched(docs, 0))


def test_connector_task_interfaces_can_be_implemented():
    class FullLoad(LoadConnector):
        def load_from_state(self):
            yield [Document(id="full", contents="body")]

    class Incremental(PollConnector):
        def poll_source(self, start, end):
            yield [
                Document(
                    id="poll",
                    contents="body",
                    metadata={"start": start, "end": end},
                )
            ]

    class Slim(SlimConnector):
        def retrieve_all_slim_docs(self, start=None, end=None):
            yield [SlimDocument(id="doc-id", metadata={"start": start, "end": end})]

    class SlimWithPerms(SlimConnectorWithPermSync):
        def retrieve_all_slim_docs_perm_sync(self, start=None, end=None):
            yield [
                SlimDocument(
                    id="doc-id",
                    metadata={"start": start, "end": end},
                    permissions={"users": ["ada@example.test"]},
                )
            ]

    class OAuthDemo(OAuthConnector):
        @classmethod
        def oauth_id(cls):
            return "demo"

        @classmethod
        def oauth_authorization_url(cls, base_domain, state, additional_kwargs):
            scope = additional_kwargs.get("scope", "read")
            return f"{base_domain}/oauth?state={state}&scope={scope}"

        @classmethod
        def oauth_code_to_token(cls, base_domain, code, additional_kwargs):
            del base_domain, additional_kwargs
            return {"access_token": code}

    assert list(FullLoad().load_from_state())[0][0].id == "full"
    assert list(Incremental().poll_source(1.0, 2.0))[0][0].metadata == {
        "start": 1.0,
        "end": 2.0,
    }
    slim = list(Slim().retrieve_all_slim_docs())[0][0]
    assert slim.id == "doc-id"
    assert slim.permissions == {}
    slim_perm = list(SlimWithPerms().retrieve_all_slim_docs_perm_sync())[0][0]
    assert slim_perm.permissions == {"users": ["ada@example.test"]}
    assert OAuthDemo.oauth_id() == "demo"
    assert (
        OAuthDemo.oauth_authorization_url(
            "https://app.test", "state-1", {"scope": "email"}
        )
        == "https://app.test/oauth?state=state-1&scope=email"
    )
    assert OAuthDemo.oauth_code_to_token("https://app.test", "token", {}) == {
        "access_token": "token"
    }


def test_checkpointed_connector_with_perm_sync_supports_mock_pattern():
    @dataclass(frozen=True)
    class MockCheckpoint(ConnectorCheckpoint):
        last_document_id: str | None = None

    class MockCheckpointConnector(CheckpointedConnectorWithPermSync[MockCheckpoint]):
        def __init__(self):
            self.saved_checkpoints = []
            self.yields = [
                {
                    "documents": [Document(id="a", contents="alpha")],
                    "failures": [ConnectorFailure("b", "boom")],
                    "checkpoint": MockCheckpoint(
                        has_more=True,
                        cursor="next",
                        last_document_id="a",
                    ),
                },
                {
                    "documents": [Document(id="c", contents="charlie")],
                    "failures": [],
                    "checkpoint": MockCheckpoint(
                        has_more=False,
                        cursor=None,
                        last_document_id="c",
                    ),
                },
                {
                    "documents": [],
                    "failures": [],
                    "checkpoint": MockCheckpoint(has_more=False),
                    "unhandled_exception": "kaboom",
                },
            ]
            self.current_yield_index = 0

        def load_credentials(self, credentials):
            self.loaded_credentials = credentials
            return None

        def _load_from_checkpoint_common(
            self,
            start,
            end,
            checkpoint,
            *,
            include_permissions=False,
        ):
            del start, end
            self.saved_checkpoints.append(checkpoint)
            current_yield = self.yields[self.current_yield_index]
            self.current_yield_index += 1
            if current_yield.get("unhandled_exception"):
                raise RuntimeError(current_yield["unhandled_exception"])
            for document in current_yield["documents"]:
                if include_permissions and not document.permissions:
                    document = Document(
                        id=document.id,
                        contents=document.contents,
                        title=document.title,
                        url=document.url,
                        metadata=document.metadata,
                        permissions={
                            "users": ["test@example.com", "admin@example.com"],
                            "groups": ["mock-group-1", "mock-group-2"],
                            "public": False,
                        },
                    )
                yield document
            yield from current_yield["failures"]
            return current_yield["checkpoint"]

        def load_from_checkpoint(self, start, end, checkpoint):
            return self._load_from_checkpoint_common(start, end, checkpoint)

        def load_from_checkpoint_with_perm_sync(self, start, end, checkpoint):
            return self._load_from_checkpoint_common(
                start,
                end,
                checkpoint,
                include_permissions=True,
            )

        def build_dummy_checkpoint(self):
            return MockCheckpoint(has_more=True, last_document_id=None)

        def validate_checkpoint_json(self, checkpoint_json):
            return MockCheckpoint.from_json(checkpoint_json)

    connector = MockCheckpointConnector()
    connector.load_credentials({"token": "secret"})

    checkpoint = connector.build_dummy_checkpoint()
    items, next_checkpoint = _collect_checkpoint_output(
        connector.load_from_checkpoint(1.0, 2.0, checkpoint)
    )
    assert connector.loaded_credentials == {"token": "secret"}
    assert connector.saved_checkpoints == [checkpoint]
    assert items == [
        Document(id="a", contents="alpha"),
        ConnectorFailure("b", "boom"),
    ]
    assert next_checkpoint == MockCheckpoint(
        has_more=True,
        cursor="next",
        last_document_id="a",
    )

    perm_items, final_checkpoint = _collect_checkpoint_output(
        connector.load_from_checkpoint_with_perm_sync(2.0, 3.0, next_checkpoint)
    )
    assert perm_items[0].permissions == {
        "users": ["test@example.com", "admin@example.com"],
        "groups": ["mock-group-1", "mock-group-2"],
        "public": False,
    }
    assert final_checkpoint.last_document_id == "c"
    assert MockCheckpoint.from_json(final_checkpoint.to_json()) == final_checkpoint
    assert connector.validate_checkpoint_json(
        '{"has_more": true, "cursor": "x", "last_document_id": "z"}'
    ) == MockCheckpoint(has_more=True, cursor="x", last_document_id="z")

    with pytest.raises(RuntimeError, match="kaboom"):
        list(connector.load_from_checkpoint(3.0, 4.0, final_checkpoint))


def test_in_memory_connector_loads_dicts_and_documents():
    connector = InMemoryConnector(
        [
            {"id": "one", "contents": "First", "title": "One"},
            Document(id="two", contents="Second"),
        ],
        batch_size=1,
    )

    batches = list(connector.load_from_state())

    assert [[doc.id for doc in batch] for batch in batches] == [["one"], ["two"]]
    assert batches[0][0].title == "One"


def test_local_file_connector_loads_files(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.md"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    connector = LocalFileConnector([tmp_path], batch_size=10)
    docs = [doc for batch in connector.load_from_state() for doc in batch]

    assert sorted(doc.title for doc in docs) == ["a.txt", "b.md"]
    assert {doc.contents for doc in docs} == {"alpha", "beta"}
    assert all(doc.metadata["source"] == "local_file" for doc in docs)


def test_local_file_connector_uses_metadata_file_and_credentials(tmp_path):
    source = tmp_path / "raw.txt"
    metadata_path = tmp_path / "metadata.json"
    source.write_text("alpha", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "filename": "Display.txt",
                    "document_id": "doc-1",
                    "file_display_name": "Friendly",
                    "title": "Friendly title",
                    "link": "https://docs.test/friendly",
                    "doc_updated_at": "2026-01-02T03:04:05+00:00",
                    "primary_owners": ["owner@example.test"],
                    "topic": "search",
                }
            ]
        ),
        encoding="utf-8",
    )

    connector = LocalFileConnector(
        [source],
        file_names=["Display.txt"],
        metadata_path=metadata_path,
    )
    assert connector.load_credentials({"pdf_password": "secret"}) is None
    docs = [doc for batch in connector.load_from_state() for doc in batch]

    assert connector.pdf_pass == "secret"
    assert docs[0].id == "doc-1"
    assert docs[0].title == "Friendly title"
    assert docs[0].url == "https://docs.test/friendly"
    assert docs[0].metadata["display_name"] == "Friendly"
    assert docs[0].metadata["doc_updated_at"] == "2026-01-02T03:04:05+00:00"
    assert docs[0].metadata["primary_owners"] == ["owner@example.test"]
    assert docs[0].metadata["topic"] == "search"


def test_local_file_connector_skips_unsupported_and_empty_files(tmp_path):
    supported = tmp_path / "keep.txt"
    empty = tmp_path / "empty.txt"
    unsupported = tmp_path / "skip.bin"
    supported.write_text("keep", encoding="utf-8")
    empty.write_text("   ", encoding="utf-8")
    unsupported.write_bytes(b"binary")

    connector = LocalFileConnector([tmp_path], batch_size=10)
    docs = [doc for batch in connector.load_from_state() for doc in batch]

    assert [doc.title for doc in docs] == ["keep.txt"]


def test_search_connector_loads_search_results(monkeypatch):
    async def _fake_search_tool(query, **kwargs):
        assert query == "agentic search"
        assert kwargs["provider"] == "google"
        assert kwargs["page_size"] == 2
        return [
            SearchPage(title="A", summary="Alpha", url="https://a.test"),
            SearchPage(error="ignored"),
        ]

    monkeypatch.setattr("src.backend.connectors.basic.search_tool", _fake_search_tool)

    connector = SearchConnector(
        ["agentic search"],
        provider="google",
        page_size=2,
        batch_size=10,
    )
    docs = [doc for batch in connector.load_from_state() for doc in batch]

    assert docs == [
        Document(
            id="https://a.test",
            title="A",
            url="https://a.test",
            contents="Alpha",
            metadata={"source": "google", "query": "agentic search"},
        )
    ]


def test_search_connector_async_variant_inside_event_loop(monkeypatch):
    async def _fake_search_tool(query, **kwargs):
        del query, kwargs
        return [SearchPage(title="A", summary="Alpha", url="https://a.test")]

    monkeypatch.setattr("src.backend.connectors.basic.search_tool", _fake_search_tool)

    async def _run():
        connector = SearchConnector(["q"])
        with pytest.raises(RuntimeError, match="active event loop"):
            list(connector.load_from_state())
        return await connector.aload_from_state()

    docs = asyncio.run(_run())

    assert docs[0].id == "https://a.test"


def test_load_corpus_from_connector_skips_hierarchy_nodes():
    class MixedConnector(LoadConnector):
        def load_from_state(self):
            yield [
                Document(id="d1", contents="alpha", title="A"),
                HierarchyNode(id="h1", name="folder"),
                Document(id="d2", contents="beta", title="B"),
            ]

    corpus = load_corpus_from_connector(MixedConnector())

    assert len(corpus) == 2
    assert corpus[0]["id"] == "d1"
    assert corpus[0]["contents"] == "alpha"
    assert corpus[1]["title"] == "B"
    assert corpus["id"] == ["d1", "d2"]


def test_dump_connector_to_jsonl_writes_documents(tmp_path):
    class SimpleConnector(LoadConnector):
        def load_from_state(self):
            yield [
                Document(id="x", contents="hello", title="X"),
                HierarchyNode(id="h1", name="folder"),
                Document(id="y", contents="world", title=None),
            ]

    out = tmp_path / "corpus.jsonl"
    dump_connector_to_jsonl(SimpleConnector(), out)

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0] == {"id": "x", "title": "X", "contents": "hello"}
    assert lines[1] == {"id": "y", "title": None, "contents": "world"}
