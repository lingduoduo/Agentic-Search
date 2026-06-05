"""Tests for runnable README examples."""

from __future__ import annotations

from examples.run_grpo_training_pipeline import run_demo as run_grpo_demo
from examples.run_search_pipeline import AccessPolicy
from examples.run_search_pipeline import InMemorySearchIndex
from examples.run_search_pipeline import SearchDocument
from examples.run_search_pipeline import SearchFilters
from examples.run_search_pipeline import SearchRequest
from examples.run_search_pipeline import SearchUser
from examples.run_search_pipeline import run_demo as run_pipeline_demo
from examples.run_search_pipeline import search_pipeline


def test_grpo_training_pipeline_example_runs_without_model_backends():
    result = run_grpo_demo()

    assert [round(value, 2) for value in result["rewards"]] == [0.96, 0.64, -0.08]
    assert len(result["advantages"]) == 3
    assert result["reward_components"][0]["correctness"] == 1.0
    assert "policy_loss" in result


def test_search_pipeline_example_applies_filters_and_permissions():
    sections = run_pipeline_demo()

    assert [section.document_id for section in sections] == ["public-guide"]
    assert sections[0].title == "Dense Retrieval Guide"

    index = InMemorySearchIndex(
        [
            SearchDocument(
                id="public",
                title="Public Doc",
                contents="rerank deployment notes",
                document_set="docs",
            ),
            SearchDocument(
                id="private",
                title="Private Doc",
                contents="rerank deployment secret",
                document_set="ops",
                access=AccessPolicy(
                    public=False,
                    allowed_group_ids=frozenset({"search-admins"}),
                ),
            ),
        ]
    )
    reader = SearchUser(id="reader", email="reader@example.test")
    admin = SearchUser(id="admin", group_ids=frozenset({"search-admins"}))

    reader_sections = search_pipeline(
        request=SearchRequest(query="rerank deployment", limit=10),
        index=index,
        user=reader,
    )
    admin_sections = search_pipeline(
        request=SearchRequest(query="rerank deployment", limit=10),
        index=index,
        user=admin,
    )
    filtered_sections = search_pipeline(
        request=SearchRequest(
            query="rerank deployment",
            filters=SearchFilters(document_set=frozenset({"ops"})),
            limit=10,
            bypass_acl=True,
        ),
        index=index,
        user=reader,
    )

    assert [section.document_id for section in reader_sections] == ["public"]
    assert {section.document_id for section in admin_sections} == {"public", "private"}
    assert [section.document_id for section in filtered_sections] == ["private"]


def test_search_pipeline_permission_filter_entrypoint_can_be_replaced():
    index = InMemorySearchIndex(
        [
            SearchDocument(
                id="blocked",
                title="Blocked",
                contents="dense retrieval",
                access=AccessPolicy(public=False),
            )
        ]
    )

    def allow_all(chunks, user, bypass_acl):
        del user, bypass_acl
        return list(chunks)

    sections = search_pipeline(
        request=SearchRequest(query="dense retrieval"),
        index=index,
        user=SearchUser(id="reader"),
        permission_filter=allow_all,
    )

    assert [section.document_id for section in sections] == ["blocked"]
