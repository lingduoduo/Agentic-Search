"""Tests for runnable README examples."""

from __future__ import annotations

import pytest

from examples.run_grpo_training_pipeline import run_demo as run_grpo_demo
from examples.run_search_pipeline import assemble_sections
from examples.run_search_pipeline import run_demo as run_pipeline_demo


def test_grpo_training_pipeline_example_runs_without_model_backends():
    pytest.importorskip("torch")
    result = run_grpo_demo()

    assert [round(value, 2) for value in result["rewards"]] == [0.96, 0.64, -0.08]
    assert len(result["advantages"]) == 3
    assert result["reward_components"][0]["correctness"] == 1.0
    assert "policy_loss" in result


ADMIN_GROUP = "search-admins"
OWNER_EMAIL = "owner@example.test"


def _ids(sections):
    """Document ids reaching answer context, in rank order."""
    return [section.center.metadata.get("document_id") for section in sections]


def test_anonymous_caller_reads_only_unrestricted_documents():
    sections = run_pipeline_demo(query="rerank deployment")

    assert "restricted-runbook" not in _ids(sections)
    assert "public-guide" in _ids(sections)


def test_group_membership_grants_the_restricted_document():
    anonymous = run_pipeline_demo(query="rerank deployment")
    admin = run_pipeline_demo(query="rerank deployment", group_ids=[ADMIN_GROUP])

    assert "restricted-runbook" not in _ids(anonymous)
    assert "restricted-runbook" in _ids(admin)


def test_email_grant_is_not_a_group_grant():
    owner = run_pipeline_demo(query="quarterly retrieval budget", email=OWNER_EMAIL)
    admin = run_pipeline_demo(
        query="quarterly retrieval budget", group_ids=[ADMIN_GROUP]
    )

    assert "owner-memo" in _ids(owner)
    assert "owner-memo" not in _ids(admin)


def test_skipping_enforcement_leaks_a_document_the_caller_may_not_read():
    """The reason the example exists.

    Sending filters to a retrieval backend is not enforcement: the backend is
    free to ignore them.  Dropping the caller-side check hands a restricted
    document to an anonymous reader.
    """
    enforced = run_pipeline_demo(query="rerank deployment")
    unenforced = run_pipeline_demo(query="rerank deployment", enforce=False)

    assert "restricted-runbook" not in _ids(enforced)
    assert "restricted-runbook" in _ids(unenforced)


def test_consecutive_chunks_of_one_source_merge_into_one_section():
    from src.context.models import ContextDocument

    documents = [
        ContextDocument(
            id="D1",
            title="Guide",
            content="first half",
            metadata={"document_id": "guide", "chunk_id": 0},
        ),
        ContextDocument(
            id="D2",
            title="Guide",
            content="second half",
            metadata={"document_id": "guide", "chunk_id": 1},
        ),
    ]

    sections = assemble_sections(documents)

    assert len(sections) == 1
    assert "first half" in sections[0].combined_content
    assert "second half" in sections[0].combined_content


def test_demo_actually_exercises_the_merge_step():
    """Step 4 must fire in the demo, not just be available.

    Merging needs consecutive chunks to come back consecutively and in order,
    so a corpus that ranks chunk 1 above chunk 0 would leave the step inert
    while every other assertion still passed.
    """
    sections = run_pipeline_demo(query="rerank deployment")
    merged = [section for section in sections if len(section.documents) > 1]

    assert len(merged) == 1
    assert merged[0].center.metadata.get("document_id") == "public-guide"
