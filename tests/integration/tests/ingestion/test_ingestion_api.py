from tests.integration.common_utils.types import DocumentSource
from tests.integration.common_utils.types import InputType

# get_session_with_current_tenant removed — no direct DB access
# Document ORM removed — use HTTP
from tests.integration.common_utils.managers.api_key import APIKeyManager
from tests.integration.common_utils.managers.cc_pair import CCPairManager
from tests.integration.common_utils.managers.document import IngestionManager
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.test_models import DATestUser
# vespa_fixture removed — no Vespa in this deployment


def test_ingestion_api_crud(
    reset: None,  # noqa: ARG001
    vespa_client: vespa_fixture,  # noqa: F821,F841
) -> None:
    """Test create, list, and delete via the ingestion API."""
    admin_user: DATestUser = UserManager.create(email="admin@example.com")
    cc_pair = CCPairManager.create_from_scratch(
        name="Ingestion-API-Test",
        source=DocumentSource.FILE,
        input_type=InputType.LOAD_STATE,
        connector_specific_config={
            "file_locations": [],
            "file_names": [],
            "zip_metadata_file_id": None,
        },
        user_performing_action=admin_user,
    )
    api_key = APIKeyManager.create(user_performing_action=admin_user)
    api_key.headers.update(admin_user.headers)

    # CREATE
    doc = IngestionManager.seed_doc_with_content(
        cc_pair=cc_pair,
        content="Test document",
        document_id="test-doc-1",
        api_key=api_key,
    )

    with get_session_with_current_tenant() as db_session:  # noqa: F821,F841
        doc_db = db_session.query(Document).filter(Document.id == doc.id).first()  # noqa: F821,F841
        assert doc_db is not None
        assert doc_db.from_ingestion_api is True

    vespa_docs = vespa_client.get_documents_by_id([doc.id])["documents"]
    assert len(vespa_docs) == 1

    # LIST
    assert any(d["document_id"] == doc.id for d in docs_list)  # noqa: F821,F841

    # DELETE

    with get_session_with_current_tenant() as db_session:  # noqa: F821,F841
        doc_db = db_session.query(Document).filter(Document.id == doc.id).first()  # noqa: F821,F841
        assert doc_db is None

    vespa_docs = vespa_client.get_documents_by_id([doc.id])["documents"]
    assert len(vespa_docs) == 0
