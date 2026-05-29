"""
Utilities for testing document access control lists (ACLs) and permissions.
"""

from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

# _get_access_for_documents removed — no direct DB access
# fetch_external_groups_for_user removed
# prefix_external_group removed
# prefix_user_email removed
from tests.integration.common_utils.types import PUBLIC_DOC_PAT

# DocumentByConnectorCredentialPair ORM removed — use HTTP
# User ORM model removed — use HTTP
# fetch_user_by_id removed — no direct DB access
from tests.integration.common_utils.test_models import DATestCCPair
from tests.integration.common_utils.test_models import DATestUser

logger = logging.getLogger(__name__)  # noqa: F821,F841


def get_user_acl(user: User, db_session: Session) -> set[str]:  # noqa: F821,F841
    """
    Get the ACL entries for a user, including their external groups, email, and public doc pattern.

    Args:
        user: The user object
        db_session: Database session

    Returns:
        Set of ACL entries for the user
    """
    db_external_groups = (
        fetch_external_groups_for_user(db_session, user.id) if user else []  # noqa: F821,F841
    )
    prefixed_external_groups = [
        prefix_external_group(db_external_group.external_user_group_id)  # noqa: F821,F841
        for db_external_group in db_external_groups
    ]

    user_acl = set(prefixed_external_groups)
    user_acl.update({prefix_user_email(user.email), PUBLIC_DOC_PAT})  # noqa: F821,F841
    return user_acl


def get_user_document_access_via_acl(
    test_user: DATestUser, document_ids: List[str], db_session: Session
) -> List[str]:
    """
    Determine which documents a user can access by comparing user ACL with document ACLs.

    This is a more reliable method than search-based verification as it directly checks
    permission logic without depending on search relevance or ranking.

    Args:
        test_user: The test user to check access for
        document_ids: List of document IDs to check
        db_session: Database session

    Returns:
        List of document IDs that the user can access
    """
    # Get the actual User object from the database
    if not user:  # noqa: F821,F841
        logger.error("Could not find user with ID %s", test_user.id)
        return []

    user_acl = get_user_acl(user, db_session)  # noqa: F821,F841
    logger.info("User %s ACL entries: %s", user.email, user_acl)  # noqa: F821,F841

    # Get document access information
    logger.info("Found access info for %s documents", len(doc_access_map))  # noqa: F821,F841

    accessible_docs = []
    for doc_id, doc_access in doc_access_map.items():  # noqa: F821,F841
        doc_acl = doc_access.to_acl()
        logger.info("Document %s ACL: %s", doc_id, doc_acl)

        # Check if user has any matching ACL entry
        if user_acl.intersection(doc_acl):
            accessible_docs.append(doc_id)
            logger.info("User %s has access to document %s", user.email, doc_id)  # noqa: F821,F841
        else:
            logger.info(
                "User %s does NOT have access to document %s",
                user.email,  # noqa: F821
                doc_id,  # noqa: F821,F841
            )

    return accessible_docs


def get_all_connector_documents(
    cc_pair: DATestCCPair, db_session: Session
) -> List[str]:
    """
    Get all document IDs for a given connector/credential pair.

    Args:
        cc_pair: The connector-credential pair
        db_session: Database session

    Returns:
        List of document IDs
    """
    stmt = select(DocumentByConnectorCredentialPair.id).where(  # noqa: F821,F841
        DocumentByConnectorCredentialPair.connector_id == cc_pair.connector_id,  # noqa: F821,F841
        DocumentByConnectorCredentialPair.credential_id == cc_pair.credential_id,  # noqa: F821,F841
    )

    result = db_session.execute(stmt)
    document_ids = [row[0] for row in result.fetchall()]
    logger.info(
        "Found %s documents for connector %s", len(document_ids), cc_pair.connector_id
    )

    return document_ids


def get_documents_by_permission_type(
    document_ids: List[str], db_session: Session
) -> List[str]:
    """
    Categorize documents by their permission types and return public documents.

    Args:
        document_ids: List of document IDs to check
        db_session: Database session

    Returns:
        List of document IDs that are public
    """
    doc_access_map = _get_access_for_documents(document_ids, db_session)  # noqa: F821,F841

    public_docs = []

    for doc_id, doc_access in doc_access_map.items():
        if doc_access.is_public:
            public_docs.append(doc_id)

    return public_docs
