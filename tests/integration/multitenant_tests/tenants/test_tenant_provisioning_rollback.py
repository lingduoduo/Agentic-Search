import pytest

pytestmark = pytest.mark.skip(
    reason="requires external services not available in this deployment"
)

"""
Integration tests for tenant provisioning rollback behavior.

Tests the fix for the drop_schema bug where:
1. isidentifier() rejected valid UUID tenant IDs (with hyphens)
2. SQL syntax was broken (%(schema_name)s instead of proper identifier handling)

This test verifies the full flow: provisioning failure → rollback → schema cleanup.
"""

import uuid  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402
from unittest.mock import patch  # noqa: E402

from sqlalchemy import text  # noqa: E402

# create_schema_if_not_exists removed
# drop_schema removed
# get_session_with_shared_schema removed
from shared_configs.configs import TENANT_ID_PREFIX  # noqa: E402


def _schema_exists(schema_name: str) -> bool:
    """Check if a schema exists in the database."""
    with get_session_with_shared_schema() as session:  # noqa: F821,F841
        result = session.execute(
            text(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = :schema"
            ),
            {"schema": schema_name},
        ).fetchone()
        return result is not None


class TestTenantProvisioningRollback:
    """Integration tests for provisioning failure and rollback."""

    def test_failed_provisioning_cleans_up_schema(self) -> None:
        """
        When setup_tenant fails after schema creation, rollback should
        clean up the orphaned schema.

        This is the actual bug scenario: pre_provision_tenant creates a schema,
        setup_tenant fails, rollback is called, but drop_schema was broken
        (isidentifier rejected UUIDs with hyphens), leaving orphaned schemas.
        """
        pass  # tenant_provisioning tasks removed

        # Track which tenant_id gets created
        created_tenant_id = None

        def track_schema_creation(tenant_id: str) -> bool:
            nonlocal created_tenant_id
            created_tenant_id = tenant_id
            return create_schema_if_not_exists(tenant_id)  # noqa: F821,F841

        # Mock setup_tenant to fail after schema creation.
        # Also mock the Redis lock so the test doesn't compete with a live
        # monitoring worker that may already hold the provision lock.
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True

        with patch(
            "ee.onyx.background.celery.tasks.tenant_provisioning.tasks.get_redis_client"
        ) as mock_redis:
            mock_redis.return_value.lock.return_value = mock_lock

            with patch(
                "ee.onyx.background.celery.tasks.tenant_provisioning.tasks.setup_tenant"
            ) as mock_setup:
                mock_setup.side_effect = Exception("Simulated provisioning failure")

                with patch(
                    "ee.onyx.background.celery.tasks.tenant_provisioning.tasks.create_schema_if_not_exists",
                    side_effect=track_schema_creation,
                ):
                    pass  # pre-provisioning — removed

        # Verify that the schema was created and then cleaned up
        assert created_tenant_id is not None, "Schema should have been created"
        assert created_tenant_id.startswith(TENANT_ID_PREFIX), (
            f"Should have tenant prefix: {created_tenant_id}"
        )
        assert not _schema_exists(created_tenant_id), (
            f"Schema {created_tenant_id} should have been rolled back"
        )

    def test_drop_schema_works_with_uuid_tenant_id(self) -> None:
        """
        drop_schema should work with UUID-format tenant IDs.

        This directly tests the fix: UUID tenant IDs contain hyphens,
        which isidentifier() rejected. The new regex validation accepts them.
        """
        tenant_id = f"{TENANT_ID_PREFIX}{uuid.uuid4()}"

        # Create schema
        assert _schema_exists(tenant_id), "Schema should exist after creation"

        # Drop schema
        assert not _schema_exists(tenant_id), "Schema should be dropped"
