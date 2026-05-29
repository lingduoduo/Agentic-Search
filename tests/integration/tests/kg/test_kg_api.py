import pytest

pytestmark = pytest.mark.skip(
    reason="requires external services not available in this deployment"
)

import json  # noqa: E402
from datetime import datetime  # noqa: E402
from http import HTTPStatus  # noqa: E402

import pytest  # noqa: E402
import requests  # noqa: E402

from tests.integration.common_utils.types import DocumentSource  # noqa: E402
from tests.integration.common_utils.types import InputType  # noqa: E402

# create_connector removed
# get_session_with_current_tenant removed — no direct DB access
# get_kg_config_settings removed
# set_kg_config_settings removed
# Connector ORM removed
# ConnectorBase removed
# DisableKGConfigRequest removed
# EnableKGConfigRequest removed
# EntityType removed
# KGConfig removed
# SourceAndEntityTypeView removed
from tests.integration.common_utils.constants import API_SERVER_URL  # noqa: E402
from tests.integration.common_utils.managers.user import UserManager  # noqa: E402
from tests.integration.common_utils.reset import reset_all  # noqa: E402


@pytest.fixture(autouse=True)
def reset_for_test() -> None:
    """Reset all data before each test."""
    reset_all()

    kg_config_settings = get_kg_config_settings()  # noqa: F821,F841
    kg_config_settings.KG_EXPOSED = True
    set_kg_config_settings(kg_config_settings)  # noqa: F821,F841


@pytest.fixture()
def connectors() -> None:
    """Set up connectors for tests."""
    with get_session_with_current_tenant() as db_session:  # noqa: F821,F841
        # Create Salesforce connector
        connector_data = ConnectorBase(  # noqa: F821,F841
            name="Salesforce Test",
            source=DocumentSource.SALESFORCE,
            input_type=InputType.POLL,
            connector_specific_config={},
            refresh_freq=None,
            indexing_start=None,
            prune_freq=None,
        )
        create_connector(db_session, connector_data)  # noqa: F821,F841


def test_kg_enable_and_disable(connectors: None) -> None:  # noqa: ARG001
    admin_user = UserManager.create(name="admin_user")

    # Enable KG
    # Need to `.model_dump_json()` and then `json.loads`.
    # Seems redundant, but this is because simply calling `json=data.model_dump()`
    # returns in a "datetime cannot be JSON serialized error".
    req1 = json.loads(
        EnableKGConfigRequest(  # noqa: F821,F841
            vendor="Test",
            vendor_domains=["test.app", "tester.ai"],
            ignore_domains=[],
            coverage_start=datetime(1970, 1, 1, 0, 0),
        ).model_dump_json()
    )
    res1 = requests.put(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
        json=req1,
    )
    assert res1.status_code == HTTPStatus.OK, (
        f"Error response: {res1.status_code} - {res1.text}"
    )

    # Check KG
    res2 = requests.get(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
    )
    assert res2.status_code == HTTPStatus.OK, (
        f"Error response: {res2.status_code} - {res2.text}"
    )

    actual_config = KGConfigAPIModel.model_validate_json(res2.text)  # noqa: F821,F841
    assert actual_config == KGConfigAPIModel(  # noqa: F821,F841
        enabled=True,
        vendor="Test",
        vendor_domains=["test.app", "tester.ai"],
        ignore_domains=[],
        coverage_start=datetime(1970, 1, 1, 0, 0),
    )

    # Disable KG
    res3 = requests.put(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
        json=req3,  # noqa: F821,F841
    )
    assert res3.status_code == HTTPStatus.OK, (
        f"Error response: {res3.status_code} - {res3.text}"
    )

    # Check KG
    res4 = requests.get(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
    )
    assert res4.status_code == HTTPStatus.OK, (
        f"Error response: {res4.status_code} - {res4.text}"
    )

    actual_config = KGConfigAPIModel.model_validate_json(res4.text)  # noqa: F821,F841
    assert actual_config == KGConfigAPIModel(  # noqa: F821,F841
        enabled=False,
        vendor="Test",
        vendor_domains=["test.app", "tester.ai"],
        ignore_domains=[],
        coverage_start=datetime(1970, 1, 1, 0, 0),
    )


def test_kg_enable_with_missing_fields_should_fail() -> None:
    admin_user = UserManager.create(name="admin_user")

    req = json.loads(
        EnableKGConfigRequest(  # noqa: F821,F841
            vendor="Test",
            vendor_domains=[],
            ignore_domains=[],
            coverage_start=datetime(1970, 1, 1, 0, 0),
        ).model_dump_json()
    )
    res = requests.put(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
        json=req,
    )
    assert res.status_code == HTTPStatus.BAD_REQUEST


def test_update_kg_entity_types(connectors: None) -> None:  # noqa: ARG001
    admin_user = UserManager.create(name="admin_user")

    # Enable kg and populate default entity types
    req1 = json.loads(
        EnableKGConfigRequest(  # noqa: F821,F841
            vendor="Test",
            vendor_domains=["test.app", "tester.ai"],
            ignore_domains=[],
            coverage_start=datetime(1970, 1, 1, 0, 0),
        ).model_dump_json()
    )
    res1 = requests.put(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
        json=req1,
    )
    assert res1.status_code == HTTPStatus.OK, (
        f"Error response: {res1.status_code} - {res1.text}"
    )

    # Get old entity types
    res2 = requests.get(
        f"{API_SERVER_URL}/admin/kg/entity-types",
        headers=admin_user.headers,
    )
    assert res2.status_code == HTTPStatus.OK, (
        f"Error response: {res2.status_code} - {res2.text}"
    )
    res2_parsed = SourceAndEntityTypeView.model_validate(res2.json())  # noqa: F821,F841

    # Update entity types
    req3 = [
        EntityType(  # noqa: F821,F841
            name="ACCOUNT",
            description="Test.",
            active=True,
            grounded_source_name="salesforce",
        ).model_dump(),
        EntityType(  # noqa: F821,F841
            name="OPPORTUNITY",
            description="Test 2.",
            active=False,
        ).model_dump(),
    ]
    res3 = requests.put(
        f"{API_SERVER_URL}/admin/kg/entity-types",
        headers=admin_user.headers,
        json=req3,
    )
    assert res3.status_code == HTTPStatus.OK, (
        f"Error response: {res3.status_code} - {res3.text}"
    )

    # Check connector kg_processing is enabled
    with get_session_with_current_tenant() as db_session:  # noqa: F821,F841
        connector = (
            db_session.query(Connector)  # noqa: F821,F841
            .filter(Connector.name == "Salesforce Test")  # noqa: F821,F841
            .scalar()
        )
        assert connector.kg_processing_enabled

    # Check entity types looks correct
    res4 = requests.get(
        f"{API_SERVER_URL}/admin/kg/entity-types",
        headers=admin_user.headers,
    )
    assert res4.status_code == HTTPStatus.OK, (
        f"Error response: {res4.status_code} - {res4.text}"
    )
    res4_parsed = SourceAndEntityTypeView.model_validate(res4.json())  # noqa: F821,F841

    def to_entity_type_map(map: dict[str, list[EntityType]]) -> dict[str, EntityType]:  # noqa: F821,F841
        return {
            entity_type.name: entity_type
            for entity_types in map.values()
            for entity_type in entity_types
        }

    expected_entity_types = to_entity_type_map(map=res2_parsed.entity_types)
    new_entity_types = to_entity_type_map(map=res4_parsed.entity_types)

    # These are the updates.
    # We're just manually updating them.
    expected_entity_types["ACCOUNT"].active = True
    expected_entity_types["ACCOUNT"].description = "Test."
    expected_entity_types["OPPORTUNITY"].active = False
    expected_entity_types["OPPORTUNITY"].description = "Test 2."

    assert new_entity_types == expected_entity_types


def test_update_invalid_kg_entity_type_should_do_nothing(
    connectors: None,  # noqa: ARG001
) -> None:
    admin_user = UserManager.create(name="admin_user")

    # Enable kg and populate default entity types
    req1 = json.loads(
        EnableKGConfigRequest(  # noqa: F821,F841
            vendor="Test",
            vendor_domains=["test.app", "tester.ai"],
            ignore_domains=[],
            coverage_start=datetime(1970, 1, 1, 0, 0),
        ).model_dump_json()
    )
    res1 = requests.put(
        f"{API_SERVER_URL}/admin/kg/config",
        headers=admin_user.headers,
        json=req1,
    )
    assert res1.status_code == HTTPStatus.OK, (
        f"Error response: {res1.status_code} - {res1.text}"
    )

    # Get old entity types
    res2 = requests.get(
        f"{API_SERVER_URL}/admin/kg/entity-types",
        headers=admin_user.headers,
    )
    assert res2.status_code == HTTPStatus.OK, (
        f"Error response: {res2.status_code} - {res2.text}"
    )

    # Update entity types with non-existent entity type
    req3 = [
        EntityType(name="NON-EXISTENT", description="Test.", active=False).model_dump(),  # noqa: F821,F841
    ]
    res3 = requests.put(
        f"{API_SERVER_URL}/admin/kg/entity-types",
        headers=admin_user.headers,
        json=req3,
    )
    assert res3.status_code == HTTPStatus.OK, (
        f"Error response: {res3.status_code} - {res3.text}"
    )

    # Get entity types after the update attempt
    res4 = requests.get(
        f"{API_SERVER_URL}/admin/kg/entity-types",
        headers=admin_user.headers,
    )
    assert res4.status_code == HTTPStatus.OK, (
        f"Error response: {res4.status_code} - {res4.text}"
    )

    # Should be the same as before since non-existent entity type should be ignored
    assert res2.json() == res4.json()
