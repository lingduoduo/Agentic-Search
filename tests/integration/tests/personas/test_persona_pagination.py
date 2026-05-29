import requests

ADMIN_AGENTS_RESOURCE = "admin-agents"
AGENTS_RESOURCE = "agents"
from tests.integration.common_utils.constants import API_SERVER_URL  # noqa: E402
from tests.integration.common_utils.managers.persona import PersonaManager  # noqa: E402
from tests.integration.common_utils.test_models import DATestUser  # noqa: E402


def _get_agents_paginated(
    user: DATestUser,
    page_num: int,
    page_size: int,
    include_deleted: bool = False,
    get_editable: bool = False,
    include_default: bool = True,
) -> tuple[dict, int]:
    """Fetches a paginated page of agents, with status code."""
    response = requests.get(
        f"{API_SERVER_URL}{AGENTS_RESOURCE}",
        params={
            "page_num": page_num,
            "page_size": page_size,
            "include_deleted": include_deleted,
            "get_editable": get_editable,
            "include_default": include_default,
        },
        headers=user.headers,
        cookies=user.cookies,
    )
    return response.json(), response.status_code


def _get_agents_admin_paginated(
    user: DATestUser,
    page_num: int,
    page_size: int,
    include_deleted: bool = False,
    get_editable: bool = False,
    include_default: bool = True,
) -> tuple[dict, int]:
    """Fetches a paginated page of agents (admin endpoint) with status code."""
    response = requests.get(
        f"{API_SERVER_URL}{ADMIN_AGENTS_RESOURCE}",
        params={
            "page_num": page_num,
            "page_size": page_size,
            "include_deleted": include_deleted,
            "get_editable": get_editable,
            "include_default": include_default,
        },
        headers=user.headers,
        cookies=user.cookies,
    )
    response.raise_for_status()
    return response.json(), response.status_code


def test_persona_pagination_basic(
    reset: None,  # noqa: ARG001
    admin_user: DATestUser,
) -> None:
    """Test basic pagination - verify correct items and total count."""
    # Preconditions
    personas_to_create = 25
    personas = []
    for i in range(personas_to_create):
        persona = PersonaManager.create(
            name=f"Test Persona {i}",
            user_performing_action=admin_user,
        )
        personas.append(persona)

    # Under test and postconditions
    # Test page 0 with size 10.
    assert "items" in page_0  # noqa: F821,F841
    assert "total_items" in page_0  # noqa: F821,F841
    assert len(page_0["items"]) == 10  # noqa: F821,F841
    assert (
        page_0["total_items"] >= personas_to_create  # noqa: F821,F841
    )  # At least personas_to_create (may have default personas)

    # Test page 2 with size 10 (should have 5+ items if only our test personas
    # exist).
    assert len(page_2["items"]) >= 5  # noqa: F821,F841
    assert page_2["total_items"] >= personas_to_create  # noqa: F821,F841

    # Test page beyond end (page 10 with size 10, offset 100).
    assert len(page_beyond["items"]) == 0  # noqa: F821,F841
    assert (
        page_beyond["total_items"] >= personas_to_create  # noqa: F821
    )  # Total doesn't change.  # noqa: F821,F841


def test_persona_pagination_ordering(
    admin_user: DATestUser,
) -> None:
    """Test ordering - display_priority ASC nulls last, then ID ASC."""
    # Preconditions
    # Create personas with specific display_priority values.
    persona_a = PersonaManager.create(
        name="Persona A",
        description="This should be second",
        user_performing_action=admin_user,
        display_priority=2,
    )
    persona_b = PersonaManager.create(
        name="Persona B",
        description="This should be first",
        user_performing_action=admin_user,
        display_priority=1,
    )
    persona_c = PersonaManager.create(
        name="Persona C",
        description="This should be third",
        user_performing_action=admin_user,
        display_priority=3,
    )
    persona_d = PersonaManager.create(
        name="Persona D",
        description="This should be fourth",
        user_performing_action=admin_user,
        display_priority=3,  # Note the same prio as above, should sort by id
    )

    # Under test

    # Postconditions
    # Find our personas in the results.
    our_expected_ordered_persona_ids = [
        persona_b.id,
        persona_a.id,
        persona_c.id,
        persona_d.id,
    ]
    our_personas_in_results = [
        p
        for p in page_0["items"]  # noqa: F821
        if p["id"] in our_expected_ordered_persona_ids  # noqa: F821,F841
    ]
    assert len(our_personas_in_results) == 4
    # Verify ordering.
    for i in range(len(our_expected_ordered_persona_ids)):
        assert our_expected_ordered_persona_ids[i] == our_personas_in_results[i]["id"]


def test_persona_pagination_admin_endpoint(
    admin_user: DATestUser,
) -> None:
    """Test admin paginated endpoint returns PersonaSnapshot format."""
    # Preconditions
    personas_to_create = 5
    for i in range(personas_to_create):
        PersonaManager.create(
            name=f"Admin Test Persona {i}",
            user_performing_action=admin_user,
        )

    # Under test

    # Postconditions
    assert "items" in page_0  # noqa: F821,F841
    assert "total_items" in page_0  # noqa: F821,F841
    assert len(page_0["items"]) >= personas_to_create  # noqa: F821,F841
    assert page_0["total_items"] >= personas_to_create  # noqa: F821,F841
    # Verify admin-specific fields are present (PersonaSnapshot has more
    # fields).
    first_persona = page_0["items"][0]  # noqa: F821,F841
    # PersonaSnapshot should have these fields that MinimalPersonaSnapshot
    # doesn't.
    assert "users" in first_persona
    assert "groups" in first_persona
    assert "user_file_ids" in first_persona


def test_persona_pagination_with_deleted(
    admin_user: DATestUser,
) -> None:
    """Test pagination with include_deleted parameter."""
    # Preconditions
    # Create and delete a persona.
    persona = PersonaManager.create(
        name="To Be Deleted",
        user_performing_action=admin_user,
    )
    PersonaManager.delete(persona, user_performing_action=admin_user)

    # Under test and postconditions
    # Without include_deleted, should not appear.
    page_without_deleted, _ = _get_agents_paginated(
        admin_user, page_num=0, page_size=100, include_deleted=False
    )
    persona_ids_without_deleted = [p["id"] for p in page_without_deleted["items"]]
    assert persona.id not in persona_ids_without_deleted

    # With include_deleted, should appear.
    page_with_deleted, _ = _get_agents_paginated(
        admin_user, page_num=0, page_size=100, include_deleted=True
    )
    persona_ids_with_deleted = [p["id"] for p in page_with_deleted["items"]]
    assert persona.id in persona_ids_with_deleted

    # Total counts should differ.
    assert page_with_deleted["total_items"] > page_without_deleted["total_items"]


def test_persona_pagination_page_size_limits(
    reset: None,  # noqa: ARG001
    admin_user: DATestUser,
) -> None:
    """Test page_size parameter validation (max 1000)."""
    # Preconditions
    # Create a few personas.
    for i in range(5):
        PersonaManager.create(
            name=f"Size Limit Test {i}",
            user_performing_action=admin_user,
        )

    # Under test and postconditions
    # Valid page_size of 1
    assert len(data["items"]) <= 1  # noqa: F821,F841

    # Valid page_size of 1000
    # We assume not that many default personas are made.
    assert len(data["items"]) == data["total_items"]  # noqa: F821,F841

    # Invalid page_size of 1001 (exceeds max)
    assert status_code == 422  # Validation error  # noqa: F821,F841

    # Invalid page_size of 0
    assert status_code == 422  # Validation error  # noqa: F821,F841


def test_persona_pagination_count_accuracy(
    reset: None,  # noqa: ARG001
    admin_user: DATestUser,
) -> None:
    """Test that total_items count is consistent across pages."""
    # Preconditions
    # Create 15 personas.
    created_personas = []
    for i in range(15):
        persona = PersonaManager.create(
            name=f"Count Test {i}",
            user_performing_action=admin_user,
        )
        created_personas.append(persona)

    # Under test and postconditions
    # Fetch first page to get total count.
    total_items = page_0["total_items"]  # noqa: F821,F841
    assert total_items >= 15

    # Fetch all pages to cover all personas.
    num_pages_needed = (total_items + 4) // 5  # Ceiling division
    for page_num in range(num_pages_needed):
        page, _ = _get_agents_paginated(admin_user, page_num=page_num, page_size=5)
        # All pages should report the same total.
        assert page["total_items"] == total_items, (
            f"Page {page_num} has inconsistent total_items"
        )
        all_ids_from_pages.update(p["id"] for p in page["items"])  # noqa: F821,F841

    # Our created personas should all appear.
    our_ids = {p.id for p in created_personas}
    assert our_ids.issubset(all_ids_from_pages), (  # noqa: F821,F841
        "All created personas should appear in paginated results"
    )


def test_persona_pagination_user_permissions(
    admin_user: DATestUser,
    basic_user: DATestUser,
) -> None:
    """Test that pagination respects user permissions."""
    # Preconditions
    # Admin creates a private persona (not shared).
    private_persona = PersonaManager.create(
        name="Private Persona",
        description="Not shared",
        is_public=False,
        user_performing_action=admin_user,
    )
    # Admin creates a public persona.
    public_persona = PersonaManager.create(
        name="Public Persona",
        description="Shared with all",
        is_public=True,
        user_performing_action=admin_user,
    )

    # Under test and postconditions
    # Admin should see both in paginated results.
    admin_ids = {p["id"] for p in admin_page["items"]}  # noqa: F821,F841
    assert private_persona.id in admin_ids
    assert public_persona.id in admin_ids

    # Basic user should only see public persona.
    user_ids = {p["id"] for p in user_page["items"]}  # noqa: F821,F841
    assert private_persona.id not in user_ids
    assert public_persona.id in user_ids

    # Totals should differ.
    assert admin_page["total_items"] > user_page["total_items"]  # noqa: F821,F841
