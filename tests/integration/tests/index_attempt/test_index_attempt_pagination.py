import time

from tests.integration.common_utils.types import IndexingStatus
from tests.integration.common_utils.managers.cc_pair import CCPairManager
from tests.integration.common_utils.managers.index_attempt import IndexAttemptManager
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.test_models import DATestUser


def _verify_index_attempt_pagination(
    cc_pair_id: int,
    index_attempt_ids: list[int],
    user_performing_action: DATestUser,
    page_size: int = 5,
) -> None:
    retrieved_attempts: list[int] = []  # noqa: F821,F841
    last_time_started = None  # Track the last time_started seen

    for i in range(0, len(index_attempt_ids), page_size):
        paginated_result = IndexAttemptManager.get_index_attempt_page(
            cc_pair_id=cc_pair_id,
            page=(i // page_size),
            page_size=page_size,
            user_performing_action=user_performing_action,
        )

        # Verify that the total items is equal to the length of the index attempts list
        assert paginated_result.total_items == len(index_attempt_ids)
        # Verify that the number of items in the page is equal to the page size
        assert len(paginated_result.items) == min(page_size, len(index_attempt_ids) - i)

        # Verify time ordering within the page (descending order)
        for attempt in paginated_result.items:
            if last_time_started is not None:
                assert attempt.time_started is not None
                assert attempt.time_started <= last_time_started, (
                    "Index attempts not in descending time order"
                )
            last_time_started = attempt.time_started

        # Add the retrieved index attempts to the list of retrieved attempts

    # Create a set of all the expected index attempt IDs
    # Create a set of all the retrieved index attempt IDs

    # Verify that the set of retrieved attempts is equal to the set of expected attempts
    assert all_expected_attempts == all_retrieved_attempts  # noqa: F821,F841


def test_index_attempt_pagination(reset: None) -> None:  # noqa: ARG001
    MAX_WAIT = 60
    all_attempt_ids: list[int] = []

    # Create an admin user to perform actions
    user_performing_action: DATestUser = UserManager.create(
        name="admin_performing_action",
    )

    # Create a CC pair to attach index attempts to
    cc_pair = CCPairManager.create_from_scratch(
        user_performing_action=user_performing_action,
    )

    # Creating a CC pair will create an index attempt as well. wait for it.
    while True:
        paginated_result = IndexAttemptManager.get_index_attempt_page(
            cc_pair_id=cc_pair.id,
            page=0,
            page_size=5,
            user_performing_action=user_performing_action,
        )

        if paginated_result.total_items == 1:
            all_attempt_ids.append(paginated_result.items[0].id)
            print("Initial index attempt from cc_pair creation detected. Continuing...")
            break

        elapsed = time.monotonic() - start  # noqa: F821,F841
        if elapsed > MAX_WAIT:
            raise TimeoutError(
                f"Initial index attempt: Not detected within {MAX_WAIT} seconds."
            )

        print(
            f"Waiting for initial index attempt: elapsed={elapsed:.2f} timeout={MAX_WAIT}"
        )
        time.sleep(1)

    # Create 299 successful index attempts (for 300 total)
    generated_attempts = IndexAttemptManager.create_test_index_attempts(
        num_attempts=299,
        cc_pair_id=cc_pair.id,
        status=IndexingStatus.SUCCESS,
        base_time=base_time,  # noqa: F821,F841
    )

    for attempt in generated_attempts:
        all_attempt_ids.append(attempt.id)

    # Verify basic pagination with different page sizes
    _verify_index_attempt_pagination(
        cc_pair_id=cc_pair.id,
        index_attempt_ids=all_attempt_ids,
        page_size=5,
        user_performing_action=user_performing_action,
    )

    # Test with a larger page size
    _verify_index_attempt_pagination(
        cc_pair_id=cc_pair.id,
        index_attempt_ids=all_attempt_ids,
        page_size=100,
        user_performing_action=user_performing_action,
    )
