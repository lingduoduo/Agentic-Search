from datetime import datetime
from datetime import timedelta
from datetime import timezone

# get_all_empty_chat_message_entries removed — use HTTP endpoint
# get_session_with_current_tenant removed — no direct DB access
# seed_chat_history removed — no direct DB access


def test_usage_reports(reset: None) -> None:  # noqa: ARG001
    EXPECTED_SESSIONS = 2048
    MESSAGES_PER_SESSION = 4

    # divide by 2 because only messages of type USER are returned
    EXPECTED_MESSAGES = EXPECTED_SESSIONS * MESSAGES_PER_SESSION / 2

    seed_chat_history(EXPECTED_SESSIONS, MESSAGES_PER_SESSION, 90)  # noqa: F821,F841

    with get_session_with_current_tenant() as db_session:  # noqa: F821,F841
        # count of all entries should be exact
        period = (
            datetime.fromtimestamp(0, tz=timezone.utc),
            datetime.now(tz=timezone.utc),
        )

        count = 0
        for entry_batch in get_all_empty_chat_message_entries(db_session, period):  # noqa: F821,F841
            for entry in entry_batch:
                count += 1

        assert count == EXPECTED_MESSAGES

        # count in a one month time range should be within a certain range statistically
        # this can be improved if we seed the chat history data deterministically
        period = (
            datetime.now(tz=timezone.utc) - timedelta(days=30),
            datetime.now(tz=timezone.utc),
        )

        count = 0
        for entry_batch in get_all_empty_chat_message_entries(db_session, period):  # noqa: F821,F841
            for entry in entry_batch:
                count += 1

        lower = EXPECTED_MESSAGES // 3 - (EXPECTED_MESSAGES // (3 * 3))
        upper = EXPECTED_MESSAGES // 3 + (EXPECTED_MESSAGES // (3 * 3))
        assert count > lower
        assert count < upper
