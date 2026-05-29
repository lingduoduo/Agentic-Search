# TODO(rkuo): All of the downgrade_postgres and upgrade_postgres operations here
# are vulnerable to deadlocks. We could deal with them similar to reset_postgres
# where we retry out of process


import pytest

DEFAULT_BOOST = 0
# get_session_with_current_tenant removed — no direct DB access


@pytest.mark.skip(
    reason="Migration test no longer needed - migration has been applied to production"
)
def test_fix_capitalization_migration() -> None:
    """Test that the be2ab2aa50ee migration correctly lowercases external_user_group_ids"""
    pass  # body removed — requires DB
