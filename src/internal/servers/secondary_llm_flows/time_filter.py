from __future__ import annotations

import logging
from datetime import datetime
from datetime import timezone

from dateutil.parser import parse

from src.internal.llm.interfaces import LLM

logger = logging.getLogger(__name__)


def best_match_time(time_str: str) -> datetime | None:
    preferred_formats = ["%m/%d/%Y", "%m-%d-%Y"]

    for fmt in preferred_formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        dt = parse(time_str)
        return (
            dt.astimezone(timezone.utc)
            if dt.tzinfo
            else dt.replace(tzinfo=timezone.utc)
        )
    except ValueError:
        return None


def extract_time_filter(query: str, llm: LLM) -> tuple[datetime | None, bool]:
    """Returns a datetime if a hard time filter should be applied for the given query.
    Additionally returns a bool, True if more recently updated Documents should be heavily favored.
    """
    raise NotImplementedError("This function should not be getting called right now")
