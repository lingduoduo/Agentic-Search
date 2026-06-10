import os

ONYX_BOT_NUM_RETRIES = int(os.environ.get("ONYX_BOT_NUM_RETRIES", "5"))
ONYX_BOT_NUM_DOCS_TO_DISPLAY = int(os.environ.get("ONYX_BOT_NUM_DOCS_TO_DISPLAY", "5"))
ONYX_BOT_DISABLE_DOCS_ONLY_ANSWER = os.environ.get(
    "ONYX_BOT_DISABLE_DOCS_ONLY_ANSWER", ""
).lower() not in ["false", ""]
ONYX_BOT_REACT_EMOJI = os.environ.get("ONYX_BOT_REACT_EMOJI") or "eyes"
ONYX_BOT_FOLLOWUP_EMOJI = os.environ.get("ONYX_BOT_FOLLOWUP_EMOJI") or "sos"
ONYX_BOT_FEEDBACK_VISIBILITY = (
    os.environ.get("ONYX_BOT_FEEDBACK_VISIBILITY") or "private"
)
NOTIFY_SLACKBOT_NO_ANSWER = (
    os.environ.get("NOTIFY_SLACKBOT_NO_ANSWER", "").lower() == "true"
)
ONYX_BOT_DISPLAY_ERROR_MSGS = os.environ.get(
    "ONYX_BOT_DISPLAY_ERROR_MSGS", ""
).lower() not in ["false", ""]

ONYX_BOT_MAX_QPM = int(os.environ.get("ONYX_BOT_MAX_QPM") or 0) or None
ONYX_BOT_MAX_WAIT_TIME = int(os.environ.get("ONYX_BOT_MAX_WAIT_TIME") or 180)

ONYX_BOT_FEEDBACK_REMINDER = int(os.environ.get("ONYX_BOT_FEEDBACK_REMINDER") or 0)

ONYX_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD = int(
    os.environ.get("ONYX_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD", "5000")
)
ONYX_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS = int(
    os.environ.get("ONYX_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS", "86400")
)
