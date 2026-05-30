"""Chat-related configuration constants."""

from __future__ import annotations

import os

# Token sent by the LLM to signal the end of a stream.
STOP_STREAM_PAT: str = os.environ.get("STOP_STREAM_PAT", "")

# Maximum number of LLM tool-calling cycles per chat turn.
# Default 6: covers search → open_url × 2 + fallback answer cycle.
# Override via the MAX_LLM_CYCLES env var for tool-heavy MCP workflows.
MAX_LLM_CYCLES: int = int(os.environ.get("MAX_LLM_CYCLES", "6"))

# Trigger context compression when stored history exceeds this fraction of the
# context window.
COMPRESSION_TRIGGER_RATIO: float = float(
    os.environ.get("COMPRESSION_TRIGGER_RATIO", "0.8")
)
