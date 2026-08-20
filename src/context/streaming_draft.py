"""Incremental reader for a partially-received answer draft.

Assist's grounded path streams a JSON draft whose claims are shown only after
they verify. This reader extracts complete claim objects from the partial text so
each can be verified and emitted on arrival.

It is deliberately advisory. It never decides the final answer -- that always
comes from the whole-text `parse_answer_draft`. On anything it does not fully
recognise it gives up permanently and the run reverts to non-streaming
behaviour, because emitting a claim that later proves wrong is worse than
emitting nothing.
"""

from __future__ import annotations

import json
import re

from .models import AnswerClaim
from .safety import build_claim

_ABSTAIN = re.compile(r'"abstain"\s*:\s*(true|false)')
_CLAIMS = re.compile(r'"claims"\s*:\s*\[')


def _next_object(text: str, start: int) -> tuple[str, int] | None:
    """Return (object_text, end_index) for the first complete {...} at or after start.

    Returns None when the text is merely incomplete, so the caller waits for more.
    String-aware, so braces and escaped quotes inside claim text do not confuse
    the depth count.
    """
    index = start
    length = len(text)
    while index < length and text[index] in ", \t\r\n":
        index += 1
    if index >= length or text[index] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False
    cursor = index
    while cursor < length:
        char = text[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[index : cursor + 1], cursor + 1
        cursor += 1
    return None


class IncrementalDraftReader:
    """Yield complete claims from a draft as it streams in."""

    def __init__(self, evidence_ids: set[str]) -> None:
        self._evidence_ids = evidence_ids
        self._buffer = ""
        self._abstain: bool | None = None
        self._cursor: int | None = None
        self._gave_up = False

    @property
    def abstain(self) -> bool | None:
        return self._abstain

    @property
    def gave_up(self) -> bool:
        return self._gave_up

    def feed(self, chunk: str) -> list[AnswerClaim]:
        if self._gave_up:
            return []
        self._buffer += chunk

        if self._abstain is None:
            match = _ABSTAIN.search(self._buffer)
            if match is None:
                # Claims already opened but abstain is still unknown: the draft
                # is in the old key order, and a claim we emit now could be
                # discarded by an abstain we have not read yet.
                if _CLAIMS.search(self._buffer):
                    self._gave_up = True
                return []
            self._abstain = match.group(1) == "true"
            if self._abstain:
                # Every claim is discarded, so there is nothing to stream.
                self._gave_up = True
                return []

        if self._cursor is None:
            opening = _CLAIMS.search(self._buffer)
            if opening is None:
                # Only give up once we can tell this is not a draft at all;
                # a bare prefix may simply be incomplete.
                if self._buffer.lstrip()[:1] not in ("", "{"):
                    self._gave_up = True
                return []
            self._cursor = opening.end()

        claims: list[AnswerClaim] = []
        while True:
            found = _next_object(self._buffer, self._cursor)
            if found is None:
                return claims
            object_text, self._cursor = found
            try:
                claims.append(self._build_claim(object_text))
            except ValueError:
                self._gave_up = True
                return claims

    def _build_claim(self, object_text: str) -> AnswerClaim:
        """Parse one claim object and delegate validation to `safety.build_claim`.

        Using the same function `parse_answer_draft` uses is what keeps this
        reader from ever showing a claim that the authoritative whole-text parse
        would later reject. `json.JSONDecodeError` is a `ValueError` subclass,
        so the caller's single `except ValueError` also covers malformed JSON.
        """
        value = json.loads(object_text)
        return build_claim(value, self._evidence_ids)
